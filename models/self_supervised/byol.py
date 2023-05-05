from copy import deepcopy
from os.path import join

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import BYOL

from models.encoders import SiameseArm
from models.self_supervised.utils import (
    validation_metrics,
    init_model,
    roc_auc,
    configure_optimizers, compute_num_samples
)


class BYOLModel(BYOL):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        self.save_hyperparameters()
        self.config = config

        super().__init__(
            num_classes=config.num_classes,
            learning_rate=self.config.ssl.learning_rate,
            weight_decay=self.config.ssl.weight_decay,
            input_height=self.config.ssl.input_height,
            batch_size=self.config.hyper_parameters.batch_size,
            num_workers=self.config.ssl.num_workers,
            warmup_epochs=self.config.ssl.warmup_epochs,
            max_epochs=config.hyper_parameters.n_epochs,
            **kwargs
        )

        self._init_encoders()

        train_data_path = join(
            config.data_folder,
            config.dataset.name,
            "raw",
            config.train_holdout
        )

        self.train_iters_per_epoch = compute_num_samples(train_data_path) // self.config.hyper_parameters.batch_size

    def _init_encoders(self):
        encoder = init_model(self.config)
        self.online_network = SiameseArm(
            encoder=encoder,
            input_dim=self.config.num_classes,
            output_dim=self.config.num_classes,
            hidden_size=self.config.num_classes
        )
        self.target_network = deepcopy(self.online_network)

    def forward(self, x):
        *_, x = self.online_network(x)
        x = F.normalize(x, dim=1)
        return x

    def _loss(self, h1_1, h2_1, z1_2, z2_2):
        loss_a = -2 * F.cosine_similarity(h1_1, z1_2).mean()
        loss_b = -2 * F.cosine_similarity(h2_1, z2_2).mean()
        return loss_a + loss_b

    def representation(self, q, k):
        # Image 1 to image 2 loss
        y1_1, z1_1, h1_1 = self.online_network(q)
        with torch.no_grad():
            y1_2, z1_2, h1_2 = self.target_network(k)

        # Image 2 to image 1 loss
        y2_1, z2_1, h2_1 = self.online_network(k)
        with torch.no_grad():
            y2_2, z2_2, h2_2 = self.target_network(q)

        return h1_1, h2_1, z1_2, z2_2

    def training_step(self, batch, batch_idx):
        (q, k, _), labels = batch

        h1_1, h2_1, z1_2, z2_2 = self.representation(q=q, k=k)
        loss = self._loss(h1_1, h2_1, z1_2, z2_2)
        queries, keys = h1_1, h2_1

        roc_auc_ = roc_auc(queries=queries, keys=keys, labels=labels)
        self.log_dict({"train_loss": loss, "train_roc_auc": roc_auc_})
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        features = self(features)
        labels = labels.contiguous().view(-1, 1)

        return {"features": features, "labels": labels}

    def validation_epoch_end(self, outputs):
        log = validation_metrics(outputs, task=self.config.dataset.name)
        self.log_dict(log)

    def configure_optimizers(self):
        return configure_optimizers(
            self,
            learning_rate=self.config.ssl.learning_rate,
            weight_decay=self.config.ssl.weight_decay,
            warmup_epochs=self.config.ssl.warmup_epochs,
            max_epochs=self.config.hyper_parameters.n_epochs,
            exclude_bn_bias=self.config.ssl.exclude_bn_bias,
            train_iters_per_epoch=self.train_iters_per_epoch
        )


class BYOLTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y

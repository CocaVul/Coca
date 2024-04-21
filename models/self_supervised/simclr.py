from os.path import join

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import SimCLR

from models.self_supervised.utils import (
    validation_metrics,
    prepare_features,
    clone_classification_step,
    compute_num_samples,
    init_model,
    roc_auc, configure_optimizers
)


class SimCLRModel(SimCLR):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        self.save_hyperparameters()
        self.config = config
        self.base_encoder = config.name
        train_data_path = join(
            config.data_folder,
            config.dataset.name,
            "raw",
            config.train_holdout
        )

        num_samples = compute_num_samples(train_data_path)

        super().__init__(
            gpus=config.ssl.gpus,
            num_nodes=config.ssl.num_nodes,
            batch_size=config.hyper_parameters.batch_size,
            max_epochs=config.hyper_parameters.n_epochs,
            hidden_mlp=config.num_classes,
            feat_dim=config.num_classes,
            temperature=config.ssl.temperature,
            warmup_epochs=config.ssl.warmup_epochs,
            start_lr=config.ssl.start_lr,
            learning_rate=config.ssl.learning_rate,
            weight_decay=config.ssl.weight_decay,
            exclude_bn_bias=config.ssl.exclude_bn_bias,
            num_samples=num_samples,
            dataset="",
            **kwargs
        )

    def init_model(self):
        encoder = init_model(self.config)
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, dim=1)
        return x

    def _loss(self, logits, mask):
        batch_size = mask.shape[0] // 2

        # compute logits
        anchor_dot_contrast = logits / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=self.device),
            1,
            torch.arange(2 * batch_size, device=self.device).view(-1, 1),
            0
        )
        mask_ = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
        loss = -mean_log_prob_pos.view(2, batch_size).mean()

        return loss

    def training_step(self, batch, batch_idx):
        (q, k, _), labels = batch
        queries, keys = self(q), self(k)

        # get z representations
        z1 = self.projection(queries)
        z2 = self.projection(keys)

        embeddings, loss_labels = prepare_features(z1, z2, labels)
        loss_logits, loss_mask = clone_classification_step(embeddings, loss_labels)
        loss = self._loss(loss_logits, loss_mask)

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

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs=outputs)

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


class SimCLRTransform:
    def __call__(self, batch):
        (x1, x2), y = batch
        return (x1, x2, None), y

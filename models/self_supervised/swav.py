from os.path import join

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised.swav.swav_resnet import MultiPrototypes
from torch import nn

from models.self_supervised.utils import (
    validation_metrics,
    compute_num_samples,
    init_model,
    prepare_features,
    roc_auc,
    configure_optimizers
)


class SwAVModel(SwAV):
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
            epsilon=config.ssl.epsilon,
            hidden_mlp=config.ssl.hidden_mlp,
            feat_dim=config.ssl.feat_dim,
            nmb_prototypes=config.ssl.nmb_prototypes,
            freeze_prototypes_epochs=config.ssl.freeze_prototypes_epochs,
            sinkhorn_iterations=config.ssl.sinkhorn_iterations,
            warmup_epochs=config.ssl.warmup_epochs,
            start_lr=config.ssl.start_lr,
            learning_rate=config.ssl.learning_rate,
            weight_decay=config.ssl.weight_decay,
            exclude_bn_bias=config.ssl.exclude_bn_bias,
            num_samples=num_samples,
            dataset="",
        )

        self.prototypes = None
        if isinstance(config.ssl.nmb_prototypes, list):
            self.prototypes = MultiPrototypes(config.ssl.feat_dim, config.ssl.nmb_prototypes)
        elif config.ssl.nmb_prototypes > 0:
            self.prototypes = torch.nn.Linear(config.ssl.feat_dim, config.ssl.nmb_prototypes, bias=False)

        # normalize output features
        self.l2norm = config.ssl.normalize

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(config.ssl.feat_dim, config.ssl.hidden_mlp),
            nn.BatchNorm1d(config.ssl.hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(config.ssl.hidden_mlp, config.num_classes),
        )

    def init_model(self):
        encoder = init_model(self.config)
        return encoder

    def forward(self, x):
        q = self.model(x)
        q = F.normalize(q, dim=1)
        return q

    def _loss(self, queries, keys, labels):
        embedding, loss_labels = prepare_features(queries, keys, labels)
        embedding = self.projection_head(embedding)
        if self.l2norm:
            embedding = F.normalize(embedding, dim=1, p=2)
        prototype = self.prototypes(embedding)
        bs = labels.shape[0]
        # prototype output: 2BxK
        scores_t = prototype[:bs]
        scores_s = prototype[bs:]

        q_t = torch.exp(scores_t / self.epsilon).t()
        q_s = torch.exp(scores_s / self.epsilon).t()

        with torch.no_grad():
            q_t = self.sinkhorn(q_t, self.sinkhorn_iterations)
            q_s = self.sinkhorn(q_s, self.sinkhorn_iterations)

        # convert scores to probabilities
        p_t = self.softmax(scores_t / self.temperature)
        p_s = self.softmax(scores_s / self.temperature)

        loss = -0.5 * torch.mean(q_t * torch.log(p_s) + q_s * torch.log(p_t))

        return loss

    def training_step(self, batch, batch_idx):
        (q, k), labels = batch

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        queries, keys = self.model(q), self.model(k)
        loss = self._loss(queries=queries, keys=keys, labels=labels)

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

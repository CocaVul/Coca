from os.path import join

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pl_bolts.models.self_supervised import Moco_v2

from models.self_supervised.utils import (
    validation_metrics,
    init_model,
    roc_auc,
    configure_optimizers, compute_num_samples
)


class MocoV2Model(Moco_v2):
    def __init__(
        self,
        config: DictConfig,
        **kwargs
    ):
        self.save_hyperparameters()
        self.config = config

        super().__init__(
            base_encoder=config.name,
            emb_dim=config.num_classes,
            num_negatives=config.ssl.num_negatives,
            encoder_momentum=config.ssl.encoder_momentum,
            softmax_temperature=config.ssl.softmax_temperature,
            learning_rate=config.ssl.learning_rate,
            weight_decay=config.ssl.weight_decay,
            use_mlp=config.ssl.use_mlp,
            batch_size=config.hyper_parameters.batch_size,
            **kwargs
        )

        # create the validation queue
        self.register_buffer("labels_queue", torch.zeros(config.ssl.num_negatives).long() - 1)

        train_data_path = join(
            config.data_folder,
            config.dataset.name,
            "raw",
            config.train_holdout
        )

        self.train_iters_per_epoch = compute_num_samples(train_data_path) // self.config.hyper_parameters.batch_size

    def init_encoders(self, base_encoder: str):
        encoder_q = init_model(self.config)
        encoder_k = init_model(self.config)
        return encoder_q, encoder_k

    def forward(self, x):
        x = self.encoder_q(x)
        x = F.normalize(x, dim=1)
        return x

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.labels_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def representation(self, q, k):
        # compute query features
        q = self.encoder_q(q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(k)  # keys: NxC
            k = F.normalize(k, dim=1)
        return q, k

    def uni_con(self, logits, target):
        sum_neg = ((1 - target) * torch.exp(logits)).sum(1)
        sum_pos = (target * torch.exp(-logits)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        return torch.mean(loss)

    def _loss(self, q, k, labels, queue, labels_queue):
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        batch_size, *_ = q.shape
        # positive label for the augmented version
        target_aug = torch.ones((batch_size, 1), device=q.device)
        # comparing the query label with l_que
        target_que = torch.eq(labels.reshape(-1, 1), labels_queue)
        target_que = target_que.float()
        # labels: Nx(1+K)
        target = torch.cat([target_aug, target_que], dim=1)
        # calculate the contrastive loss, Eqn.(7)
        loss = self.uni_con(logits=logits, target=target)
        return loss

    def training_step(self, batch, batch_idx):
        (q, k), labels = batch

        # update the key encoder
        self._momentum_update_key_encoder()

        queries, keys = self.representation(q=q, k=k)
        loss = self._loss(
            q=queries,
            k=keys,
            labels=labels,
            queue=self.queue,
            labels_queue=self.labels_queue
        )

        # dequeue and enqueue
        self._dequeue_and_enqueue(keys, labels)

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

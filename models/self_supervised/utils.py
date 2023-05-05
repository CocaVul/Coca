from os import listdir
from os.path import isdir, join

import torch
from code2seq.data.vocabulary import Vocabulary
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torch.optim import Adam
from torch_cluster import knn
from torchmetrics.functional import auroc

from models import encoder_models


def exclude_from_wt_decay(named_params, weight_decay, skip_list=("bias", "bn")):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return [{"params": params, "weight_decay": weight_decay}, {"params": excluded_params, "weight_decay": 0.0}]


def configure_optimizers(
    model,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    max_epochs: int,
    exclude_bn_bias: bool,
    train_iters_per_epoch: int
):
    if exclude_bn_bias:
        params = exclude_from_wt_decay(model.named_parameters(), weight_decay=weight_decay)
    else:
        params = model.parameters()

    optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)

    warmup_steps = train_iters_per_epoch * warmup_epochs
    total_steps = train_iters_per_epoch * max_epochs

    scheduler = {
        "scheduler": torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            linear_warmup_decay(warmup_steps, total_steps, cosine=True),
        ),
        "interval": "step",
        "frequency": 1,
    }
    return [optimizer], [scheduler]


def init_model(config):
    if config.name in ["transformer", "gnn", "code-transformer"]:
        encoder = encoder_models[config.name](config)
    elif config.name == "code2class":
        _vocabulary = Vocabulary(
            join(
                config.data_folder,
                config.dataset.name,
                config.dataset.dir,
                config.vocabulary_name
            ),
            config.dataset.max_labels,
            config.dataset.max_tokens
        )
        encoder = encoder_models[config.name](config=config, vocabulary=_vocabulary)
    else:
        raise ValueError(f"Unknown model: {config.name}")
    return encoder


@torch.no_grad()
def roc_auc(queries, keys, labels):
    features, labels = prepare_features(queries, keys, labels)
    logits, mask = clone_classification_step(features, labels)
    logits = scale(logits)
    logits = logits.reshape(-1)
    mask = mask.reshape(-1)

    return auroc(logits, mask)


def compute_f1(conf_matrix):
    assert conf_matrix.shape == (2, 2)
    tn, fn, fp, tp = conf_matrix.reshape(-1).tolist()
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1


def compute_map_at_k(preds):
    avg_precisions = []

    k = preds.shape[1]
    for pred in preds:
        positions = torch.arange(1, k + 1, device=preds.device)[pred > 0]
        if positions.shape[0]:
            avg = torch.arange(1, positions.shape[0] + 1, device=positions.device) / positions
            avg_precisions.append(avg.sum() / k)
        else:
            avg_precisions.append(torch.tensor(0.0, device=preds.device))
    return torch.stack(avg_precisions).mean().item()


def validation_metrics(outputs, task: str = "poj_104"):
    features = torch.cat([out["features"] for out in outputs])
    _, hidden_size = features.shape

    labels = torch.cat([out["labels"] for out in outputs]).reshape(-1)

    if task == "poj_104":
        ks = [100, 200, 500]
    elif task == "codeforces":
        ks = [5, 10, 15]
    else:
        raise ValueError(f"Unknown task {task}")

    logs = {}
    for k in ks:
        if k < labels.shape[0]:
            top_ids = knn(x=features, y=features, k=k + 1)
            top_ids = top_ids[1, :].reshape(-1, k + 1)
            top_ids = top_ids[:, 1:]

            top_labels = labels[top_ids]
            preds = torch.eq(top_labels, labels.reshape(-1, 1))
            logs[f"val_map@{k}"] = compute_map_at_k(preds)
    return logs


def clone_classification_step(features, labels):
    logits = torch.matmul(features, features.T)
    mask = torch.eq(labels, labels.T)
    return logits, mask


def prepare_features(queries, keys, labels):
    features = torch.cat([queries, keys], dim=0)
    labels = labels.contiguous().view(-1, 1)
    labels = labels.repeat(2, 1)
    return features, labels


def scale(x):
    x = torch.clamp(x, min=-1, max=1)
    return (x + 1) / 2


def compute_num_samples(train_data_path: str):
    num_samples = 0
    for class_ in listdir(train_data_path):
        class_path = join(train_data_path, class_)
        if isdir(class_path):
            num_files = len([_ for _ in listdir(class_path)])
            num_samples += num_files * (num_files - 1) // 2
    return num_samples

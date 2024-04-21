from tap import Tap
import random
import sys
from global_defines import vul_types, cur_dir, device, num_classes, cur_vul_type_idx
### DeepWuKong Configuration

type = vul_types[cur_vul_type_idx]


class ModelParser(Tap):
    pretrain_word2vec_model: str = f"{cur_dir}/models/{type}/word/w2v_slice.model"
    vector_size: int = 128  # 图结点的向量维度
    hidden_size: int = 128  # GNN隐层向量维度
    layer_num: int = 3  # GNN层数
    rnn_layer_num: int = 1 # RNN层数
    num_classes: int = num_classes
    model_dir = f"{cur_dir}/models/{type}/model/"
    device = device
    model_name = 'gcn'
    detector = 'dwk'


class DataParser(Tap):
    dataset_dir: str = f'{cur_dir}/datasets/{type}'

    shuffle_data: bool = True # 是否随机打乱数据集
    num_workers: int = 8

    random_split: bool = True
    batch_size: int = 64
    test_batch_size: int = 64

    device = device
    num_classes = 2


class TrainParser(Tap):
    max_epochs: int = 100
    early_stopping: int = 5
    save_epoch: int = 5
    learning_rate: float = 0.002
    weight_decay: float = 1.3e-6


random.seed(2)
model_args = ModelParser().parse_args(known_only=True)
data_args = DataParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)
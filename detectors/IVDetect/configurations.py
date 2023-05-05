import torch
from tap import Tap
import sys
sys.path.append("D:/PythonSpace/VulDetectors-master")
from global_defines import vul_types, cur_dir, device, num_classes, cur_vul_type_idx

# IVDetect Configuration
type = vul_types[cur_vul_type_idx]


class ModelParser(Tap):
    pretrain_word2vec_model: str = f'{cur_dir}/models/{type}/word/w2v_ivdetect.model'
    hidden_size: int = 128  # GNN隐层向量维度
    feature_representation_size = 128
    num_node_features: int = 5
    num_classes: int = num_classes
    num_layers: int = 3  # GNN层数
    dropout_rate: float = 0.3
    model_dir = f"{cur_dir}/models/{type}/model/"
    device = device

    model_name = 'gcn'
    detector = "ivdetect"


class DataParser(Tap):
    dataset_dir = f'{cur_dir}/datasets/{type}/'
    shuffle_data: bool = True  # 是否随机打乱数据集
    num_workers: int = 8
    random_split: bool = True
    seed: int = 2
    batch_size: int = 64
    device = device
    num_classes = 2


class TrainParser(Tap):
    max_epochs: int = 100
    early_stopping: int = 3
    save_epoch: int = 10
    learning_rate: float = 0.0001
    weight_decay: int = 0.0

    batch_size: int = 64
    test_batch_size: int = 64


model_args = ModelParser().parse_args(known_only=True)
data_args = DataParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)

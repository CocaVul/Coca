from .byol import BYOLModel, BYOLTransform
from .moco import MocoV2Model
from .simclr import SimCLRModel, SimCLRTransform
from .swav import SwAVModel

__all__ = [
    "MocoV2Model",
    "BYOLModel",
    "SimCLRModel",
    "SwAVModel",
    "ssl_models",
    "ssl_models_transforms"
]

ssl_models = {
    "MocoV2": MocoV2Model,
    "BYOL": BYOLModel,
    "SimCLR": SimCLRModel,
    "SwAV": SwAVModel
}

ssl_models_transforms = {
    "BYOL": BYOLTransform,
    "SimCLR": SimCLRTransform
}

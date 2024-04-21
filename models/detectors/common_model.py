import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


# suit the API in DIG/xgraph
class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)


class GlobalMaxPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)


class GlobalMaxMeanPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return torch.cat((global_max_pool(x, batch), global_mean_pool(x, batch)), dim=1)

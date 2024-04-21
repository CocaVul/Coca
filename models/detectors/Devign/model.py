import torch.nn as nn
import torch

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.data import Batch
from detectors.Devign.configurations import model_args, type_map
from detectors.common_model import GlobalMaxPool


# Devign分类模型
class DevignModel(nn.Module):
    def __init__(self, num_layers=1, MLP_hidden_dim=256, need_node_emb=False):
        super().__init__()
        MLP_internal_dim = int(MLP_hidden_dim / 2)
        input_dim = len(type_map) + model_args.vector_size
        self.hidden_dim = input_dim
        # GGNN层
        self.GGNN = GatedGraphConv(out_channels=input_dim, num_layers=5)
        self.readout = GlobalMaxPool()
        self.dropout_p = 0.2
        self.need_node_emb = need_node_emb

        # MLP层
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=MLP_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=MLP_hidden_dim, out_features=MLP_internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=MLP_internal_dim, out_features=MLP_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        ) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=MLP_hidden_dim, out_features=2),
        )

    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def embed_graph(self, x, edge_index, batch):
        node_emb = self.GGNN(x, edge_index)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        graph_emb = self.readout(node_emb, batch)  # [batch_size, embedding_dim]
        return graph_emb, node_emb

    def arguments_read(self, *args, **kwargs):
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=args[0].device)
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, edge_index, batch

    def forward(self, *args, **kwargs):
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        graph_emb, node_emb = self.embed_graph(x, edge_index, batch)
        feature_emb = self.extract_feature(graph_emb)
        probs = self.classifier(feature_emb)  # [batch_size, 2]
        # 返回node_emb适应PGExplainer
        if self.need_node_emb:
            return probs, node_emb
        else:
            return probs

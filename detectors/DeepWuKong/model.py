import torch
import torch.nn as nn

from torch_geometric.nn.conv import GCNConv, gcn_conv
from torch_geometric.data import Batch, Data
from detectors.DeepWuKong.configurations import model_args
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor

from detectors.common_model import GlobalMaxMeanPool

class GCNConvGrad(GCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)
        x = torch.matmul(x, self.weight)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out


class DeepWuKongModel(nn.Module):
    def __init__(self, need_node_emb=False):
        super(DeepWuKongModel, self).__init__()
        self.need_node_emb = need_node_emb
        self.hidden_dim = model_args.hidden_size
        cons = [GCNConv(model_args.vector_size, self.hidden_dim)]
        cons.extend([
                GCNConv(self.hidden_dim, self.hidden_dim)
                for _ in range(model_args.layer_num - 1)
            ])
        self.convs = nn.ModuleList(cons)

        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(model_args.layer_num)
            ]
        )
        self.readout = GlobalMaxMeanPool()
        self.ffn = nn.Sequential(*(
                [nn.Linear(self.hidden_dim * 2, self.hidden_dim)] +
                [nn.ReLU(), nn.Dropout()]
        ))

        self.final_layer = nn.Linear(self.hidden_dim, model_args.num_classes)
        self.dropout = nn.Dropout()

    # def generateGraphData(self, sequenceData: List[torch.FloatTensor], edge_index: torch.LongTensor, y: int) -> Data:
    #     '''
    #     :param sequenceData: List of tensor size [seq_length, num_embedding]
    #     :param edge_index:
    #     :return:
    #     '''
    #     feature = []
    #     for seq_data in sequenceData:
    #         feature.append(self.gru(seq_data.unsqueeze(dim=0))[0][0, -1, :])
    #     feature = torch.stack(feature)
    #     # feature = pack_sequence(sequenceData, enforce_sorted=False)
    #     # feature, _ = self.gru(feature.float())
    #     # feature, out_len = pad_packed_sequence(feature, batch_first=True)  # [node_num, max_seq_len, feature_size]
    #     # feature = feature[:, -1:, :].squeeze() # [node_num, 2 * hidden_size]
    #     return Data(x=feature, edge_index=edge_index, y=y)

    # support backwards-based explainers defined in DIG
    def arguments_read(self, *args, **kwargs):
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                           torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, edge_index, batch

    def forward(self, *args, **kwargs):
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = x
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
        out_readout = self.readout(post_conv, batch) # [batch_graph, hidden_size]
        out = self.ffn(out_readout)
        out = self.final_layer(out)

        if self.need_node_emb:
            return out, post_conv
        else:
            return out


if __name__ == '__main__':
    x1 = torch.ones(size=(5, model_args.vector_size))
    x2 = torch.ones(size=(6, model_args.vector_size))
    edge_index = torch.LongTensor([0, 1]).reshape(2, -1)
    y = 1

    dwk_model = DeepWuKongModel()
    data: Data = dwk_model.generateGraphData([x1, x2], edge_index, y)
    result = dwk_model(data=Batch.from_data_list([data]))
    print(result)
import torchsnooper
from treelstm import TreeLSTM
import torch
from torch import nn, Tensor
from torch.nn import GRU, Dropout, ReLU
from torch_sparse import SparseTensor

from gensim.models import Word2Vec
from detectors.IVDetect.util import IVDetectUtil, sample1

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import GCNConv, gcn_conv
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

from typing import List, Tuple
from detectors.IVDetect.configurations import model_args
from detectors.common_model import GlobalMaxPool


class GCNConvGrad(GCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    @torchsnooper.snoop()
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
        x = torch.matmul(x, edge_weight)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out


class IVDetectModel(nn.Module):
    def __init__(self, need_node_emb=False):
        super(IVDetectModel, self).__init__()
        self.need_node_emb = need_node_emb
        self.h_size = model_args.hidden_size
        self.hidden_dim = model_args.hidden_size
        self.num_node_feature = model_args.num_node_features
        self.num_classes = model_args.num_classes
        self.feature_representation_size = model_args.feature_representation_size
        self.drop_out_rate = model_args.dropout_rate
        self.layer_num = model_args.num_layers
        # The 1th feature input (subtoken sequence)
        self.gru_1 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # The 2th feature input (tree)
        self.tree_lstm = TreeLSTM(self.feature_representation_size, self.h_size)
        # The 3th feature input (variable and type sequence)
        self.gru_2 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # The 4th feature input (control dependence sequence)
        self.gru_3 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # The 5th feature input (data dependence sequence)
        self.gru_4 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # This layer is the bi-directional GRU layer
        self.gru_combine = GRU(input_size=self.h_size, hidden_size=self.h_size, bidirectional=True, batch_first=True)
        self.dropout = Dropout(self.drop_out_rate)
        # This layer is the GCN Model layer
        # h_size : in channels.  self.num_classes: out channels(2)
        self.connect = nn.Linear(self.h_size * self.num_node_feature * 2, self.h_size)
        gcn_list = [
                GCNConvGrad(self.h_size, self.h_size)
                for _ in range(model_args.num_layers - 1)
            ]
        gcn_list.append(GCNConvGrad(self.h_size, self.feature_representation_size))
        self.convs = nn.ModuleList(gcn_list)
        self.relu = ReLU(inplace=True)
        self.readout = GlobalMaxPool()

        self.final_layer = nn.Linear(self.h_size, self.num_classes)


    def vectorize_graph(self, data: Tuple[List[torch.Tensor], List[Tuple], List[torch.Tensor],
                                          List[torch.Tensor], List[torch.Tensor], torch.LongTensor, int]) -> Data:
        # vectorizing every node first
        feature_1, feature_2, feature_3, feature_4, feature_5, edge_index, label = data
        # process feature_1
        feature_1 = pack_sequence(feature_1, enforce_sorted=False)
        feature_1, _ = self.gru_1(feature_1.float())
        feature_1, out_len = pad_packed_sequence(feature_1, batch_first=True) # [node_num, max_seq_len, feature_size]

        # process feature_2
        feature_2_vec = []
        for ast in feature_2:
            # single ast node
            if len(ast) == 1:
                feature_2_vec.append(ast[0])
                continue
            features, edges, node_order, edge_order = ast
            h, c = self.tree_lstm(features.float(), node_order, edges, edge_order)
            # add embedding of root node
            feature_2_vec.append(h[0])
        feature_2_vec = torch.stack(feature_2_vec).to(model_args.device)

        feature_3 = pack_sequence(feature_3, enforce_sorted=False)
        feature_3, _ = self.gru_2(feature_3.float())
        feature_3, out_len = pad_packed_sequence(feature_3, batch_first=True) # [node_num, max_seq_len, feature_size]

        feature_4 = pack_sequence(feature_4, enforce_sorted=False)
        feature_4, _ = self.gru_3(feature_4.float())
        feature_4, out_len = pad_packed_sequence(feature_4, batch_first=True) # [node_num, max_seq_len, feature_size]

        feature_5 = pack_sequence(feature_5, enforce_sorted=False)
        feature_5, _ = self.gru_4(feature_5.float())
        feature_5, out_len = pad_packed_sequence(feature_5, batch_first=True) # [node_num, max_seq_len, feature_size]

        # [node_num, 5, feature_size]
        feature_input = torch.cat(
            (feature_2_vec.unsqueeze(dim=1), feature_1[:, -1:, :], feature_3[:, -1:, :], feature_4[:, -1:, :], feature_5[:, -1:, :]), 1) # (feature_2_vec.unsqueeze(dim=1),feature_3[:, -1:, :],
        feature_vec, _ = self.gru_combine(feature_input)
        feature_vec = self.dropout(feature_vec)
        feature_vec = torch.flatten(feature_vec, 1)
        # [num_node, h_size]
        feature_vec = self.connect(feature_vec)
        return Data(x=feature_vec, edge_index=edge_index, y=label)

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
        for i in range(self.layer_num - 1):
            post_conv = self.relu(self.convs[i](post_conv, edge_index))
        post_conv = self.convs[self.layer_num - 1](post_conv, edge_index)
        out_readout = self.readout(post_conv, batch)
        out = self.final_layer(out_readout)
        if self.need_node_emb:
            return out, post_conv
        else:
            return out



if __name__ == '__main__':
    ivdetect_model: IVDetectModel = IVDetectModel()
    ivdetect_model.to(model_args.device)


    pretrain_model = Word2Vec.load(model_args.pretrain_word2vec_model)
    ivdetect_util = IVDetectUtil(pretrain_model)

    feature = ivdetect_util.generate_all_features(sample1)
    data: Data = ivdetect_model.vectorize_graph(feature)
    output = ivdetect_model(data=Batch.from_data_list([data]).to(model_args.device))
    pass

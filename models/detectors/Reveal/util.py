from gensim.models import Word2Vec
import numpy as np
import torch
from torch_geometric.data import Data, Batch

import json
from typing import Dict, List, Tuple

from detectors.Reveal.configurations import type_map, model_args, data_args
from detectors.Reveal.model import ClassifyModel

class RevealUtil(object):
    def __init__(self, pretrain_model: Word2Vec, reveal_model: ClassifyModel):
        self.pretrain_model = pretrain_model
        self.reveal_model = reveal_model
        self.arrays = np.eye(80)

    # 生成图中每个ASTNode的初始embedding，在训练阶段这个函数只执行一次
    # nodeContent[0] 为type, nodeContent[1] 为token sequence
    def generate_initial_astNode_embedding(self, nodeContent: List[str]) -> np.array:
        # type vector
        n_c = self.arrays[type_map[nodeContent[0]]]
        # token sequence
        token_seq: List[str] = nodeContent[1].split(' ')
        n_v = np.array([self.pretrain_model[word] if word in self.pretrain_model.wv.vocab else
                        np.zeros(model_args.vector_size) for word in token_seq]).mean(axis=0)

        v = np.concatenate([n_c, n_v])
        return v

    # 生成每个AST初始结点的信息
    def generate_initial_node_info(self, ast: Dict) -> Data:
        astEmbedding: np.array = np.array(
            [self.generate_initial_astNode_embedding(node_info) for node_info in ast["contents"]])
        x: torch.FloatTensor = torch.FloatTensor(astEmbedding)
        edges: List[List[int]] = [[edge[1], edge[0]] for edge in ast["edges"]]
        edge_index: torch.LongTensor = torch.LongTensor(edges).t()
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device=data_args.device)

    # 预处理训练数据， 训练过程只调用1次
    def generate_initial_training_datas(self, data: Dict) -> Tuple[int, List[Data], torch.LongTensor]:
        # label, List[ASTNode], edge
        label: int = data["target"]

        # edges
        cfgEdges = [json.loads(edge)[:2] for edge in data["cfgEdges"]]
        cdgEdges = [json.loads(edge) for edge in data["cdgEdges"]]
        ddgEdges = [json.loads(edge)[:2] for edge in data["ddgEdges"]]
        edges = cfgEdges + cdgEdges + ddgEdges
        edge_index: torch.LongTensor = torch.LongTensor(edges).t()

        # nodes
        nodes_info: List[Dict] = [json.loads(node_infos) for node_infos in data["nodes"]]
        graph_data_for_each_nodes: List[Data] = [self.generate_initial_node_info(node_info) for node_info in nodes_info]

        return (label, graph_data_for_each_nodes, edge_index)

    # 生成图初始向量， 每个epoch会调用1次
    def generate_initial_graph_embedding(self, graph_info: Tuple[int, List[Data], torch.LongTensor]) -> Data:
        # self.reveal_model.embed_graph(data)[0] return graph_embedding for initial CPG node
        initial_embeddings: List[torch.FloatTensor] = [
            self.reveal_model.embed_graph(data.x, data.edge_index, None)[0].reshape(-1, )
            # 某些AST子树可能没有子结点，直接取其值作为node embedding
            if len(data.edge_index) > 0 else data.x[0]
            for data in graph_info[1]]
        X: torch.FloatTensor = torch.stack(initial_embeddings)
        return Data(x=X, edge_index=graph_info[2], y=torch.tensor([graph_info[0]], dtype=torch.long))


if __name__ == '__main__':
    pretrain_model = Word2Vec.load(model_args.pretrain_word2vec_model)
    reveal_model: ClassifyModel = ClassifyModel().to(model_args.device)
    reveal_util = RevealUtil(pretrain_model, reveal_model)

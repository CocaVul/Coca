from gensim.models import Word2Vec
import numpy as np
import torch
from torch_geometric.data import Data, Batch

import json
from typing import Dict, List, Tuple, Set

from detectors.Devign.configurations import type_map, model_args, data_args
from detectors.Devign.model import DevignModel


class DevignUtil(object):
    def __init__(self, pretrain_model: Word2Vec, devign_model: DevignModel):
        self.pretrain_model = pretrain_model
        self.devign_model = devign_model
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
    # 与Reveal相比,它还需要在AST终端结点添加NCS边
    def generate_initial_node_info(self, ast: Dict) -> Data:
        astEmbedding: np.array = np.array(
            [self.generate_initial_astNode_embedding(node_info) for node_info in ast["contents"]])
        x: torch.FloatTensor = torch.FloatTensor(astEmbedding)
        edges: List[List[int]] = [[edge[1], edge[0]] for edge in ast["edges"]]
        # 添加NCS边
        # 先找出图中所有的AST终端结点，按照索引顺序升序排序
        # 找出非终端结点的索引
        parent_idxs: Set[int] = set()
        terminal_idxs: List[int] = list()
        for edge in edges:
            parent_idxs.add(edge[1])
        # 找出终端结点的索引
        for i in range(len(x)):
            if i not in parent_idxs:
                terminal_idxs.append(i)
        # 添加NCS边
        for i in range(1, len(terminal_idxs)):
            edges.append([terminal_idxs[i - 1], terminal_idxs[i]])

        edge_index: torch.LongTensor = torch.LongTensor(edges).t()
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device=data_args.device)

    # 预处理训练数据， 训练过程只调用1次
    def generate_initial_training_datas(self, data: Dict) -> Tuple[int, List[Data], torch.LongTensor]:
        # label, List[ASTNode], edge
        label: int = data["target"]

        # edges
        cfgEdges = [json.loads(edge)[:2] for edge in data["cfgEdges"]]
        ddgEdges = [json.loads(edge)[:2] for edge in data["ddgEdges"]]
        edges = cfgEdges + ddgEdges
        edge_index: torch.LongTensor = torch.LongTensor(edges).t()

        # nodes
        nodes_info: List[Dict] = [json.loads(node_infos) for node_infos in data["nodes"]]
        graph_data_for_each_nodes: List[Data] = [self.generate_initial_node_info(node_info) for node_info in nodes_info]

        return (label, graph_data_for_each_nodes, edge_index)

    # 生成图初始向量， 每个epoch会调用1次
    def generate_initial_graph_embedding(self, graph_info: Tuple[int, List[Data], torch.LongTensor]) -> Data:
        # self.reveal_model.embed_graph(data)[0] return graph_embedding for initial CPG node
        initial_embeddings: List[torch.FloatTensor] = [
            self.devign_model.embed_graph(data.x, data.edge_index, None)[0].reshape(-1, )
            # 某些AST子树可能没有子结点，直接取其值作为node embedding
            if len(data.edge_index) > 0 else data.x[0]
            for data in graph_info[1]]
        X: torch.FloatTensor = torch.stack(initial_embeddings)
        return Data(x=X, edge_index=graph_info[2], y=torch.tensor([graph_info[0]], dtype=torch.long))

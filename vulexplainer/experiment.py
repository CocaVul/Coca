from explainer import VulExplainer
from util import SinkVisitor, BufferOverflowVisitor, IncorrectCalculationVisitor \
    , BackwardLeakVisitor, ForwardLeakVisitor, PathTraversalVisitor

from CppCodeAnalyzer.mainTool.CPG import CPG, CodeEdge
from CppCodeAnalyzer.mainTool.ast.builders import json2astNode
from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode

from global_defines import device, sparsity_value, cur_dir, cur_vul_type_idx
import json
from typing import Dict, List
import os
import sys
from tqdm import tqdm, trange

import numpy as np
from gensim.models.word2vec import Word2Vec
import torch
from torch_geometric.data import Data, Batch

# from vulexplainer import data_path, cur_detector
# Reveal
sys.path.append('D:/Program Files/PythonProjects/VulDetectors-master')
from detectors.Reveal.model import ClassifyModel
from detectors.Reveal.util import RevealUtil
from detectors.Reveal.configurations import model_args as reveal_model_args, \
    data_args as reveal_data_args

# Devign
from detectors.Devign.model import DevignModel
from detectors.Devign.util import DevignUtil
from detectors.Devign.configurations import model_args as devign_model_args, \
    data_args as devign_data_args

# IVDetect
from detectors.IVDetect.model import IVDetectModel
from detectors.IVDetect.util import IVDetectUtil
from detectors.IVDetect.configurations import model_args as ivdetect_model_args, \
    data_args as ivdetect_data_args

# DeepWuKong
from detectors.DeepWuKong.model import DeepWuKongModel
from detectors.DeepWuKong.configurations import model_args as deepwukong_model_args, \
    data_args as deepwukong_data_args
from detectors.DeepWuKong.train import TrainUtil

node_num = 3

cur_detector = "reveal"
slice_level: bool = (cur_detector == "deepwukong")

data_path = {
    "reveal": "function/explain_reveal.json",
    "devign": "function/explain_devign.json",
    "ivdetect": "function/explain_ivdetect.json",
    "deepwukong": "slice/explain_deepwukong.json"
}


class VulExplainerTester:
    def __init__(self):
        # self.explainer1: VulForwardExplainer = None
        if cur_vul_type_idx == 0:
            self.visitor: SinkVisitor = BufferOverflowVisitor()
        elif cur_vul_type_idx == 1:
            self.visitor: SinkVisitor = IncorrectCalculationVisitor()
        elif cur_vul_type_idx == 2:
            self.visitor: SinkVisitor = BackwardLeakVisitor()
        else:
            self.visitor: SinkVisitor = PathTraversalVisitor()
        self.explainer: VulExplainer = VulExplainer(self.visitor)
        self.explainer.limits = node_num
        # Devign doesn't use control dependence
        if cur_detector == "devign":
            self.explainer.add_cdg = False

    def fromSerJson(self, serJsonData: Dict):
        cfgEdges: List[list] = [json.loads(serEdge) for serEdge in serJsonData["cfgEdges"]]
        cdgEdges: List[list] = [json.loads(serEdge) for serEdge in serJsonData["cdgEdges"]]
        ddgEdges: List[list] = [json.loads(serEdge) for serEdge in serJsonData["ddgEdges"]]
        jsonStatements: List[dict] = [json.loads(serStmt) for serStmt in serJsonData["nodes"]]
        json_data: Dict = {
            "fileName": serJsonData["fileName"],
            "functionName": serJsonData["functionName"],
            "nodes": jsonStatements,
            "cfgEdges": cfgEdges,
            "cdgEdges": cdgEdges,
            "ddgEdges": ddgEdges
        }

        return CPG.fromJson(json_data)

    def constructCPGfromXFG(self, xfg_data: Dict) -> CPG:
        cpg: CPG = CPG()
        stmts: List[ASTNode] = list()
        # load nodes
        for node_info in xfg_data["line-nodes"]:
            node_content = json.loads(node_info)
            astNode: ASTNode = json2astNode(node_content)
            stmts.append(astNode)
        cpg.statements.extend(stmts)
        # load edge
        # cdg
        cpg.CDGEdges.extend(list(map(lambda e: CodeEdge.fromJson(json.loads(e)),
                                     xfg_data["control-dependences"])))
        # ddg
        cpg.DDGEdges.extend(list(map(lambda e: CodeEdge.fromJson(json.loads(e)),
                                     xfg_data["data-dependences"])))
        return cpg

    def explain(self, model, vul_idxs_list: List[List[int]], cpgs: List[CPG], all_datas: List[Data]):
        length = len(vul_idxs_list)
        recalls = []

        for idx in trange(length, desc="explaining", file=sys.stdout):
            cpg = cpgs[idx]
            data: Data = all_datas[idx]
            vul_idxs = vul_idxs_list[idx]
            sink_points, cdg_prec, ddg_prec = self.explainer.identify_sink_points(cpg)
            self.explainer.slices = list()
            if len(sink_points) == 0:
                recalls.append(0)
                continue

            self.explainer.generate_backward_slices(cpg, sink_points, cdg_prec, ddg_prec)
            cur_slices: List[List[int]] = self.explainer.slices

            # if cur_vul_type_idx != 2:
            # cur_slices = list(filter(lambda slice: len(slice) > 1, cur_slices))
            sub_data_list: List[Data] = list()
            for slice in cur_slices:
                # fidelity+
                mask = torch.FloatTensor(
                    [1 if i not in slice else 0 for i in range(len(data.x))]).unsqueeze(
                    dim=1).to(device)
                new_data: Data = Data(x=data.x * mask, edge_index=data.edge_index)
                sub_data_list.append(new_data)
            sub_probs = torch.softmax(model(data=Batch.from_data_list(sub_data_list).to(device)), dim=1)
            sub_probs = sub_probs.cpu().tolist()
            data_dicts = [(i, value[1]) for i, value in enumerate(sub_probs)]
            # each item is a tuple (path_idx, prob)
            sorted_dicts = sorted(data_dicts, key=lambda x: x[1], reverse=False)
            selected_path_idx = sorted_dicts[0][0]
            path = cur_slices[selected_path_idx]
            recall = len(set(vul_idxs) & set(path)) / len(vul_idxs)
            recalls.append(recall)

        print(f"{np.nanmean(recalls):.3f}")
        print("======================")

    def process_deepwukong(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(deepwukong_data_args.dataset_dir, data_path[cur_detector]),
                 'r', encoding='utf-8'))
        print(len(test_datas))
        print("=================")

        checkpoint = torch.load(
            os.path.join(deepwukong_model_args.model_dir,
                         f'{deepwukong_model_args.model_name}_{deepwukong_model_args.detector}_best.pth'))
        pretrain_model = Word2Vec.load(deepwukong_model_args.pretrain_word2vec_model)
        model: DeepWuKongModel = DeepWuKongModel()
        model.to(deepwukong_model_args.device)
        model.load_state_dict(checkpoint['net'])
        dwk_util: TrainUtil = TrainUtil(pretrain_model, model)

        cpgs: List[CPG] = [self.constructCPGfromXFG(sample) for sample in
                           tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        all_datas: List[Data] = [dwk_util.generate_initial_training_datas(sample)
                                 for sample in tqdm(test_datas, desc="embedding datas", file=sys.stdout)]
        self.explain(model, vul_idxs_list, cpgs, all_datas)

    def process_ivdetect(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(ivdetect_data_args.dataset_dir, data_path[cur_detector]),
                 'r', encoding='utf-8'))
        print(len(test_datas))
        print("=================")

        pretrain_model = Word2Vec.load(ivdetect_model_args.pretrain_word2vec_model)
        checkpoint = torch.load(
            os.path.join(ivdetect_model_args.model_dir,
                         f'{ivdetect_model_args.model_name}_{ivdetect_model_args.detector}_best.pth'))
        model: IVDetectModel = IVDetectModel()
        model.load_state_dict(checkpoint['net'])
        model.to(device)
        model.eval()
        ivdetect_util: IVDetectUtil = IVDetectUtil(pretrain_model)

        vul_features = [ivdetect_util.generate_all_features(sample) for sample in
                        tqdm(test_datas, desc="generating feature for vul sample", file=sys.stdout)]
        all_datas: List[Data] = [model.vectorize_graph(feature) for feature in
                                 tqdm(vul_features, desc="vectorizing data",
                                      file=sys.stdout)]
        cpgs: List[CPG] = [self.fromSerJson(sample) for sample in
                           tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        self.explain(model, vul_idxs_list, cpgs, all_datas)

    def process_reveal(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(reveal_data_args.dataset_dir, data_path[cur_detector]),
                 'r', encoding='utf-8'))
        print(len(test_datas))
        print("=================")

        # load model
        pretrain_model = Word2Vec.load(reveal_model_args.pretrain_word2vec_model)
        model: ClassifyModel = ClassifyModel()
        model.to(device)
        model.eval()
        checkpoint = torch.load(
            os.path.join(reveal_model_args.model_dir,
                         f'{reveal_model_args.model_name}_{reveal_model_args.detector}_best.pth'))
        model.load_state_dict(checkpoint['net'])
        reveal_util: RevealUtil = RevealUtil(pretrain_model, model)

        graph_infos: List[tuple] = [reveal_util.generate_initial_training_datas(sample) for sample in tqdm(test_datas,
                                                                                                           desc="parsing raw datas",
                                                                                                           file=sys.stdout)]
        all_datas: List[Data] = [reveal_util.generate_initial_graph_embedding(graph_info)
                                 for graph_info in tqdm(graph_infos, desc="embedding datas", file=sys.stdout)]
        cpgs: List[CPG] = [self.fromSerJson(sample) for sample in
                           tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        self.explain(model, vul_idxs_list, cpgs, all_datas)

    def process_devign(self):
        test_datas: List[Dict] = json.load(
            open(os.path.join(devign_data_args.dataset_dir, data_path[cur_detector]),
                 'r', encoding='utf-8'))
        # load model
        checkpoint = torch.load(
            os.path.join(devign_model_args.model_dir,
                         f'{devign_model_args.model_name}_{devign_model_args.detector}_best.pth'))
        pretrain_model = Word2Vec.load(devign_model_args.pretrain_word2vec_model)
        model: DevignModel = DevignModel()
        model.to(devign_model_args.device)
        model.load_state_dict(checkpoint['net'])
        devign_util = DevignUtil(pretrain_model, model)
        graph_infos: List[tuple] = [devign_util.generate_initial_training_datas(sample) for sample in tqdm(test_datas,
                                                                                                           desc="parsing raw datas",
                                                                                                           file=sys.stdout)]
        all_datas: List[Data] = [devign_util.generate_initial_graph_embedding(graph_info)
                                 for graph_info in tqdm(graph_infos, desc="embedding datas", file=sys.stdout)]
        cpgs: List[CPG] = [self.fromSerJson(sample) for sample in
                           tqdm(test_datas, desc="restoring CPG", file=sys.stdout)]
        vul_idxs_list: List[List[int]] = [sample["vul_idxs"] for sample in test_datas]
        self.explain(model, vul_idxs_list, cpgs, all_datas)


if __name__ == '__main__':
    tester = VulExplainerTester()
    if cur_detector == "reveal":
        tester.process_reveal()
    elif cur_detector == "devign":
        tester.process_devign()
    elif cur_detector == "deepwukong":
        tester.process_deepwukong()
    elif cur_detector == "ivdetect":
        tester.process_ivdetect()

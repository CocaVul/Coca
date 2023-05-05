from gensim.models import Word2Vec
import numpy as np
import json
from typing import Dict, List, Tuple, Set
from detectors.IVDetect.configurations import model_args
import torch
from treelstm import calculate_evaluation_orders

from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode
from CppCodeAnalyzer.mainTool.ast.builders import json2astNode
from CppCodeAnalyzer.extraTools.vuldetect.ivdetect import lexical_parse, generate_feature3, find_control, find_data


class IVDetectUtil(object):
    def __init__(self, w2v_model: Word2Vec):
        self.pretrain_model = w2v_model

    def generate_feature4(self, cpg: Dict, limit: int = 1) -> List[List[List[str]]]:
        edges = [json.loads(edge) for edge in cpg["cdgEdges"]]
        cdg_edge_idxs: Dict[int, int] = {edge[1]: edge[0] for edge in edges}
        nodes_infos: List[Dict] = cpg["nodes"]
        #  每个statement的控制依赖结点
        cd_idxs_for_stmt: List[List[int]] = list()
        for stmt_idx in range(len(nodes_infos)):
            seq: List[int] = list()
            find_control(stmt_idx, cdg_edge_idxs, seq, 1, limit)
            cd_idxs_for_stmt.append(seq)

        feature4_for_stmts: List[List[List[str]]] = list()
        for cd_idxs in cd_idxs_for_stmt:
            sub_tokens_in_stmts: List[List[str]] = [lexical_parse(nodes_infos[idx]["contents"][0][1])
                                                    for idx in cd_idxs]
            feature4_for_stmts.append(sub_tokens_in_stmts)

        return feature4_for_stmts

    def generate_feature5(self, cpg: Dict, limit: int = 1) -> List[List[List[str]]]:
        edges = [json.loads(edge) for edge in cpg["ddgEdges"]]
        ddg_edge_idxs: Dict[int, Set[int]] = dict()
        for edge in edges:
            # source -> destination
            if edge[1] not in ddg_edge_idxs.keys():
                ddg_edge_idxs[edge[1]] = {edge[0]}
            else:
                ddg_edge_idxs[edge[1]].add(edge[0])

            # destination -> source
            if edge[0] not in ddg_edge_idxs.keys():
                ddg_edge_idxs[edge[0]] = {edge[1]}
            else:
                ddg_edge_idxs[edge[0]].add(edge[1])

        nodes_infos: List[Dict] = cpg["nodes"]
        #  每个statement的控制依赖结点
        dd_idxs_for_stmt: List[List[int]] = list()
        for stmt_idx in range(len(nodes_infos)):
            seq: List[int] = list()
            find_data(stmt_idx, ddg_edge_idxs, seq, 1, limit)
            dd_idxs_for_stmt.append(seq)

        feature5_for_stmts: List[List[List[str]]] = list()
        for dd_idxs in dd_idxs_for_stmt:
            sub_tokens_in_stmts: List[List[str]] = [lexical_parse(nodes_infos[idx]["contents"][0][1])
                                                    for idx in dd_idxs]
            feature5_for_stmts.append(sub_tokens_in_stmts)
        return feature5_for_stmts

    def generate_all_features(self, data: Dict) -> Tuple[
        List[torch.Tensor], List[Tuple], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
        torch.LongTensor, int]:
        ## generate feature 1, sub token list
        nodes_infos: List[Dict] = [json.loads(sample) if isinstance(sample, str) else sample for sample in
                                   data["nodes"]]
        temp = data["nodes"]
        data["nodes"] = nodes_infos
        sub_tokens_list: List[List[str]] = [lexical_parse(node_infos["contents"][0][1]) for node_infos in nodes_infos]
        # vectorizing sub token list
        # 所有statement的token list
        feature1 = []
        for i, sub_tokens in enumerate(sub_tokens_list):
            stmt_vector = []
            for j, sub_token in enumerate(sub_tokens):
                if sub_token in self.pretrain_model.wv.vocab:
                    stmt_vector.append(self.pretrain_model[sub_token])
                else:
                    stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            if len(stmt_vector) == 0:
                stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            stmt_vector = np.stack(stmt_vector)
            feature1.append(torch.from_numpy(stmt_vector).to(model_args.device))

        ## generate feature2 , AST subtrees
        # each ast subtree is parsed into node feature, edge
        feature2: List[Tuple] = list()
        for ast in nodes_infos:
            edges = ast["edges"]
            if len(edges) == 0:
                tokens: List[str] = lexical_parse(ast["contents"][0][1])
                if len(tokens) == 0:
                    feature2.append(
                        (torch.zeros(size=(model_args.feature_representation_size,)).to(model_args.device),))
                    continue
                vecs = [self.pretrain_model[token] if token in self.pretrain_model.wv.vocab
                        else np.zeros(shape=(model_args.feature_representation_size)) for token in
                        tokens]
                vecs = np.stack(vecs).mean(axis=0)
                stmt_vector = torch.from_numpy(vecs).to(model_args.device)
                feature2.append((stmt_vector,))
                continue

            edges = torch.LongTensor(edges)
            stmt_vectors = []
            for node in ast["contents"]:
                tokens: List[str] = lexical_parse(node[1])
                if len(tokens) == 0:
                    stmt_vector = np.zeros(shape=(model_args.feature_representation_size))
                else:
                    stmt_vector = np.array([self.pretrain_model[token] if token in self.pretrain_model.wv.vocab
                                            else np.zeros(shape=(model_args.feature_representation_size)) for token in
                                            tokens]).mean(axis=0)
                stmt_vectors.append(stmt_vector)
            features = torch.from_numpy(np.stack(stmt_vectors))
            node_order, edge_order = calculate_evaluation_orders(edges, len(features))
            feature2.append((features.to(model_args.device), edges.to(model_args.device), node_order, edge_order))

        ## generate feature3, variable list
        astNodes: List[ASTNode] = [json2astNode(node_infos) for node_infos in nodes_infos]
        varLists: List[List[str]] = generate_feature3(astNodes)
        # vectorizing sub token list
        feature3 = []
        for i, varList in enumerate(varLists):
            stmt_vector = []
            for j, var in enumerate(varList):
                if var in self.pretrain_model.wv.vocab:
                    stmt_vector.append(self.pretrain_model[var])
                else:
                    stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            if len(stmt_vector) == 0:
                stmt_vector.append(np.zeros(shape=(model_args.feature_representation_size)))
            stmt_vector = np.stack(stmt_vector)
            feature3.append(torch.from_numpy(stmt_vector).to(model_args.device))

        ## generate feature4, control dependence list
        feature4_for_stmts: List[List[List[str]]] = self.generate_feature4(data, 1)

        # List[List[str]]
        feature4 = []
        for feature4_for_stmt in feature4_for_stmts:
            if len(feature4_for_stmt) == 0:
                feature4.append(torch.zeros(size=(1, model_args.feature_representation_size)).to(model_args.device))
                continue
            # List[str]
            vectors = []
            for context in feature4_for_stmt:
                # 一维向量
                if len(context) == 0:
                    stmt_vector = np.zeros(shape=(model_args.feature_representation_size))
                else:
                    stmt_vector = np.array([self.pretrain_model[token] if token in self.pretrain_model.wv.vocab
                                            else np.zeros(shape=(model_args.feature_representation_size)) for token in
                                            context]).mean(axis=0)
                vectors.append(stmt_vector)
            vectors = np.stack(vectors)
            feature4.append(torch.from_numpy(vectors).to(model_args.device))

        # generate feature 5, data dependence list
        feature5_for_stmts: List[List[List[str]]] = self.generate_feature5(data, 1)

        # List[List[str]]
        feature5 = []
        for feature5_for_stmt in feature5_for_stmts:
            if len(feature5_for_stmt) == 0:
                feature5.append(torch.zeros(size=(1, model_args.feature_representation_size)).to(model_args.device))
                continue
            # vector
            vectors = []
            # List[str]
            for context in feature5_for_stmt:
                # 一维向量
                if len(context) == 0:
                    stmt_vector = np.zeros(shape=(model_args.feature_representation_size))
                else:
                    stmt_vector = np.array([self.pretrain_model[token] if token in self.pretrain_model.wv.vocab
                                            else np.zeros(shape=(model_args.feature_representation_size)) for token in
                                            context]).mean(axis=0)
                vectors.append(stmt_vector)
            vectors = np.stack(vectors)
            feature5.append(torch.from_numpy(vectors).to(model_args.device))

        # edge indexes
        edges = [json.loads(edge)[:2] for edge in data["ddgEdges"]] + [json.loads(edge) for edge in data["cdgEdges"]]
        edge_index: torch.LongTensor = torch.LongTensor(edges).t().to(model_args.device)
        # label
        data["nodes"] = temp
        return (feature1, feature2, feature3, feature4, feature5, edge_index, data["target"])


sample = {
    "fileName": "CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01.c",
    "cfgEdges": [
        "[0,1,\"\"]",
        "[1,2,\"\"]",
        "[2,3,\"\"]",
        "[3,4,\"\"]",
        "[4,5,\"\"]",
        "[5,6,\"\"]"
    ],
    "nodes": [
        "{\"line\":37,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"IdentifierDeclStatement\",\"charVoid structCharVoid ;\",\"\"],[\"IdentifierDecl\",\"structCharVoid\",\"\"],[\"IdentifierDeclType\",\"charVoid\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
        "{\"line\":38,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . voidSecond = ( void * ) SRC_STR ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . voidSecond = ( void * ) SRC_STR\",\"=\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( void * ) SRC_STR\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"],[\"CastTarget\",\"void *\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"]]}",
        "{\"line\":40,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}",
        "{\"line\":42,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[8,11],[8,12],[10,13],[10,14]],\"contents\":[[\"ExpressionStatement\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) ) ;\",\"\"],[\"CallExpression\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) )\",\"\"],[\"Callee\",\"memcpy\",\"\"],[\"ArgumentList\",\"structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"memcpy\",\"\"],[\"Argument\",\"structCharVoid . charFirst\",\"\"],[\"Argument\",\"SRC_STR\",\"\"],[\"Argument\",\"sizeof ( structCharVoid )\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
        "{\"line\":43,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[4,6],[4,7],[5,8],[5,9],[8,10],[8,11],[10,12],[10,13],[11,14],[11,15],[13,16],[13,17]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0'\",\"=\"],[\"ArrayIndexing\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ]\",\"\"],[\"CharExpression\",\"'\\\\0'\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"AdditiveExpression\",\"( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1\",\"-\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"MultiplicativeExpression\",\"sizeof ( structCharVoid . charFirst ) / sizeof ( char )\",\"/\"],[\"IntegerExpression\",\"1\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid . charFirst )\",\"\"],[\"SizeofExpr\",\"sizeof ( char )\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"SizeofOperand\",\"char\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
        "{\"line\":44,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . charFirst ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . charFirst )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
        "{\"line\":45,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}"
    ],
    "cdgEdges": [],
    "ddgEdges": [
        "[0,1,\"structCharVoid\"]",
        "[1,2,\"structCharVoid\"]",
        "[1,3,\"structCharVoid\"]",
        "[1,4,\"structCharVoid\"]",
        "[1,5,\"structCharVoid\"]",
        "[1,6,\"structCharVoid\"]",
        "[1,2,\"structCharVoid . voidSecond\"]",
        "[1,6,\"structCharVoid . voidSecond\"]"
    ],
    "functionName": "CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_bad",
    "target": 1
}

sample1 = {
    "fileName": "CWE124_Buffer_Underwrite__malloc_wchar_t_ncpy_15.c",
    "cfgEdges": [
        "[0,1,\"\"]",
        "[1,2,\"\"]",
        "[2,3,\"case 6\"]",
        "[2,11,\"default\"]",
        "[3,4,\"\"]",
        "[4,5,\"\"]",
        "[5,6,\"True\"]",
        "[5,7,\"False\"]",
        "[6,7,\"\"]",
        "[7,8,\"\"]",
        "[8,9,\"\"]",
        "[9,10,\"\"]",
        "[10,14,\"\"]",
        "[11,12,\"\"]",
        "[12,13,\"\"]",
        "[13,14,\"\"]",
        "[14,15,\"\"]",
        "[15,16,\"\"]",
        "[16,17,\"\"]",
        "[17,18,\"\"]",
        "[18,19,\"\"]"
    ],
    "nodes": [
        "{\"line\":25,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"IdentifierDeclStatement\",\"wchar_t * data ;\",\"\"],[\"IdentifierDecl\",\"* data\",\"\"],[\"IdentifierDeclType\",\"wchar_t *\",\"\"],[\"Identifier\",\"data\",\"\"]]}",
        "{\"line\":26,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"ExpressionStatement\",\"data = NULL ;\",\"\"],[\"AssignmentExpr\",\"data = NULL\",\"=\"],[\"Identifier\",\"data\",\"\"],[\"PointerExpression\",\"NULL\",\"\"]]}",
        "{\"line\":27,\"edges\":[[0,1]],\"contents\":[[\"Condition\",\"6\",\"\"],[\"IntegerExpression\",\"6\",\"\"]]}",
        "{\"line\":29,\"edges\":[],\"contents\":[[\"Label\",\"case 6 :\",\"\"]]}",
        "{\"line\":31,\"edges\":[[0,1],[1,2],[1,3],[1,4],[4,5],[4,6],[6,7],[6,8],[8,9],[8,10],[9,11],[10,12],[12,13],[13,14],[13,15],[15,16],[15,17]],\"contents\":[[\"IdentifierDeclStatement\",\"wchar_t * dataBuffer = ( wchar_t * ) malloc ( 100 * sizeof ( wchar_t ) ) ;\",\"\"],[\"IdentifierDecl\",\"* dataBuffer = ( wchar_t * ) malloc ( 100 * sizeof ( wchar_t ) )\",\"\"],[\"IdentifierDeclType\",\"wchar_t *\",\"\"],[\"Identifier\",\"dataBuffer\",\"\"],[\"AssignmentExpr\",\"= ( wchar_t * ) malloc ( 100 * sizeof ( wchar_t ) )\",\"=\"],[\"Identifier\",\"dataBuffer\",\"\"],[\"CastExpression\",\"( wchar_t * ) malloc ( 100 * sizeof ( wchar_t ) )\",\"\"],[\"CastTarget\",\"wchar_t *\",\"\"],[\"CallExpression\",\"malloc ( 100 * sizeof ( wchar_t ) )\",\"\"],[\"Callee\",\"malloc\",\"\"],[\"ArgumentList\",\"100 * sizeof ( wchar_t )\",\"\"],[\"Identifier\",\"malloc\",\"\"],[\"Argument\",\"100 * sizeof ( wchar_t )\",\"\"],[\"MultiplicativeExpression\",\"100 * sizeof ( wchar_t )\",\"*\"],[\"IntegerExpression\",\"100\",\"\"],[\"SizeofExpr\",\"sizeof ( wchar_t )\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"SizeofOperand\",\"wchar_t\",\"\"]]}",
        "{\"line\":32,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"Condition\",\"dataBuffer == NULL\",\"\"],[\"EqualityExpression\",\"dataBuffer == NULL\",\"==\"],[\"Identifier\",\"dataBuffer\",\"\"],[\"PointerExpression\",\"NULL\",\"\"]]}",
        "{\"line\":32,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8]],\"contents\":[[\"ExpressionStatement\",\"exit ( - 1 ) ;\",\"\"],[\"CallExpression\",\"exit ( - 1 )\",\"\"],[\"Callee\",\"exit\",\"\"],[\"ArgumentList\",\"- 1\",\"\"],[\"Identifier\",\"exit\",\"\"],[\"Argument\",\"- 1\",\"\"],[\"UnaryOp\",\"- 1\",\"\"],[\"UnaryOperator\",\"-\",\"-\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":33,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[10,11],[10,12]],\"contents\":[[\"ExpressionStatement\",\"wmemset ( dataBuffer , L'A' , 100 - 1 ) ;\",\"\"],[\"CallExpression\",\"wmemset ( dataBuffer , L'A' , 100 - 1 )\",\"\"],[\"Callee\",\"wmemset\",\"\"],[\"ArgumentList\",\"dataBuffer , L'A' , 100 - 1\",\"\"],[\"Identifier\",\"wmemset\",\"\"],[\"Argument\",\"dataBuffer\",\"\"],[\"Argument\",\"L'A'\",\"\"],[\"Argument\",\"100 - 1\",\"\"],[\"Identifier\",\"dataBuffer\",\"\"],[\"CharExpression\",\"L'A'\",\"\"],[\"AdditiveExpression\",\"100 - 1\",\"-\"],[\"IntegerExpression\",\"100\",\"\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":34,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[5,6],[5,7]],\"contents\":[[\"ExpressionStatement\",\"dataBuffer [ 100 - 1 ] = L'\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"dataBuffer [ 100 - 1 ] = L'\\\\0'\",\"=\"],[\"ArrayIndexing\",\"dataBuffer [ 100 - 1 ]\",\"\"],[\"CharExpression\",\"L'\\\\0'\",\"\"],[\"Identifier\",\"dataBuffer\",\"\"],[\"AdditiveExpression\",\"100 - 1\",\"-\"],[\"IntegerExpression\",\"100\",\"\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":36,\"edges\":[[0,1],[1,2],[1,3],[3,4],[3,5]],\"contents\":[[\"ExpressionStatement\",\"data = dataBuffer - 8 ;\",\"\"],[\"AssignmentExpr\",\"data = dataBuffer - 8\",\"=\"],[\"Identifier\",\"data\",\"\"],[\"AdditiveExpression\",\"dataBuffer - 8\",\"-\"],[\"Identifier\",\"dataBuffer\",\"\"],[\"IntegerExpression\",\"8\",\"\"]]}",
        "{\"line\":38,\"edges\":[],\"contents\":[[\"BreakStatement\",\"break ;\",\"\"]]}",
        "{\"line\":39,\"edges\":[],\"contents\":[[\"Label\",\"default :\",\"\"]]}",
        "{\"line\":41,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6]],\"contents\":[[\"ExpressionStatement\",\"printLine ( \\\"Benign, fixed string\\\" ) ;\",\"\"],[\"CallExpression\",\"printLine ( \\\"Benign, fixed string\\\" )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"\\\"Benign, fixed string\\\"\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"\\\"Benign, fixed string\\\"\",\"\"],[\"StringExpression\",\"\\\"Benign, fixed string\\\"\",\"\"]]}",
        "{\"line\":42,\"edges\":[],\"contents\":[[\"BreakStatement\",\"break ;\",\"\"]]}",
        "{\"line\":45,\"edges\":[[0,1],[1,2],[1,3],[1,4]],\"contents\":[[\"IdentifierDeclStatement\",\"wchar_t source [ 100 ] ;\",\"\"],[\"IdentifierDecl\",\"source [ 100 ]\",\"\"],[\"IdentifierDeclType\",\"wchar_t *\",\"\"],[\"Identifier\",\"source\",\"\"],[\"IntegerExpression\",\"100\",\"\"]]}",
        "{\"line\":46,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[10,11],[10,12]],\"contents\":[[\"ExpressionStatement\",\"wmemset ( source , L'C' , 100 - 1 ) ;\",\"\"],[\"CallExpression\",\"wmemset ( source , L'C' , 100 - 1 )\",\"\"],[\"Callee\",\"wmemset\",\"\"],[\"ArgumentList\",\"source , L'C' , 100 - 1\",\"\"],[\"Identifier\",\"wmemset\",\"\"],[\"Argument\",\"source\",\"\"],[\"Argument\",\"L'C'\",\"\"],[\"Argument\",\"100 - 1\",\"\"],[\"Identifier\",\"source\",\"\"],[\"CharExpression\",\"L'C'\",\"\"],[\"AdditiveExpression\",\"100 - 1\",\"-\"],[\"IntegerExpression\",\"100\",\"\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":47,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[5,6],[5,7]],\"contents\":[[\"ExpressionStatement\",\"source [ 100 - 1 ] = L'\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"source [ 100 - 1 ] = L'\\\\0'\",\"=\"],[\"ArrayIndexing\",\"source [ 100 - 1 ]\",\"\"],[\"CharExpression\",\"L'\\\\0'\",\"\"],[\"Identifier\",\"source\",\"\"],[\"AdditiveExpression\",\"100 - 1\",\"-\"],[\"IntegerExpression\",\"100\",\"\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":49,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[10,11],[10,12]],\"contents\":[[\"ExpressionStatement\",\"wcsncpy ( data , source , 100 - 1 ) ;\",\"\"],[\"CallExpression\",\"wcsncpy ( data , source , 100 - 1 )\",\"\"],[\"Callee\",\"wcsncpy\",\"\"],[\"ArgumentList\",\"data , source , 100 - 1\",\"\"],[\"Identifier\",\"wcsncpy\",\"\"],[\"Argument\",\"data\",\"\"],[\"Argument\",\"source\",\"\"],[\"Argument\",\"100 - 1\",\"\"],[\"Identifier\",\"data\",\"\"],[\"Identifier\",\"source\",\"\"],[\"AdditiveExpression\",\"100 - 1\",\"-\"],[\"IntegerExpression\",\"100\",\"\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":51,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[5,6],[5,7]],\"contents\":[[\"ExpressionStatement\",\"data [ 100 - 1 ] = L'\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"data [ 100 - 1 ] = L'\\\\0'\",\"=\"],[\"ArrayIndexing\",\"data [ 100 - 1 ]\",\"\"],[\"CharExpression\",\"L'\\\\0'\",\"\"],[\"Identifier\",\"data\",\"\"],[\"AdditiveExpression\",\"100 - 1\",\"-\"],[\"IntegerExpression\",\"100\",\"\"],[\"IntegerExpression\",\"1\",\"\"]]}",
        "{\"line\":52,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6]],\"contents\":[[\"ExpressionStatement\",\"printWLine ( data ) ;\",\"\"],[\"CallExpression\",\"printWLine ( data )\",\"\"],[\"Callee\",\"printWLine\",\"\"],[\"ArgumentList\",\"data\",\"\"],[\"Identifier\",\"printWLine\",\"\"],[\"Argument\",\"data\",\"\"],[\"Identifier\",\"data\",\"\"]]}"
    ],
    "cdgEdges": [
        "[2,13]",
        "[2,12]",
        "[2,9]",
        "[2,10]",
        "[2,5]",
        "[2,11]",
        "[2,7]",
        "[2,4]",
        "[2,3]",
        "[2,8]",
        "[5,6]"
    ],
    "ddgEdges": [
        "[1,17,\"data\"]",
        "[9,17,\"data\"]",
        "[1,18,\"data\"]",
        "[9,18,\"data\"]",
        "[1,19,\"data\"]",
        "[9,19,\"data\"]",
        "[4,5,\"dataBuffer\"]",
        "[4,7,\"dataBuffer\"]",
        "[4,8,\"dataBuffer\"]",
        "[4,9,\"dataBuffer\"]",
        "[14,15,\"source\"]",
        "[14,16,\"source\"]",
        "[14,17,\"source\"]"
    ],
    "functionName": "CWE124_Buffer_Underwrite__malloc_wchar_t_ncpy_15_bad",
    "target": 1,
    "vul_lines": [
        49
    ],
    "vul_idxs": [
        17
    ]
}

if __name__ == '__main__':
    pretrain_model = Word2Vec.load(model_args.pretrain_word2vec_model)
    ivdetect_util = IVDetectUtil(pretrain_model)

    pass
    # from treelstm import TreeLSTM, calculate_evaluation_orders
    #
    # features = torch.FloatTensor([[1.1, 2.2, 3.3], [3, 2, 1], [1, 3, 5], [4, 5, 6]])
    # adjacency_list = torch.LongTensor([[0, 1],
    #                     [0, 2],
    #                     [2, 3]])
    # node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
    #
    # print(type(node_order))
    #
    # treeLSTM: TreeLSTM = TreeLSTM(in_features=3, out_features=2)
    # h, c = treeLSTM(features, node_order, adjacency_list, edge_order)
    # print(h[0])
    # print(c)

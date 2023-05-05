from CppCodeAnalyzer.mainTool.CPG import CPG
from CppCodeAnalyzer.mainTool.ast.expressions.expressionHolders import Condition
from CppCodeAnalyzer.mainTool.ast.declarations.simpleDecls import ForInit
from util import SinkVisitor, checkisREexpr, checkDependence, extract_vars, BackwardLeakVisitor
from __init__ import slice_level

from typing import List, Dict, Set

import sys

# sys.path.append("D:/PythonSpace/VulDetectors-master")
from global_defines import sparsity_value


## 773 401 forward, 789 400, backward
# detector = "DeepWuKong"

class VulExplainer:
    def __init__(self, visitor: SinkVisitor):
        self.visitor: SinkVisitor = visitor
        self.slices: List[List[int]] = list()
        self.add_cdg: bool = True
        self.limits: int = 5

    # 识别sink点
    def identify_sink_points(self, cpg: CPG):
        sink_idxs: List[int] = list()
        cdg_prec: Dict[int, List[int]] = dict()  # cdg前驱
        # 计算控制依赖信息
        for edge in cpg.CDGEdges:
            if edge.destination in cdg_prec.keys():
                cdg_prec[edge.destination].append(edge.source)
            else:
                cdg_prec[edge.destination] = [edge.source]

        key_vars: Dict[int, Set[int]] = dict()

        for i, stmt in enumerate(cpg.statements):
            self.visitor.reset()
            stmt.accept(self.visitor)
            if self.visitor.isSink:
                sink_idxs.append(i)
                key_vars[i] = self.visitor.key_vars
            elif self.visitor.potential:
                # 有没有对结果进行限制
                if not checkDependence(i, self.visitor.potential_var, self.visitor.check_upper,
                                       self.visitor.check_lower,
                                       cdg_prec, cpg.statements):
                    sink_idxs.append(i)
                    key_vars[i] = self.visitor.key_vars
            elif self.visitor.isCond:
                if isinstance(self.visitor, BackwardLeakVisitor):
                    key_vars[i] = self.visitor.key_vars
                    # 有关变量不能为常量并且是for loop
                    if len(key_vars) > 0 and isinstance(cpg.statements[i - 1], ForInit):
                        sink_idxs.append(i)

        cdg_prec: Dict[int, List[int]] = dict()  # cdg前驱
        # 计算控制依赖信息
        for edge in cpg.CDGEdges:
            if edge.destination in key_vars.keys():
                vars = key_vars[edge.destination]
                flag = False  # 数据依赖对应的变量是否可能是污点
                for var in vars:
                    source_vars = extract_vars(cpg.statements[edge.source])
                    if var in source_vars:
                        flag = True
                if not flag:
                    continue
            if edge.destination in cdg_prec.keys():
                cdg_prec[edge.destination].append(edge.source)
            else:
                cdg_prec[edge.destination] = [edge.source]

        # 计算数据依赖信息
        ddg_prec: Dict[int, List[int]] = dict()  # ddg前驱
        for edge in cpg.DDGEdges:
            # sink点
            if edge.destination in key_vars.keys():
                vars = key_vars[edge.destination]
                flag = False  # 数据依赖对应的变量是否可能是污点
                for var in vars:
                    # if you are explaining deepwukong, use following code
                    if slice_level:
                        source_vars = extract_vars(cpg.statements[edge.source])
                        if var in source_vars:
                            flag = True
                    # if you are explaining other 3 function-level detectors, use following code
                    else:
                        if var in edge.property:
                            flag = True
                if not flag:
                    continue

            if edge.destination in ddg_prec.keys():
                ddg_prec[edge.destination].append(edge.source)
            else:
                ddg_prec[edge.destination] = [edge.source]
        return sink_idxs, cdg_prec, ddg_prec

    def dfs(self, stack: List[int], num: int, ddg_prec: Dict[int, List[int]]
            , cdg_prec: Dict[int, List[int]], max_num: int):
        # 达到长度限制
        if num == max_num:
            self.slices.append(stack.copy())
            return
        cur = stack[-1]
        added_node: Set[int] = set()
        for prec_idx in ddg_prec.get(cur, []):
            added_node.add(prec_idx)
        if self.add_cdg:
            for prec_idx in cdg_prec.get(cur, []):
                if prec_idx in self.conds:
                    added_node.add(prec_idx)

        # 遍历到头了
        if len(added_node) == 0:
            self.slices.append(stack.copy())
            return

        for node_idx in added_node:
            stack.append(node_idx)
            self.slices.append(stack.copy())
            self.dfs(stack, num + 1, ddg_prec, cdg_prec, max_num)
            stack.pop()

    # 默认进行前向遍历
    def generate_backward_slices(self, cpg: CPG, sink_idxs: List[int]
                                 , cdg_prec: Dict[int, List[int]], ddg_prec: Dict[int, List[int]]):
        slices: List[List[int]] = list()
        self.conds: List[int] = list()

        # 遍历CPG中的condition结点
        for i, stmt in enumerate(cpg.statements):
            if isinstance(stmt, Condition):
                if checkisREexpr(stmt):
                    self.conds.append(i)
                else:
                    for prec_idx in ddg_prec.get(i, []):
                        if checkisREexpr(cpg.statements[prec_idx]):
                            self.conds.append(i)
                            break

        max_num: int = min(int(sparsity_value * len(cpg.statements)), self.limits)
        for idx in sink_idxs:
            stack: List[int] = [idx]
            self.dfs(stack, 1, ddg_prec, cdg_prec, max_num)

        return slices

from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode, ASTNodeVisitor
from CppCodeAnalyzer.mainTool.ast.expressions.expression import ArrayIndexing, UnaryOp, Identifier, ClassStaticIdentifier
from CppCodeAnalyzer.mainTool.ast.expressions.binaryExpressions import RelationalExpression, EqualityExpression
from CppCodeAnalyzer.mainTool.ast.expressions.binaryExpressions import BinaryExpression, AssignmentExpr
from CppCodeAnalyzer.mainTool.ast.expressions.postfixExpressions import IncDecOp, NewExpression, MemberAccess, CallExpression
from CppCodeAnalyzer.mainTool.ast.expressions.primaryExpressions import PrimaryExpression, IntegerExpression
from CppCodeAnalyzer.mainTool.ast.expressions.expressionHolders import Condition

from typing import List, Dict, Set

def extract_vars(item: ASTNode) -> Set[str]:
    vars: Set[str] = set()
    if isinstance(item, Identifier) and not isinstance(item, ClassStaticIdentifier):
        vars.add(item.getEscapedCodeStr())
    else:
        for i in range(item.getChildCount()):
            sub_vars = extract_vars(item.getChild(i))
            vars.update(sub_vars)
    return vars


class SinkVisitor(ASTNodeVisitor):
    def __init__(self):
        self.reset()
        self.conds: List[ASTNode] = list() # 存储可能为约束条件的condition，像常量表达式就不属于这种

    def reset(self):
        self.isSink: bool = False  # 表示该CFG结点是否有可能是sink点
        self.isCond: bool = False # 表示该CFG结点是条件表达式
        self.potential: bool = False  # 表示该CFG结点需要检查其控制依赖约束条件是否有可能造成sink
        self.potential_var: Identifier = None
        self.check_upper: bool = False  # 是否需要检查上界
        self.check_lower: bool = False  # 是否需要检查下界
        self.key_vars: Set[str] = set()
        self.conds: List[ASTNode] = list()  # 存储可能为约束条件的condition，像常量表达式就不属于这种
    # 提取关键的变量名

    def extract_vars(self, item: ASTNode) -> Set[str]:
        vars: Set[str] = set()
        if isinstance(item, Identifier) and not isinstance(item, ClassStaticIdentifier):
            vars.add(item.getEscapedCodeStr())
        else:
            for i in range(item.getChildCount()):
                sub_vars = self.extract_vars(item.getChild(i))
                vars.update(sub_vars)
        return vars


# 用来搜寻条件表达式
class RelationExprVisitor(ASTNodeVisitor):
    def __init__(self):
        self.containsRelationExpr: bool = False

    def visit(self, item: ASTNode):
        if isinstance(item, RelationalExpression) or isinstance(item, EqualityExpression):
            self.containsRelationExpr = True
        super().visit(item)

# 检测表达式是否使用变量
def checkExpression(item: ASTNode) -> bool:
    # 该表达式是常量，返回false
    if isinstance(item, PrimaryExpression):
        return False
    # 函数调用语句，检查参数是否引用变量
    elif isinstance(item, CallExpression):
        # 第一个子结点为参数列表
        return checkExpression(item.getChild(1))
    # 此时Identifier为变量名
    elif isinstance(item, Identifier):
        return True
    flag = False
    for i in range(item.getChildCount()):
        flag |= checkExpression(item.getChild(i))
    return flag

#检测表达式是否是条件表达式
def checkisREexpr(item: ASTNode) -> bool:
    visitor: RelationExprVisitor = RelationExprVisitor()
    item.accept(visitor)
    return visitor.containsRelationExpr

# 检查该变量是否有对应的约束条件
def checkDependence(sink_idx: int, sink_var: Identifier, check_upper: bool, check_lower: bool,
                    cdg_precs: Dict[int, List[int]], stmts: List[ASTNode]) -> bool:
    queue: List[int] = [prec_idx for prec_idx in cdg_precs.get(sink_idx, [])]
    visited: Set[int] = set()
    while len(queue) > 0:
        cond_idx: int = queue.pop(0)
        visited.add(cond_idx)
        cond: ASTNode = stmts[cond_idx]
        if checkCondition(sink_var, cond, check_upper, check_lower):
            return True
        for prec_idx in cdg_precs.get(cond_idx, []):
            if prec_idx not in visited:
                queue.append(prec_idx)
    return False


def checkCondition(sink_var: Identifier, condition: ASTNode,
                   check_upper: bool, check_lower: bool) -> bool:
    if isinstance(condition, RelationalExpression):
        # f(var) < xxx
        if sink_var.getEscapedCodeStr() in condition.getChild(0).getEscapedCodeStr() \
            and condition.operator == "<" and check_upper:
            return True
        if sink_var.getEscapedCodeStr() in condition.getChild(0).getEscapedCodeStr() \
            and condition.operator == ">" and check_lower:
            return True
        return False

    flag: bool = False
    for i in range(condition.getChildCount()):
        flag |= checkCondition(sink_var, condition.getChild(i), check_upper, check_lower)
    return flag


class BufferOverflowVisitor(SinkVisitor):
    # 缓冲区溢出漏洞主要发生在字符串拷贝，数组/指针访问,包括ptrMemberAccess
    def __init__(self):
        super().__init__()
        self.cpy_funcs: List[str] = ['memcpy', 'memmove', 'memcmp', 'strncpy', 'wcsncpy', 'strcpy',
                                     'wcscpy', 'strncat', 'strcat', 'wcscat', 'snprintf', 'wcsncat']

    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            if item.getChild(0).getEscapedCodeStr().lower() in self.cpy_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.cpy_funcs:
                self.isSink = True
                # 参数列表中所有的变量都需要访问
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
        # Array Usage
        elif isinstance(item, ArrayIndexing):
            # 检查其索引是否都是常量
            if checkExpression(item.getChild(1)):
                self.key_vars.update(self.extract_vars(item.getChild(0)))
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                self.isSink = True
            return
        # Pointer Usage
        elif isinstance(item, UnaryOp):
            if item.operator == '*':
                self.key_vars.update(self.extract_vars(item))
                self.isSink = True
                return
        super().visit(item)


class IncorrectCalculationVisitor(SinkVisitor):
    # 整数溢出漏洞主要发生在算数运算中
    def __init__(self):
        super().__init__()

    def visit(self, item: ASTNode):
        if isinstance(item, BinaryExpression):
            if item.operator in {'+', '-', '*', '<<', '>>', '/', '%'}:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item))
                return
        # increament assignment
        elif isinstance(item, AssignmentExpr):
            if item.operator in {"+=", "-=", "*=", ">>=", "<<=", '/=', '%='}:
                # 看看左操作数是不是常量1
                if isinstance(item.getChild(1), IntegerExpression) and item.getChild(1).getEscapedCodeStr() == '1':
                    self.potential = True
                    self.potential_var = item.getChild(0)
                    if item.operator in {"+=", "*=", "<<="}:
                        self.check_upper = True
                    else:
                        self.check_lower = True
                else:
                    self.isSink = True
                self.key_vars.update(self.extract_vars(item))
                return
        # x++ / x-- / ++x / --x
        elif isinstance(item, IncDecOp):
            self.potential = True
            self.key_vars.update(self.extract_vars(item))
            for i in range(item.getChildCount()):
                if isinstance(item.getChild(i), Identifier):
                    self.potential_var = item.getChild(i)
                    if "++" in item.getEscapedCodeStr():
                        self.check_upper = True
                    elif "--" in item.getEscapedCodeStr():
                        self.check_lower = True
                    break
            return
        self.visitChildren(item)

# forward
class ForwardLeakVisitor(SinkVisitor):
    def __init__(self):
        super().__init__()
        self.alloc_funcs = ['alloca', 'malloc', 'realloc', 'calloc', 'strdup',
                            "fopen", "open", "CreateFileA", "CreateFileW", "CreateFile",
                            'open']


    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            if item.getChild(0).getEscapedCodeStr().lower() in self.alloc_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.alloc_funcs:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
        elif isinstance(item, NewExpression):
            self.isSink = True
            self.key_vars.update(self.extract_vars(item))
            return
        self.visitChildren(item)

# backward
class BackwardLeakVisitor(SinkVisitor):
    def __init__(self):
        super().__init__()
        self.resource_funcs = ["fopen", "CreateFile", "CreateFileA", "CreateFileW", "open",
                               'alloca', 'malloc', 'realloc', 'calloc', 'strdup', "sleep",
                               "fwrite"]
        self.member_funcs = ["open"]

    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            if item.getChild(0).getEscapedCodeStr().lower() in self.resource_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.resource_funcs:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
            else:
                if isinstance(item.getChild(0).getChild(0), MemberAccess):
                    memAccess: MemberAccess = item.getChild(0).getChild(0)
                    # 获取成员函数名
                    funcName: Identifier = memAccess.getChild(1)
                    if funcName.getEscapedCodeStr() in self.member_funcs:
                        self.isSink = True
                        self.key_vars.update(self.extract_vars(item.getChild(1)))
                        return
        elif isinstance(item, NewExpression):
            self.isSink = True
            self.key_vars.update(self.extract_vars(item))
            return
        elif isinstance(item, Condition):
            self.isCond = True
            if checkisREexpr(item):
                self.key_vars.update(self.extract_vars(item.getChild(0)))#.getChild(1)))
        self.visitChildren(item)


class PathTraversalVisitor(SinkVisitor):
    def __init__(self):
        super().__init__()
        self.file_funcs = ["fopen", "open", "CreateFileA", "CreateFileW", "CreateFile"]
        self.member_funcs = ["open"]


    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            # 直接函数调用
            if item.getChild(0).getEscapedCodeStr().lower() in self.file_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.file_funcs:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
            else:
                if isinstance(item.getChild(0).getChild(0), MemberAccess):
                    memAccess: MemberAccess = item.getChild(0).getChild(0)
                    # 获取成员函数名
                    funcName: Identifier = memAccess.getChild(1)
                    if funcName.getEscapedCodeStr() in self.member_funcs:
                        self.isSink = True
                        self.key_vars.update(self.extract_vars(item.getChild(1)))
                        return

        self.visitChildren(item)
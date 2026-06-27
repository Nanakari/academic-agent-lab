import ast  # 导入 ast 模块，用来把表达式解析成 Python 抽象语法树

import operator  # 导入 operator 模块，用来安全地执行加减乘除等运算

from app.tool.base import BaseTool, ToolResult  # 从工具基类文件中导入 BaseTool 和 ToolResult


class CalculatorTool(BaseTool):  # 定义 CalculatorTool，它继承 BaseTool
    name = "calculator"  # 设置工具名称，后续模型和工具集合会用这个名字识别工具

    description = "Calculate a safe mathematical expression."  # 设置工具描述，说明这个工具用于计算数学表达式

    parameters = {  # 定义工具参数 schema，用来描述这个工具需要什么输入
        "type": "object",  # 参数整体是一个 JSON object
        "properties": {  # properties 用来定义 object 里面有哪些字段
            "expression": {  # 定义 expression 参数
                "type": "string",  # expression 的类型是字符串
                "description": "A mathematical expression, such as '123 * 456'.",  # 描述 expression 应该传入什么内容
            },  # expression 参数定义结束
        },  # properties 定义结束
        "required": ["expression"],  # required 表示 expression 是必填参数
    }  # parameters 定义结束

    allowed_operators = {  # 定义允许使用的运算符映射表
        ast.Add: operator.add,  # 允许加法，例如 1 + 2
        ast.Sub: operator.sub,  # 允许减法，例如 3 - 1
        ast.Mult: operator.mul,  # 允许乘法，例如 2 * 3
        ast.Div: operator.truediv,  # 允许普通除法，例如 6 / 2
        ast.FloorDiv: operator.floordiv,  # 允许整除，例如 7 // 2
        ast.Mod: operator.mod,  # 允许取余，例如 7 % 2
        ast.Pow: operator.pow,  # 允许乘方，例如 2 ** 3
    }  # allowed_operators 定义结束

    allowed_unary_operators = {  # 定义允许使用的一元运算符映射表
        ast.UAdd: operator.pos,  # 允许正号，例如 +1
        ast.USub: operator.neg,  # 允许负号，例如 -1
    }  # allowed_unary_operators 定义结束

    def execute(self, expression: str) -> ToolResult:  # 实现具体执行方法，接收 expression 字符串
        try:  # 开始捕获解析和计算过程中的异常
            tree = ast.parse(expression, mode="eval")  # 把表达式解析成 AST，mode="eval" 表示只解析表达式
            value = self._eval_node(tree.body)  # 递归计算 AST 的主体节点
            return ToolResult(output=str(value))  # 把计算结果转成字符串，并包装成 ToolResult 返回
        except Exception as error:  # 如果解析或计算过程中出现错误
            return ToolResult(error=f"Calculator error: {error}")  # 返回统一错误格式，而不是让程序崩溃

    def _eval_node(self, node) -> int | float:  # 定义内部递归方法，用来计算 AST 节点
        if isinstance(node, ast.Constant):  # 如果当前节点是常量，例如 123 或 3.14
            if isinstance(node.value, int | float):  # 如果常量值是整数或浮点数
                return node.value  # 直接返回这个数字
            raise ValueError("Only numbers are allowed.")  # 如果不是数字，就抛出错误

        if isinstance(node, ast.BinOp):  # 如果当前节点是二元运算，例如 1 + 2
            operator_type = type(node.op)  # 获取运算符类型，例如 ast.Add
            if operator_type not in self.allowed_operators:  # 如果运算符不在允许列表中
                raise ValueError(f"Operator not allowed: {operator_type.__name__}")  # 抛出不允许使用该运算符的错误
            left_value = self._eval_node(node.left)  # 递归计算左边表达式
            right_value = self._eval_node(node.right)  # 递归计算右边表达式
            return self.allowed_operators[operator_type](left_value, right_value)  # 调用对应运算函数并返回结果

        if isinstance(node, ast.UnaryOp):  # 如果当前节点是一元运算，例如 -1
            operator_type = type(node.op)  # 获取一元运算符类型，例如 ast.USub
            if operator_type not in self.allowed_unary_operators:  # 如果一元运算符不在允许列表中
                raise ValueError(f"Unary operator not allowed: {operator_type.__name__}")  # 抛出不允许使用该运算符的错误
            operand_value = self._eval_node(node.operand)  # 递归计算一元运算的操作数
            return self.allowed_unary_operators[operator_type](operand_value)  # 调用对应一元运算函数并返回结果

        raise ValueError(f"Unsupported expression: {type(node).__name__}")  # 如果节点类型不支持，就抛出错误
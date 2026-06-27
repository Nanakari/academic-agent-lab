from abc import ABC, abstractmethod  # 从 abc 导入 ABC 和 abstractmethod，用来定义抽象工具基类

from dataclasses import dataclass  # 从 dataclasses 导入 dataclass，用来快速定义工具返回结果类


@dataclass  # 使用 dataclass 自动生成 __init__、__repr__ 等基础方法
class ToolResult:  # 定义 ToolResult 类，用来统一表示工具执行结果
    output: str | None = None  # output 表示工具正常执行后的输出内容，默认是 None
    error: str | None = None  # error 表示工具执行失败时的错误信息，默认是 None

    @property  # 把下面的方法变成一个只读属性，调用时可以写 result.is_success
    def is_success(self) -> bool:  # 定义判断工具是否执行成功的方法
        return self.error is None  # 如果 error 是 None，就表示没有错误，也就是执行成功


class BaseTool(ABC):  # 定义 BaseTool 抽象基类，所有具体工具都要继承它
    name: str = ""  # name 表示工具名称，后续会给 LLM 看，也会给 ToolCollection 查找工具用
    description: str = ""  # description 表示工具描述，后续会帮助 LLM 理解什么时候使用这个工具
    parameters: dict = {}  # parameters 表示工具参数 schema，后续会告诉 LLM 这个工具需要什么参数

    def __call__(self, **kwargs) -> ToolResult:  # 定义对象被直接调用时的行为，例如 calculator(expression="1+1")
        try:  # 开始捕获工具执行过程中的异常，防止整个程序崩溃
            return self.execute(**kwargs)  # 调用具体工具实现的 execute 方法，并返回 ToolResult
        except Exception as error:  # 如果 execute 过程中出现任何异常
            return ToolResult(error=str(error))  # 把异常转换成 ToolResult(error=...) 返回，而不是直接崩溃

    @abstractmethod  # 标记 execute 是抽象方法，子类必须实现
    def execute(self, **kwargs) -> ToolResult:  # 定义工具执行入口，不同工具会有不同参数
        pass  # 抽象方法占位，具体执行逻辑由子类实现

    def to_param(self) -> dict:  # 定义把工具转换成 LLM 可读 schema 的方法
        return {  # 返回一个字典，这个格式接近 OpenAI tools/function calling 的结构
            "type": "function",  # type 表示这是一个 function 类型工具
            "function": {  # function 字段里放工具的具体说明
                "name": self.name,  # 工具名称，例如 calculator
                "description": self.description,  # 工具描述，例如用于计算数学表达式
                "parameters": self.parameters,  # 工具参数 schema，例如 expression 是字符串
            },  # function 字段结束
        }  # 整个工具 schema 返回结束
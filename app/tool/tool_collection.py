from app.tool.base import BaseTool, ToolResult  # 从工具基类模块中导入 BaseTool 和 ToolResult


class ToolCollection:  # 定义 ToolCollection 类，用来统一管理多个工具
    def __init__(self, *tools: BaseTool) -> None:  # 定义初始化方法，接收任意数量的工具对象
        self.tools = list(tools)  # 把传入的工具对象转换成列表，保存到 self.tools 中

        self.tool_map = {tool.name: tool for tool in self.tools}  # 构建工具名字到工具对象的映射字典

    def to_params(self) -> list[dict]:  # 定义方法，把所有工具转换成 LLM 可读的 schema 列表
        return [tool.to_param() for tool in self.tools]  # 遍历所有工具，调用每个工具的 to_param() 方法

    def get_tool(self, name: str) -> BaseTool | None:  # 定义根据工具名查找工具的方法
        return self.tool_map.get(name)  # 从 tool_map 中按 name 查找工具，找不到就返回 None

    def execute(self, name: str, tool_input: dict | None = None) -> ToolResult:  # 定义统一执行工具的方法
        if tool_input is None:  # 如果调用方没有传入工具参数
            tool_input = {}  # 就把工具参数设置为空字典，避免后续 **None 报错

        tool = self.get_tool(name)  # 根据工具名称查找对应工具对象

        if tool is None:  # 如果没有找到工具
            return ToolResult(error=f"Tool not found: {name}")  # 返回统一错误结果，而不是让程序崩溃

        if not isinstance(tool_input, dict):  # 如果工具参数不是字典
            return ToolResult(error="Tool input must be a dict.")  # 返回参数类型错误

        return tool(**tool_input)  # 通过 BaseTool.__call__ 统一调用工具，并返回 ToolResult
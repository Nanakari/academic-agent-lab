from app.tool.base import BaseTool, ToolResult  # 从工具基类文件中导入 BaseTool 和 ToolResult


class TerminateTool(BaseTool):  # 定义 TerminateTool，它继承 BaseTool
    name = "terminate"  # 设置工具名称，后续 Agent 会用这个名字识别终止工具

    description = "Terminate the current task when it is finished."  # 设置工具描述，说明这个工具用于结束任务

    parameters = {  # 定义工具参数 schema
        "type": "object",  # 参数整体是一个 JSON object
        "properties": {  # properties 用来定义 object 里面有哪些字段
            "reason": {  # 定义 reason 参数
                "type": "string",  # reason 的类型是字符串
                "description": "The reason why the task should terminate.",  # 描述 reason 的作用
            },  # reason 参数定义结束
        },  # properties 定义结束
        "required": [],  # required 为空，表示这个工具可以不传参数
    }  # parameters 定义结束

    def execute(self, reason: str = "Task finished.") -> ToolResult:  # 实现具体执行方法，reason 有默认值
        return ToolResult(output=f"Terminate: {reason}")  # 返回终止原因，真正修改 Agent 状态会在后续 ToolCallAgent 里做
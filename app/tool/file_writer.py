from pathlib import Path  # 导入 Path，用于处理文件路径

from app.tool.base import BaseTool, ToolResult  # 导入工具基类和工具返回结果类


class FileWriterTool(BaseTool):  # 定义文件写入工具，继承统一工具基类
    name = "file_writer"  # 定义工具名称，模型调用工具时会使用这个名字

    description = "将文本内容写入项目 outputs 目录下的文件，例如 outputs/summary.md。"  # 定义工具描述

    parameters = {  # 定义工具参数 schema
        "type": "object",  # 参数整体是一个 JSON 对象
        "properties": {  # 定义对象内部字段
            "path": {  # 定义 path 参数
                "type": "string",  # path 必须是字符串
                "description": "要写入的目标路径，例如 outputs/summary.md",  # 说明 path 的含义
            },  # path 参数定义结束
            "content": {  # 定义 content 参数
                "type": "string",  # content 必须是字符串
                "description": "要写入文件的文本内容",  # 说明 content 的含义
            },  # content 参数定义结束
        },  # properties 定义结束
        "required": ["path", "content"],  # path 和 content 都是必填参数
    }  # parameters 定义结束

    def __init__(self) -> None:  # 初始化文件写入工具
        self.root_dir = Path(__file__).resolve().parents[2]  # 获取项目根目录 academic-agent-lab
        self.outputs_dir = self.root_dir / "outputs"  # 设置允许写入的 outputs 目录

    def execute(self, path: str, content: str) -> ToolResult:  # 执行文件写入逻辑
        try:  # 捕获可能发生的路径或写入错误
            file_path = self._resolve_safe_path(path)  # 将用户传入路径解析成安全路径

            file_path.parent.mkdir(parents=True, exist_ok=True)  # 如果父目录不存在，就自动创建

            file_path.write_text(content, encoding="utf-8")  # 使用 UTF-8 写入文本内容

            relative_path = file_path.relative_to(self.root_dir)  # 将绝对路径转换成相对项目根目录的路径

            return ToolResult(output=f"File written successfully: {relative_path}")  # 返回写入成功信息

        except Exception as exc:  # 捕获所有未预期异常
            return ToolResult(error=f"FileWriter error: {exc}")  # 将异常转换为 ToolResult 错误

    def _resolve_safe_path(self, path: str) -> Path:  # 定义安全路径解析方法
        raw_path = Path(path)  # 将字符串路径转换为 Path 对象

        if raw_path.is_absolute():  # 如果用户传入的是绝对路径
            candidate = raw_path.resolve()  # 解析绝对路径
        else:  # 如果用户传入的是相对路径
            candidate = (self.root_dir / raw_path).resolve()  # 拼接到项目根目录下

        if self.outputs_dir not in candidate.parents and candidate != self.outputs_dir:  # 检查目标路径是否位于 outputs 目录内
            raise ValueError("只能写入 outputs 目录内的文件")  # 如果越界写入，则抛出安全错误

        return candidate  # 返回安全解析后的路径
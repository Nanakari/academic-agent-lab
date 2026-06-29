from pathlib import Path  # 导入 Path，用来安全处理文件路径

from app.rag.document_loader import load_document_text  # 导入统一文档读取函数，用于支持 txt、md 和文本型 PDF
from app.tool.base import BaseTool, ToolResult  # 导入工具基类和工具返回结果类


class FileReaderTool(BaseTool):  # 定义文件读取工具，继承统一工具基类
    name = "file_reader"  # 定义工具名称，模型调用工具时会使用这个名字

    description = "读取项目目录下的 txt、md 或文本型 PDF 文件内容。"  # 定义工具描述并声明支持文本型 PDF

    parameters = {  # 定义工具参数 schema，供 LLM 进行 function calling
        "type": "object",  # 参数整体是一个 JSON 对象
        "properties": {  # 定义对象内部有哪些字段
            "path": {  # 定义 path 参数
                "type": "string",  # path 必须是字符串
                "description": "要读取的文件路径，例如 data/demo_paper.txt、data/demo_paper.md 或 data/demo_paper.pdf",  # 说明支持的文件类型
            },  # path 参数定义结束
            "max_chars": {  # 定义 max_chars 参数
                "type": "integer",  # max_chars 必须是整数
                "description": "最多读取多少个字符，默认 12000",  # 说明 max_chars 的含义
            },  # max_chars 参数定义结束
        },  # properties 定义结束
        "required": ["path"],  # path 是必填参数
    }  # parameters 定义结束

    def __init__(self) -> None:  # 初始化文件读取工具
        self.root_dir = Path(__file__).resolve().parents[2]  # 获取项目根目录 academic-agent-lab

    def execute(self, path: str, max_chars: int = 12000) -> ToolResult:  # 执行文件读取逻辑
        try:  # 捕获可能发生的文件路径或读取错误
            file_path = self._resolve_safe_path(path)  # 将用户传入路径转换为安全的绝对路径

            if not file_path.exists():  # 判断文件是否存在
                return ToolResult(error=f"File not found: {path}")  # 文件不存在时返回错误结果

            if not file_path.is_file():  # 判断路径是否是普通文件
                return ToolResult(error=f"Path is not a file: {path}")  # 如果不是文件则返回错误结果

            text = load_document_text(file_path)  # 通过统一文档读取层加载 txt、md 或文本型 PDF 内容

            if len(text) > max_chars:  # 如果文本长度超过最大字符数
                text = text[:max_chars] + "\n\n[内容过长，已截断]"  # 截断文本，避免一次传给模型太长

            return ToolResult(output=text)  # 返回读取到的文本内容

        except Exception as exc:  # 捕获所有未预期异常
            return ToolResult(error=f"FileReader error: {exc}")  # 将异常转换为 ToolResult 错误

    def _resolve_safe_path(self, path: str) -> Path:  # 定义安全路径解析方法
        raw_path = Path(path)  # 将字符串路径转换为 Path 对象

        if raw_path.is_absolute():  # 如果用户传入的是绝对路径
            candidate = raw_path.resolve()  # 解析绝对路径
        else:  # 如果用户传入的是相对路径
            candidate = (self.root_dir / raw_path).resolve()  # 将相对路径拼接到项目根目录下

        if self.root_dir not in candidate.parents and candidate != self.root_dir:  # 检查目标路径是否位于项目目录内
            raise ValueError("只能读取项目目录内的文件")  # 如果越界访问，则抛出安全错误

        return candidate  # 返回安全解析后的路径
from pathlib import Path  # 导入 Path，用于处理文件路径

from app.rag.retriever import retrieve_from_file  # 导入从文件检索函数，内部支持索引缓存
from app.tool.base import BaseTool, ToolResult  # 导入工具基类和工具返回结果类


class RagSearchTool(BaseTool):  # 定义 RAG 检索工具
    name = "rag_search"  # 定义工具名称，模型调用时使用

    description = "从本地论文文本文件或 PDF 文件中检索与问题最相关的证据片段。"  # 定义工具描述并声明支持文本型 PDF

    parameters = {  # 定义工具参数 schema
        "type": "object",  # 参数整体是一个 JSON 对象
        "properties": {  # 定义参数字段
            "paper_path": {  # 定义 paper_path 参数
                "type": "string",  # paper_path 必须是字符串
                "description": "论文文件路径，例如 data/demo_paper.txt、data/demo_paper.md 或 data/demo_paper.pdf",  # 说明支持的论文文件类型
            },  # paper_path 参数定义结束
            "query": {  # 定义 query 参数
                "type": "string",  # query 必须是字符串
                "description": "用户想询问的问题，例如 这篇论文的核心方法是什么？",  # 说明 query 的含义
            },  # query 参数定义结束
            "top_k": {  # 定义 top_k 参数
                "type": "integer",  # top_k 必须是整数
                "description": "返回最相关片段数量，默认 3",  # 说明 top_k 的含义
            },  # top_k 参数定义结束
        },  # properties 定义结束
        "required": ["paper_path", "query"],  # paper_path 和 query 是必填参数
    }  # parameters 定义结束

    def __init__(self) -> None:  # 初始化 RAG 检索工具
        self.root_dir = Path(__file__).resolve().parents[2]  # 获取项目根目录 academic-agent-lab

    def execute(self, paper_path: str, query: str, top_k: int = 3) -> ToolResult:  # 执行 RAG 检索
        try:  # 捕获可能发生的读取或检索错误
            path = self._resolve_safe_path(paper_path)  # 将论文路径解析为安全路径

            if not path.exists():  # 判断文件是否存在
                return ToolResult(error=f"File not found: {paper_path}")  # 文件不存在时返回错误

            if not path.is_file():  # 判断路径是否是文件
                return ToolResult(error=f"Path is not a file: {paper_path}")  # 不是文件时返回错误

            results = retrieve_from_file(  # 调用从文件检索函数
                file_path=path,  # 传入论文文件路径
                query=query,  # 传入用户查询
                top_k=top_k,  # 传入返回片段数量
            )  # 检索结束

            if not results:  # 如果没有检索到结果
                return ToolResult(output="未检索到足够相关的证据片段。")  # 返回证据不足提示

            output = self._format_results(results)  # 将检索结果格式化为字符串

            return ToolResult(output=output)  # 返回格式化后的检索片段

        except Exception as exc:  # 捕获所有未预期异常
            return ToolResult(error=f"RagSearch error: {exc}")  # 将异常转换为 ToolResult 错误

    def _resolve_safe_path(self, paper_path: str) -> Path:  # 定义安全路径解析方法
        raw_path = Path(paper_path)  # 将字符串路径转换成 Path 对象

        if raw_path.is_absolute():  # 如果传入绝对路径
            candidate = raw_path.resolve()  # 解析绝对路径
        else:  # 如果传入相对路径
            candidate = (self.root_dir / raw_path).resolve()  # 拼接到项目根目录下再解析

        if self.root_dir not in candidate.parents and candidate != self.root_dir:  # 检查路径是否位于项目根目录内
            raise ValueError("只能检索项目目录内的文件")  # 如果越界访问，则抛出错误

        return candidate  # 返回安全路径

    def _format_results(self, results) -> str:  # 定义检索结果格式化方法
        lines = []  # 创建字符串列表，用来拼接输出

        for result in results:  # 遍历每个检索结果
            lines.append(f"[片段 {result.chunk_id}] score={result.score:.4f}")  # 添加片段编号和得分
            lines.append(result.text)  # 添加片段正文
            lines.append("")  # 添加空行分隔不同片段

        return "\n".join(lines).strip()  # 将所有行拼接成一个字符串并返回
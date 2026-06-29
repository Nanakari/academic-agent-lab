from app.agent.toolcall import ToolCallAgent  # 导入通用工具调用 Agent
from app.llm import LLM  # 导入 LLM 类型，用于类型标注
from app.schema import Message, Role  # 导入 Message 和 Role，用于写入系统提示词
from app.tool.calculator import CalculatorTool  # 导入计算器工具
from app.tool.file_reader import FileReaderTool  # 导入文件读取工具
from app.tool.file_writer import FileWriterTool  # 导入文件写入工具
from app.tool.paper_summary import PaperSummaryTool  # 导入论文摘要工具
from app.tool.terminate import TerminateTool  # 导入终止工具
from app.tool.tool_collection import ToolCollection  # 导入工具集合类
from app.tool.rag_search import RagSearchTool  # 导入 RAG 检索工具
from app.tool.paper_qa import PaperQATool  # 导入论文问答工具


class AcademicAgent(ToolCallAgent):  # 定义面向科研任务的具体 Agent
    def __init__(self, llm: LLM, max_steps: int = 10) -> None:  # 初始化 AcademicAgent
        tools = ToolCollection(  # 创建工具集合
            CalculatorTool(),  # 注册计算器工具
            FileReaderTool(),  # 注册文件读取工具
            FileWriterTool(),  # 注册文件写入工具
            PaperSummaryTool(llm=llm),  # 注册论文摘要工具，并传入 LLM
            TerminateTool(),  # 注册终止工具
            RagSearchTool(),  # 注册 RAG 检索工具
            PaperQATool(llm=llm),  # 注册论文问答工具
        )  # 工具集合创建结束

        super().__init__(  # 调用父类 ToolCallAgent 的初始化方法
            name="AcademicAgent",  # 设置 Agent 名称
            llm=llm,  # 传入 LLM 对象
            tools=tools,  # 传入工具集合
            max_steps=max_steps,  # 设置最大执行步数
        )  # 父类初始化结束

        self.academic_system_prompt = self._build_system_prompt()  # 构建并保存科研 Agent 的系统提示词

        self.memory.add_message(  # 将系统提示词写入 Memory
            Message(  # 创建系统消息
                role=Role.SYSTEM,  # 设置消息角色为 system
                content=self.academic_system_prompt,  # 设置系统提示词内容
            )  # 系统消息创建结束
        )  # 系统提示词写入 Memory 结束

    def _build_system_prompt(self) -> str:  # 定义系统提示词构建方法
        return (  # 返回系统提示词字符串
            "你是一个轻量级科研论文分析 Agent。\\n"  # 说明 Agent 角色
            "你的目标是帮助用户完成本地论文文本读取、结构化摘要生成和结果保存。\\n"  # 说明核心目标
            "你可以使用 file_reader 读取项目目录内的 txt、md 或文本型 PDF 文件。\\n"  # 说明 file_reader 支持的文档类型
            "你可以使用 paper_summary 根据论文文本生成结构化摘要。\\n"  # 说明 paper_summary 用途
            "你可以使用 file_writer 将结果保存到 outputs 目录。\\n"  # 说明 file_writer 用途
            "你可以使用 calculator 完成必要的数学计算。\\n"  # 说明 calculator 用途
            "任务完成后必须调用 terminate 工具结束任务。\\n"  # 要求任务完成后终止
            "处理论文总结任务时，推荐顺序是：file_reader -> paper_summary -> file_writer -> terminate。\\n"  # 明确推荐工具调用顺序
            "不要假装已经读取文件；必须先调用 file_reader 获得真实文本。\\n"  # 防止模型跳过文件读取
            "不要假装已经保存文件；必须调用 file_writer 写入结果。\\n"  # 防止模型跳过文件写入
            "如果工具返回错误，请根据错误信息调整下一步，不要编造成功结果。\\n"  # 要求处理工具错误
            "最终回答要简要说明已完成哪些步骤，以及结果保存在哪里。"  # 要求最终回答格式
            "当用户询问论文中的具体问题时，应该优先调用 rag_search 检索证据片段。\\n"
            "拿到检索片段后，应该调用 paper_qa 基于证据回答问题。\\n"
            "论文问答任务推荐顺序是：rag_search -> paper_qa -> terminate。\\n"
            "回答论文问题时，不要直接凭记忆回答，必须基于检索片段。\\n"  # 强调论文回答必须使用真实检索证据
            "你可以使用 rag_search 从 txt、md 或 pdf 论文文件中检索证据片段。\\n"  # 说明 rag_search 支持的文档类型
            "如果是论文问答任务，优先使用 rag_search 检索证据，再调用 paper_qa 基于证据回答。\\n"  # 强调论文问答的工具调用顺序
            "当前 PDF 支持仅限可提取文本的 PDF，扫描版 PDF 暂不支持 OCR。\\n"  # 说明当前 PDF 能力边界
        )  # 系统提示词返回结束
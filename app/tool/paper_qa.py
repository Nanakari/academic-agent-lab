from app.llm import LLM  # 导入 LLM 封装类，用于生成基于证据的回答
from app.tool.base import BaseTool, ToolResult  # 导入工具基类和工具返回结果类


class PaperQATool(BaseTool):  # 定义论文问答工具
    name = "paper_qa"  # 定义工具名称，模型调用时使用

    description = "根据 RAG 检索到的证据片段回答论文相关问题，回答必须引用片段编号。"  # 定义工具描述

    parameters = {  # 定义工具参数 schema
        "type": "object",  # 参数整体是一个 JSON 对象
        "properties": {  # 定义参数字段
            "question": {  # 定义 question 参数
                "type": "string",  # question 必须是字符串
                "description": "用户提出的论文问题",  # 说明 question 的含义
            },  # question 参数定义结束
            "retrieved_context": {  # 定义 retrieved_context 参数
                "type": "string",  # retrieved_context 必须是字符串
                "description": "rag_search 返回的证据片段文本",  # 说明 retrieved_context 的含义
            },  # retrieved_context 参数定义结束
        },  # properties 定义结束
        "required": ["question", "retrieved_context"],  # question 和 retrieved_context 是必填参数
    }  # parameters 定义结束

    def __init__(self, llm: LLM) -> None:  # 初始化论文问答工具
        self.llm = llm  # 保存 LLM 对象，用于生成回答

    def execute(self, question: str, retrieved_context: str) -> ToolResult:  # 执行论文问答
        try:  # 捕获模型调用错误
            prompt = self._build_prompt(question=question, retrieved_context=retrieved_context)  # 构建问答提示词

            messages = [  # 构造 LLM.ask 需要的消息列表
                {  # 构造 system 消息
                    "role": "system",  # 设置角色为 system
                    "content": "你是一个严谨的论文问答助手，只能根据给定证据片段回答问题。",  # 设置 system 内容
                },  # system 消息结束
                {  # 构造 user 消息
                    "role": "user",  # 设置角色为 user
                    "content": prompt,  # 设置用户消息内容
                },  # user 消息结束
            ]  # 消息列表结束

            answer = self.llm.ask(messages)  # 调用 LLM 生成回答

            return ToolResult(output=answer)  # 返回回答文本

        except Exception as exc:  # 捕获所有未预期异常
            return ToolResult(error=f"PaperQA error: {exc}")  # 返回工具错误

    def _build_prompt(self, question: str, retrieved_context: str) -> str:  # 构建问答提示词
        return (  # 返回完整提示词
            "请根据下面的检索证据片段回答问题。\\n\\n"  # 说明任务
            "要求：\\n"  # 引出要求
            "- 回答必须基于证据片段，不要编造。\\n"  # 要求不编造
            "- 回答中必须包含片段编号，例如“依据片段 2”。\\n"  # 要求引用片段编号
            "- 如果证据不足，请明确说“证据不足”。\\n"  # 要求证据不足时说明
            "- 使用中文回答，结构清晰。\\n\\n"  # 要求输出中文和结构化
            f"问题：{question}\\n\\n"  # 插入用户问题
            f"证据片段：\\n{retrieved_context}"  # 插入检索上下文
        )  # 提示词构造结束
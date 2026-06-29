from app.llm import LLM  # 导入 LLM 封装类，用于调用大模型生成摘要
from app.tool.base import BaseTool, ToolResult  # 导入工具基类和工具返回结果类


class PaperSummaryTool(BaseTool):  # 定义论文摘要工具，继承统一工具基类
    name = "paper_summary"  # 定义工具名称，模型调用工具时会使用这个名字

    description = "根据论文文本生成结构化中文摘要，包括研究问题、方法、实验、贡献、局限性和可复现性。"  # 定义工具描述

    parameters = {  # 定义工具参数 schema
        "type": "object",  # 参数整体是一个 JSON 对象
        "properties": {  # 定义对象内部字段
            "paper_text": {  # 定义 paper_text 参数
                "type": "string",  # paper_text 必须是字符串
                "description": "论文全文或论文主要内容文本",  # 说明 paper_text 的含义
            },  # paper_text 参数定义结束
            "language": {  # 定义 language 参数
                "type": "string",  # language 必须是字符串
                "description": "摘要语言，默认 zh",  # 说明 language 的含义
            },  # language 参数定义结束
        },  # properties 定义结束
        "required": ["paper_text"],  # paper_text 是必填参数
    }  # parameters 定义结束

    def __init__(self, llm: LLM) -> None:  # 初始化论文摘要工具
        self.llm = llm  # 保存 LLM 对象，后续生成摘要时使用

    def execute(self, paper_text: str, language: str = "zh") -> ToolResult:  # 执行论文摘要生成逻辑
        try:  # 捕获可能发生的模型调用错误
            prompt = self._build_prompt(paper_text=paper_text, language=language)  # 构建摘要提示词

            messages = [  # 构造传给 LLM.ask() 的消息列表
                {  # 创建 system 消息字典
                    "role": "system",  # 设置消息角色为 system
                    "content": "你是一个严谨的科研论文分析助手，擅长生成结构化论文摘要。",  # 设置 system 内容
                },  # system 消息字典结束
                {  # 创建 user 消息字典
                    "role": "user",  # 设置消息角色为 user
                    "content": prompt,  # 设置用户消息内容为摘要提示词
                },  # user 消息字典结束
            ]  # 消息列表构造结束

            summary = self.llm.ask(messages)  # 调用 LLM 生成论文摘要

            return ToolResult(output=summary)  # 将摘要文本作为工具结果返回

        except Exception as exc:  # 捕获所有未预期异常
            return ToolResult(error=f"PaperSummary error: {exc}")  # 将异常转换为 ToolResult 错误

    def _build_prompt(self, paper_text: str, language: str) -> str:  # 定义构建摘要提示词的方法
        return (  # 返回完整提示词
            f"请使用 {language} 生成下面论文文本的结构化摘要。\\n\\n"  # 指定摘要语言
            "摘要必须包含以下部分：\\n"  # 要求固定结构
            "1. 研究问题\\n"  # 要求输出研究问题
            "2. 核心方法\\n"  # 要求输出核心方法
            "3. 实验设置\\n"  # 要求输出实验设置
            "4. 主要结果\\n"  # 要求输出主要结果
            "5. 主要贡献\\n"  # 要求输出主要贡献
            "6. 局限性\\n"  # 要求输出局限性
            "7. 可复现性分析\\n"  # 要求输出可复现性分析
            "8. 一句话总结\\n\\n"  # 要求输出一句话总结
            "要求：\\n"  # 给出额外要求
            "- 不要编造论文中没有的信息。\\n"  # 防止幻觉
            "- 如果文本中没有提到某项内容，请写“原文未明确说明”。\\n"  # 要求缺失信息明确说明
            "- 使用 Markdown 格式。\\n"  # 要求 Markdown 输出
            "- 表达要适合研究生组会汇报。\\n\\n"  # 要求风格适合学术汇报
            "论文文本如下：\\n"  # 引出论文文本
            f"{paper_text}"  # 插入论文文本
        )  # 提示词构建结束
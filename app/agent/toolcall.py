from app.agent.react import ReActAgent  # 从 ReActAgent 模块中导入 ReActAgent，ToolCallAgent 要继承它

from app.llm import LLM  # 从 llm.py 中导入 LLM，用来调用大模型

from app.schema import AgentState, ToolCall, ToolChoice  # 从 schema.py 中导入 AgentState、ToolCall、ToolChoice

from app.tool.tool_collection import ToolCollection  # 从 tool_collection.py 中导入 ToolCollection，用来统一管理工具


class ToolCallAgent(ReActAgent):  # 定义 ToolCallAgent，它继承 ReActAgent
    def __init__(self, name: str, llm: LLM, tools: ToolCollection, max_steps: int = 5) -> None:  # 定义初始化方法
        super().__init__(name=name, max_steps=max_steps)  # 调用父类 BaseAgent 的初始化方法，设置 name、max_steps、state、memory 等

        self.llm = llm  # 保存 LLM 对象，后续 think() 会用它调用模型

        self.tools = tools  # 保存 ToolCollection 对象，后续 act() 会用它执行工具

        self.tool_calls: list[ToolCall] = []  # 保存当前这一步模型返回的工具调用列表

        self.final_answer = ""  # 保存最终答案，可能来自模型普通文本，也可能来自 paper_qa / paper_summary 工具结果

        self.final_answer_tool_names = {  # 定义哪些工具的输出可以作为最终答案
            "paper_qa",  # paper_qa 用于论文问答，它的输出通常就是用户要的答案
            "paper_summary",  # paper_summary 用于论文摘要，它的输出通常也是用户要的答案
        }  # 最终答案工具集合定义结束

        self.memory.add_system_message(  # 给 Agent 的 Memory 添加系统提示词
            "You are a tool-calling agent. "  # 说明当前助手是一个工具调用 Agent
            "If a calculation is needed, call the calculator tool. "  # 要求遇到计算任务时调用 calculator
            "If a paper question needs evidence, call the rag_search tool first. "  # 要求论文问答先调用 rag_search 检索证据
            "After retrieving evidence, call paper_qa to answer based on the evidence. "  # 要求拿到证据后调用 paper_qa 生成答案
            "After receiving tool results, answer the user clearly. "  # 要求拿到工具结果后清楚回答用户
            "When the task is finished, call the terminate tool."  # 说明任务完成时调用 terminate
        )  # 系统提示词添加结束

    def think(self) -> bool:  # 实现 ReActAgent 要求的 think 方法
        print(f"Step {self.current_step}: thinking...")  # 打印当前正在思考，方便观察执行过程

        content, tool_calls = self.llm.ask_tool(  # 调用 LLM，让模型基于上下文和工具 schema 进行判断
            messages=self.memory.get_messages(),  # 传入当前 Memory 中的所有消息
            tools=self.tools.to_params(),  # 传入所有工具的 schema
            tool_choice=ToolChoice.AUTO,  # 让模型自动判断是否需要调用工具
        )  # ask_tool 调用结束

        self.tool_calls = tool_calls  # 把模型返回的工具调用保存到当前 Agent 中，供 act() 使用

        if content:  # 如果模型返回了普通文本内容
            self.memory.add_assistant_message(content)  # 把模型文本回答写入 Memory

            self.final_answer = content  # 把模型文本回答保存为候选最终答案

        if self.tool_calls:  # 如果模型返回了一个或多个工具调用
            for tool_call in self.tool_calls:  # 遍历每一个工具调用
                print(f"Tool call requested: {tool_call.name} {tool_call.arguments}")  # 打印模型想调用的工具名和参数

            return True  # 返回 True，表示接下来需要进入 act() 执行工具

        self.state = AgentState.FINISHED  # 如果模型没有返回工具调用，说明任务可以结束

        return False  # 返回 False，表示不需要继续执行 act()

    def act(self) -> str:  # 实现 ReActAgent 要求的 act 方法
        print(f"Step {self.current_step}: acting...")  # 打印当前正在行动，方便观察执行过程

        observations = []  # 创建列表，用来保存这一轮所有工具执行结果

        for tool_call in self.tool_calls:  # 遍历当前这一轮需要执行的所有工具调用
            observation = self.execute_tool(tool_call)  # 执行单个工具调用，并得到观察结果文本

            observations.append(observation)  # 把观察结果加入 observations 列表

        return "\n".join(observations)  # 把多个工具观察结果合并成一个字符串返回

    def execute_tool(self, tool_call: ToolCall) -> str:  # 定义执行单个工具调用的方法
        result = self.tools.execute(tool_call.name, tool_call.arguments)  # 通过 ToolCollection 按名称执行工具

        if result.is_success:  # 如果工具执行成功
            observation = result.output or ""  # 取出工具输出，如果为空就用空字符串

        else:  # 如果工具执行失败
            observation = result.error or "Unknown tool error."  # 取出错误信息，如果为空就用默认错误

        print(f"Tool result: {observation}")  # 打印工具执行结果，方便观察

        self.memory.add_tool_message(  # 把工具结果写回 Memory
            content=observation,  # 工具结果内容
            name=tool_call.name,  # 工具名称
            tool_call_id=tool_call.id,  # 工具调用 id
        )  # 工具消息添加结束

        if result.is_success and tool_call.name in self.final_answer_tool_names:  # 如果当前工具成功执行，并且它的输出可以作为最终答案
            self.final_answer = observation  # 把该工具的输出保存为最终答案

        if tool_call.name == "terminate":  # 如果当前调用的是 terminate 工具
            self.state = AgentState.FINISHED  # 把 Agent 状态设置为 FINISHED

        return observation  # 返回工具观察结果
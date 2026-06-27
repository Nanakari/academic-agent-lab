from uuid import uuid4  # 从 uuid 模块导入 uuid4，用来在模型没有返回工具调用 id 时生成一个临时 id

from google import genai  # 从 google 包中导入 genai，用来创建 Gemini API 客户端

from google.genai import types  # 从 google.genai 导入 types，用来设置生成参数和工具声明

from app.schema import ToolCall, ToolChoice  # 从 app/schema.py 导入 ToolCall 和 ToolChoice


class LLM:  # 定义 LLM 类，专门封装大模型调用逻辑
    def __init__(self, config: dict):  # 定义初始化方法，创建 LLM 对象时会自动执行
        llm_config = config.get("llm", {})  # 从总配置中取出 llm 配置，如果没有就使用空字典

        self.model = llm_config.get("model", "gemini-3.1-flash-lite")  # 读取模型名称，如果配置中没有就使用默认模型

        self.api_key = llm_config.get("api_key", "")  # 读取 API key，如果配置中没有就使用空字符串

        self.temperature = float(llm_config.get("temperature", 0.0))  # 读取 temperature，并转换成浮点数

        self.max_tokens = int(llm_config.get("max_tokens", 1024))  # 读取最大输出 token 数，并转换成整数

        if not self.api_key:  # 判断 API key 是否为空
            raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY or write api_key in config.toml.")  # 如果没有 API key，就抛出错误并提示解决方法

        self.client = genai.Client(api_key=self.api_key)  # 创建 Gemini API 客户端，后面所有模型调用都通过它完成

    def ask(self, messages: list[dict]) -> str:  # 定义普通问答方法，输入 messages，输出模型回答文本
        prompt = self._messages_to_prompt(messages)  # 把 messages 列表转换成 Gemini 可以接收的 prompt 字符串

        response = self.client.models.generate_content(  # 调用 Gemini 的 generate_content 接口生成回答
            model=self.model,  # 指定要调用的模型名称
            contents=prompt,  # 把转换后的 prompt 作为输入内容传给模型
            config=types.GenerateContentConfig(  # 设置本次生成的参数
                temperature=self.temperature,  # 设置输出随机性，0.0 表示更稳定
                max_output_tokens=self.max_tokens,  # 设置最大输出 token 数
            ),  # GenerateContentConfig 参数设置结束
        )  # generate_content 调用结束

        return self._extract_text(response)  # 从 Gemini 响应中安全提取纯文本，避免直接访问 response.text

    def ask_tool(self, messages: list[dict], tools: list[dict], tool_choice: ToolChoice = ToolChoice.AUTO) -> tuple[str, list[ToolCall]]:  # 定义带工具声明的模型调用方法
        prompt = self._messages_to_prompt(messages)  # 把 messages 列表转换成普通 prompt 字符串

        function_declarations = self._tools_to_gemini_declarations(tools)  # 把 OpenAI 风格工具 schema 转成 Gemini function declarations

        gemini_tool = types.Tool(function_declarations=function_declarations)  # 创建 Gemini 工具声明对象

        generate_config = types.GenerateContentConfig(  # 创建 Gemini 生成配置
            temperature=self.temperature,  # 设置输出随机性
            max_output_tokens=self.max_tokens,  # 设置最大输出 token 数
            tools=[gemini_tool],  # 把工具声明传给模型
        )  # 生成配置结束

        response = self.client.models.generate_content(  # 调用 Gemini，并允许模型选择是否调用工具
            model=self.model,  # 指定模型名称
            contents=prompt,  # 传入对话上下文 prompt
            config=generate_config,  # 传入包含工具声明的生成配置
        )  # generate_content 调用结束

        content = self._extract_text(response)  # 从 Gemini 响应中安全提取纯文本，避免 function_call 响应触发 warning

        tool_calls = self._extract_tool_calls(response)  # 从模型响应中提取工具调用列表

        if tool_choice == ToolChoice.NONE:  # 如果外部指定不允许工具调用
            return content, []  # 就忽略模型可能返回的工具调用，只返回文本和空工具调用列表

        return content, tool_calls  # 返回普通文本内容和工具调用列表

    def _messages_to_prompt(self, messages: list[dict]) -> str:  # 定义内部辅助方法，把 messages 转成普通文本 prompt
        lines = []  # 创建一个空列表，用来存放每一行对话文本

        for message in messages:  # 遍历 messages 中的每一条消息
            role = message.get("role", "user")  # 读取消息角色，如果没有 role 就默认是 user

            content = message.get("content", "")  # 读取消息内容，如果没有 content 就默认是空字符串

            name = message.get("name")  # 读取消息名称，工具消息可能会有 name

            if name:  # 如果 name 有值
                lines.append(f"{role}({name}): {content}")  # 把角色、名称和内容拼成一行文本

            else:  # 如果 name 没有值
                lines.append(f"{role}: {content}")  # 把角色和内容拼成一行文本

        return "\n".join(lines)  # 用换行符把多行文本拼成一个完整 prompt 并返回

    def _extract_text(self, response) -> str:  # 定义方法，从 Gemini 响应中安全提取纯文本内容
        texts = []  # 创建空列表，用来保存所有文本片段

        candidates = getattr(response, "candidates", []) or []  # 从 response 中读取 candidates，如果没有就使用空列表

        for candidate in candidates:  # 遍历每一个候选回答
            content = getattr(candidate, "content", None)  # 读取 candidate.content

            parts = getattr(content, "parts", []) if content else []  # 如果 content 存在，就读取 parts，否则使用空列表

            for part in parts:  # 遍历 content.parts 中的每一个 part
                text = getattr(part, "text", None)  # 尝试读取当前 part 的 text 字段

                if text:  # 如果当前 part 中确实有文本内容
                    texts.append(text)  # 把文本内容加入 texts 列表

        return "".join(texts)  # 把所有文本片段拼接成一个完整字符串并返回

    def _tools_to_gemini_declarations(self, tools: list[dict]) -> list[dict]:  # 定义方法，把工具 schema 转成 Gemini 需要的 function declarations
        declarations = []  # 创建空列表，用来保存转换后的工具声明

        for tool in tools:  # 遍历工具 schema 列表
            if "function" in tool:  # 如果工具 schema 是 OpenAI 风格，也就是包含 function 字段
                function_data = tool["function"]  # 取出 function 字段中的工具说明

                declarations.append(  # 把转换后的工具声明加入列表
                    {  # 创建 Gemini function declaration 字典
                        "name": function_data["name"],  # 设置工具名称
                        "description": function_data["description"],  # 设置工具描述
                        "parameters": function_data["parameters"],  # 设置工具参数 schema
                    }  # Gemini function declaration 字典结束
                )  # declarations.append 调用结束

            else:  # 如果工具 schema 已经是 Gemini 风格
                declarations.append(tool)  # 就直接加入 declarations

        return declarations  # 返回 Gemini function declarations 列表

    def _extract_tool_calls(self, response) -> list[ToolCall]:  # 定义方法，从 Gemini 响应中提取工具调用
        tool_calls = []  # 创建空列表，用来保存 ToolCall 对象

        response_function_calls = getattr(response, "function_calls", None)  # 尝试从 response.function_calls 读取工具调用

        if response_function_calls:  # 如果 response.function_calls 有内容
            for function_call in response_function_calls:  # 遍历每一个工具调用
                tool_calls.append(self._function_call_to_tool_call(function_call))  # 把 Gemini function_call 转成项目内部 ToolCall

            return tool_calls  # 返回工具调用列表

        candidates = getattr(response, "candidates", []) or []  # 如果没有 response.function_calls，就尝试从 candidates 中读取

        for candidate in candidates:  # 遍历每一个候选回答
            content = getattr(candidate, "content", None)  # 取出候选回答中的 content

            parts = getattr(content, "parts", []) if content else []  # 取出 content.parts，如果 content 为空就用空列表

            for part in parts:  # 遍历每一个 part
                function_call = getattr(part, "function_call", None)  # 尝试读取 part.function_call

                if function_call:  # 如果当前 part 里有 function_call
                    tool_calls.append(self._function_call_to_tool_call(function_call))  # 转换成 ToolCall 并加入列表

        return tool_calls  # 返回工具调用列表

    def _function_call_to_tool_call(self, function_call) -> ToolCall:  # 定义方法，把 Gemini function_call 转成项目内部 ToolCall
        call_id = getattr(function_call, "id", None) or str(uuid4())  # 读取工具调用 id，如果没有就生成一个 uuid

        name = getattr(function_call, "name", "")  # 读取工具名称

        args = getattr(function_call, "args", {}) or {}  # 读取工具参数，如果为空就用空字典

        return ToolCall(id=call_id, name=name, arguments=dict(args))  # 创建并返回 ToolCall 对象
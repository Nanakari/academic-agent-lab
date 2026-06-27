from app.agent.toolcall import ToolCallAgent  # 从 toolcall.py 中导入 ToolCallAgent

from app.config import load_config  # 从 config.py 中导入 load_config，用来读取配置文件

from app.llm import LLM  # 从 llm.py 中导入 LLM 类

from app.tool.calculator import CalculatorTool  # 从 calculator.py 中导入 CalculatorTool

from app.tool.terminate import TerminateTool  # 从 terminate.py 中导入 TerminateTool

from app.tool.tool_collection import ToolCollection  # 从 tool_collection.py 中导入 ToolCollection


def main() -> None:  # 定义主函数，作为程序入口
    config = load_config()  # 读取 config.toml，并用环境变量中的 GEMINI_API_KEY 覆盖配置中的 api_key

    llm = LLM(config)  # 创建 LLM 对象，用来调用 Gemini

    tools = ToolCollection(  # 创建 ToolCollection 工具集合
        CalculatorTool(),  # 注册 calculator 工具
        TerminateTool(),  # 注册 terminate 工具
    )  # 工具集合创建结束

    agent = ToolCallAgent(  # 创建 ToolCallAgent 对象
        name="tool-call-agent",  # 设置 Agent 名称
        llm=llm,  # 传入 LLM 对象
        tools=tools,  # 传入工具集合
        max_steps=5,  # 设置最大执行步数，防止死循环
    )  # ToolCallAgent 创建结束

    user_request = "计算 123 * 456，并告诉我结果。"  # 准备一个测试任务

    result = agent.run(user_request)  # 调用 Agent 主循环执行任务

    print("Final result:")  # 打印最终结果标题
    print(result)  # 打印 Agent.run() 返回的结果

    print("Final answer:")  # 打印最终回答标题
    print(agent.final_answer)  # 打印模型最终自然语言回答

    print("Final state:")  # 打印最终状态标题
    print(agent.state)  # 打印 Agent 最终状态

    print("Current step:")  # 打印当前步数标题
    print(agent.current_step)  # 打印 Agent 最终执行步数

    print("Memory messages:")  # 打印 Memory 标题
    for message in agent.memory.get_messages():  # 遍历 Memory 中所有消息
        print(message)  # 打印每一条消息字典


if __name__ == "__main__":  # 判断当前文件是否被直接运行
    main()  # 调用主函数
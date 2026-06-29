from app.config import load_config  # 导入配置读取函数
from app.llm import LLM  # 导入 LLM 封装类
from app.agent.academic_agent import AcademicAgent  # 导入科研 Agent


def main() -> None:  # 定义程序入口函数
    config = load_config()  # 读取 config.toml 和环境变量配置

    llm = LLM(config)  # 根据配置创建 LLM 对象

    agent = AcademicAgent(llm=llm, max_steps=10)  # 创建 AcademicAgent，并设置最大执行步数

    # 原 txt 测试任务如下，可取消注释后用于回归测试。  # 保留原来的 txt 测试任务说明
    # user_request = (  # 定义原来的 txt embedding RAG 问答验收任务
    #     "请回答：这篇论文的核心方法是什么？"  # 提出中文论文问题
    #     "请严格按以下顺序完成："  # 要求模型按工具链执行
    #     "第一步调用 rag_search，从 data/demo_paper.txt 检索相关证据片段；"  # 要求先从 txt 检索证据
    #     "第二步调用 paper_qa，基于检索片段回答问题；"  # 要求基于证据回答
    #     "第三步调用 terminate 结束任务。"  # 要求最后终止
    # )  # 原 txt 用户任务定义结束

    # 运行前请把一个可复制文字的文本型 PDF 放到 data/demo_paper.pdf。  # 提醒准备当前测试所需的 PDF 文件
    user_request = (  # 定义文本型 PDF 的 embedding RAG 问答验收任务
        "请回答：这篇论文的核心方法是什么？"  # 提出中文论文问题
        "请严格按以下顺序完成："  # 要求模型按工具链执行
        "第一步调用 rag_search，从 data/demo_paper.pdf 检索相关证据片段；"  # 要求先从 PDF 检索证据
        "第二步调用 paper_qa，基于检索片段回答问题；"  # 要求基于证据回答
        "第三步调用 terminate 结束任务。"  # 要求最后终止
    )  # PDF 用户任务定义结束

    result = agent.run(user_request)  # 运行 Agent 主循环

    print("Final result:")  # 打印最终工具结果标题
    print(result)  # 打印 Agent run 返回结果

    print("Final answer:")  # 打印最终回答标题
    print(agent.final_answer)  # 打印 Agent 最终回答

    print("Final state:")  # 打印最终状态标题
    print(agent.state)  # 打印 Agent 状态

    print("Current step:")  # 打印当前步数标题
    print(agent.current_step)  # 打印实际执行步数


if __name__ == "__main__":  # 判断当前文件是否作为主程序运行
    main()  # 调用主函数
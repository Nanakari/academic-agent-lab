from abc import ABC, abstractmethod  # 导入 ABC 和 abstractmethod，用来定义抽象基类和抽象方法
from app.schema import AgentState, Memory  # 导入 AgentState 和 Memory，BaseAgent 需要状态和记忆


class BaseAgent(ABC):  # 定义 BaseAgent 抽象基类，所有具体 Agent 都会继承它
    def __init__(self, name: str, max_steps: int = 5) -> None:  # 定义初始化方法，name 是 Agent 名称，max_steps 是最大执行步数
        self.name = name  # 保存 Agent 名称
        self.max_steps = max_steps  # 保存最大执行步数，用来防止 Agent 无限循环
        self.current_step = 0  # 当前已经执行了多少步，初始为 0
        self.state = AgentState.IDLE  # Agent 初始状态是 IDLE，表示还没有开始执行
        self.memory = Memory()  # 每个 Agent 都有自己的 Memory，用来保存任务上下文

    def run(self, request: str) -> str:  # 定义 Agent 的主运行方法，request 是用户任务
        self.state = AgentState.RUNNING  # 开始执行任务时，把状态设置为 RUNNING
        self.current_step = 0  # 每次 run 开始时，把当前步数重置为 0
        self.memory.add_user_message(request)  # 把用户任务写入 Memory

        result = ""  # 创建 result 变量，用来保存最后一次 step 的结果

        while self.state == AgentState.RUNNING:  # 只要 Agent 状态仍然是 RUNNING，就继续循环
            if self.current_step >= self.max_steps:  # 如果当前步数已经达到最大步数
                self.state = AgentState.FINISHED  # 把状态设置为 FINISHED，表示因为达到步数上限而结束
                result = "Reached max steps."  # 设置返回结果，说明是达到最大步数结束
                break  # 跳出循环

            self.current_step += 1  # 执行新一步之前，步数加 1

            try:  # 开始捕获 step 执行过程中的异常
                result = self.step()  # 调用子类实现的 step 方法，执行一步动作
            except Exception as error:  # 如果 step 过程中出现异常
                self.state = AgentState.ERROR  # 把状态设置为 ERROR
                result = f"Agent error: {error}"  # 把错误信息保存到 result
                break  # 出错后跳出循环

        return result  # 返回最终结果

    @abstractmethod  # 标记下面的方法是抽象方法，子类必须实现
    def step(self) -> str:  # 定义单步执行方法，但 BaseAgent 不实现具体逻辑
        pass  # 抽象方法占位，具体行为交给子类
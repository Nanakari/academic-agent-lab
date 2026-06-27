from dataclasses import dataclass  # 从 dataclasses 导入 dataclass，用来快速定义数据类

from enum import Enum  # 从 enum 导入 Enum，用来定义枚举类型


class Role(str, Enum):  # 定义消息角色枚举，继承 str 是为了方便转成 API 需要的字符串
    SYSTEM = "system"  # system 表示系统提示词消息
    USER = "user"  # user 表示用户输入消息
    ASSISTANT = "assistant"  # assistant 表示模型回答消息
    TOOL = "tool"  # tool 表示工具返回消息，后续工具调用结果会用到


class AgentState(str, Enum):  # 定义 Agent 状态枚举，用来描述 Agent 当前生命周期阶段
    IDLE = "idle"  # IDLE 表示空闲状态，还没有开始执行任务
    RUNNING = "running"  # RUNNING 表示正在执行任务
    FINISHED = "finished"  # FINISHED 表示任务已经正常完成
    ERROR = "error"  # ERROR 表示任务执行过程中发生错误


class ToolChoice(str, Enum):  # 定义工具选择模式枚举，用来描述是否允许模型调用工具
    NONE = "none"  # none 表示不允许模型调用工具，只能直接回答
    AUTO = "auto"  # auto 表示让模型自己判断是否需要调用工具
    REQUIRED = "required"  # required 表示要求模型必须调用某个工具


@dataclass  # 使用 dataclass 自动生成 __init__ 等基础方法
class ToolCall:  # 定义 ToolCall 类，用来表示模型想要调用的一个工具
    id: str  # id 表示本次工具调用的唯一编号，用来对应工具请求和工具结果
    name: str  # name 表示工具名称，例如 calculator 或 terminate
    arguments: dict  # arguments 表示工具参数，例如 {"expression": "123 * 456"}

    def to_dict(self) -> dict:  # 定义方法，把 ToolCall 对象转换成普通字典
        return {  # 返回一个普通 Python 字典
            "id": self.id,  # 保存工具调用 id
            "name": self.name,  # 保存工具名称
            "arguments": self.arguments,  # 保存工具参数
        }  # 字典构造结束


@dataclass  # 使用 dataclass 自动生成 __init__ 等基础方法
class Message:  # 定义 Message 类，用来表示一条消息
    role: Role  # role 表示消息角色，例如 system、user、assistant、tool
    content: str  # content 表示消息正文内容
    name: str | None = None  # name 是可选字段，工具消息可以用它记录工具名称
    tool_call_id: str | None = None  # tool_call_id 是可选字段，用来标记这条工具结果对应哪次工具调用

    def to_dict(self) -> dict:  # 定义方法，把 Message 对象转换成普通字典
        data = {  # 创建基础字典
            "role": self.role.value,  # 把 Role 枚举转换成字符串，例如 Role.USER 转成 "user"
            "content": self.content,  # 保存消息正文
        }  # 基础字典结束

        if self.name is not None:  # 如果 name 有值
            data["name"] = self.name  # 就把 name 写入字典

        if self.tool_call_id is not None:  # 如果 tool_call_id 有值
            data["tool_call_id"] = self.tool_call_id  # 就把 tool_call_id 写入字典

        return data  # 返回转换后的字典


class Memory:  # 定义 Memory 类，用来保存当前任务运行期间的短期消息历史
    def __init__(self) -> None:  # 定义初始化方法
        self.messages: list[Message] = []  # 创建空列表，用来保存 Message 对象

    def add_message(self, message: Message) -> None:  # 定义通用消息添加方法
        self.messages.append(message)  # 把 Message 对象追加到列表末尾

    def add_system_message(self, content: str) -> None:  # 定义添加 system 消息的方法
        self.add_message(Message(role=Role.SYSTEM, content=content))  # 创建 system 消息并加入 Memory

    def add_user_message(self, content: str) -> None:  # 定义添加 user 消息的方法
        self.add_message(Message(role=Role.USER, content=content))  # 创建 user 消息并加入 Memory

    def add_assistant_message(self, content: str) -> None:  # 定义添加 assistant 消息的方法
        self.add_message(Message(role=Role.ASSISTANT, content=content))  # 创建 assistant 消息并加入 Memory

    def add_tool_message(self, content: str, name: str, tool_call_id: str) -> None:  # 定义添加 tool 消息的方法
        self.add_message(Message(role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id))  # 创建工具结果消息并加入 Memory

    def get_messages(self) -> list[dict]:  # 定义获取消息列表的方法
        return [message.to_dict() for message in self.messages]  # 把所有 Message 对象转换成 dict 列表

    def clear(self) -> None:  # 定义清空 Memory 的方法
        self.messages.clear()  # 清空消息列表
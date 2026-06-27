from pathlib import Path  # 从 pathlib 导入 Path，用来更方便地处理文件路径
import os  # 导入 os 模块，用来读取系统环境变量
import tomllib  # 导入 tomllib，用来读取 Python 3.11+ 原生支持的 TOML 配置文件


ROOT_DIR = Path(__file__).resolve().parent.parent  # 获取项目根目录，也就是 academic-agent-lab 目录
CONFIG_PATH = ROOT_DIR / "config.toml"  # 拼接出 config.toml 的完整路径


def load_config() -> dict:  # 定义一个函数，用来读取配置文件，并返回字典
    if not CONFIG_PATH.exists():  # 判断 config.toml 是否存在
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")  # 如果不存在，就抛出文件不存在错误

    with CONFIG_PATH.open("rb") as f:  # 以二进制读取模式打开 config.toml，这是 tomllib 要求的方式
        config = tomllib.load(f)  # 使用 tomllib 读取 TOML 文件，并转换成 Python 字典

    llm_config = config.get("llm", {})  # 从配置字典中取出 [llm] 这一组配置，如果没有就用空字典

    env_api_key = os.getenv("GEMINI_API_KEY", "")  # 从环境变量中读取 GEMINI_API_KEY，如果不存在就返回空字符串

    if env_api_key:  # 判断环境变量里是否真的有 API key
        llm_config["api_key"] = env_api_key  # 如果有，就用环境变量里的 API key 覆盖 config.toml 里的 api_key

    config["llm"] = llm_config  # 把更新后的 llm_config 放回总配置字典中

    return config  # 返回最终配置字典
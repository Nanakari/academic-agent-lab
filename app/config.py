"""Configuration loading helpers."""

from pathlib import Path
import os

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility.
    import tomli as tomllib


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"
EXAMPLE_CONFIG_PATH = ROOT_DIR / "config.example.toml"


def load_config() -> dict:
    """Load local config, falling back to the tracked example template."""
    config_path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            "Config file not found. Expected config.toml or config.example.toml "
            f"under {ROOT_DIR}."
        )

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    llm_config = dict(config.get("llm", {}))
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    if env_api_key:
        llm_config["api_key"] = env_api_key

    config["llm"] = llm_config
    config["_config_path"] = str(config_path)
    return config

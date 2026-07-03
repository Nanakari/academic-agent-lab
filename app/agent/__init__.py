"""Agent package exports without eagerly importing optional dependencies."""

__all__ = ["AIScientificAgent"]


def __getattr__(name: str):
    """Load the scientific agent only when requested from the package."""
    if name == "AIScientificAgent":
        from app.agent.ai_scientific_agent import AIScientificAgent

        return AIScientificAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

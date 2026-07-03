"""Idea-specific view over ScientificMemory."""

from app.memory.scientific_memory import ScientificMemory


class IdeaMemory:
    def __init__(self, memory: ScientificMemory) -> None:
        self.memory = memory

    def save(self, idea: dict) -> None:
        self.memory.save_idea(idea)

    def recent(self, limit: int = 10) -> list[dict]:
        return self.memory.load_recent_ideas(limit)

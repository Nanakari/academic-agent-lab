"""Paper-specific view over ScientificMemory."""

from app.memory.scientific_memory import ScientificMemory


class PaperMemory:
    def __init__(self, memory: ScientificMemory) -> None:
        self.memory = memory

    def save(self, note: dict) -> None:
        self.memory.save_paper_note(note)

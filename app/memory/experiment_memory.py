"""Experiment-specific view over ScientificMemory."""

from app.memory.scientific_memory import ScientificMemory


class ExperimentMemory:
    def __init__(self, memory: ScientificMemory) -> None:
        self.memory = memory

    def save(self, experiment: dict) -> None:
        self.memory.save_experiment(experiment)

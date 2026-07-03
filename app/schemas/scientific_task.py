"""Scientific task types understood by the research planner."""

from enum import Enum


class ScientificTaskType(str, Enum):
    """High-level intents supported by the MVP."""

    PAPER_READING = "paper_reading"
    LITERATURE_ANALYSIS = "literature_analysis"
    IDEA_GENERATION = "idea_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    RESEARCH_PROPOSAL = "research_proposal"

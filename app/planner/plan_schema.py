"""Structured schemas for research plans."""

from dataclasses import asdict, dataclass, field

from app.schemas.scientific_task import ScientificTaskType


@dataclass
class ResearchPlanStep:
    """One executable stage in a research workflow."""

    name: str
    description: str
    tool: str
    expected_output: str


@dataclass
class ResearchPlan:
    """A machine-readable plan produced before agent execution."""

    task_type: ScientificTaskType
    goal: str
    steps: list[ResearchPlanStep] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        return data

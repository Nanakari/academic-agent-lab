"""Human-reviewable protocol for planning a bounded pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentBlueprint:
    """Describe a pilot protocol without granting or performing execution."""

    direction_title: str
    source_idea_index: int | None
    objective: str
    hypothesis: str
    pilot_planning_ready: bool = False
    minimum_viable_experiment: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    ablations: list[str] = field(default_factory=list)
    planning_artifacts: list[str] = field(default_factory=list)
    experiment_artifacts: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_criteria: list[str] = field(default_factory=list)
    reproducibility_checklist: list[str] = field(default_factory=list)
    pre_execution_checklist: list[str] = field(default_factory=list)
    pre_execution_blockers: list[str] = field(default_factory=list)
    blueprint_note: str = ""

    @property
    def human_approval_required(self) -> bool:
        return True

    @property
    def execution_allowed(self) -> bool:
        return False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["human_approval_required"] = self.human_approval_required
        data["execution_allowed"] = self.execution_allowed
        return data

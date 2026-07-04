"""Planning-readiness assessment for the selected research direction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FeasibilityAssessment:
    """Conservative synthesis of pre-execution planning readiness."""

    direction_title: str
    source_idea_index: int | None
    planning_readiness_score: float
    recommendation: str
    evidence_readiness: str
    experiment_readiness: str
    reproducibility_readiness: str
    resource_requirement: str
    implementation_readiness: str
    dataset_clarity: str
    baseline_clarity: str
    metric_clarity: str
    main_risks: list[str] = field(default_factory=list)
    mitigation_strategies: list[str] = field(default_factory=list)
    minimum_viable_experiment: list[str] = field(default_factory=list)
    assessment_note: str = ""

    @property
    def overall_score(self) -> float:
        """Deprecated alias retained for Python callers during migration."""
        return self.planning_readiness_score

    def to_dict(self) -> dict:
        data = asdict(self)
        data["overall_score"] = self.planning_readiness_score
        return data

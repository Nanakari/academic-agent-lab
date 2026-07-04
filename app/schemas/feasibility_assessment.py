"""Planning-readiness assessment for the selected research direction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FeasibilityAssessment:
    """Conservative synthesis of pre-execution planning readiness."""

    direction_title: str
    source_idea_index: int | None
    overall_score: float
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

    def to_dict(self) -> dict:
        return asdict(self)

"""Structured planning-stage research direction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ResearchDirection:
    """Summarize one candidate idea as an evidence-aware planning direction."""

    title: str
    target_gap: str
    core_problem: str
    hypothesis: str
    method_sketch: str
    source_idea_title: str | None = None
    source_idea_index: int | None = None
    supporting_evidence: list[str] = field(default_factory=list)
    evidence_support_level: str = "insufficient"
    novelty_risk: str = "unknown"
    feasibility_risk: str = "unknown"
    recommended_priority: str = "exploratory"
    assessment_status: str = "heuristic_unverified"
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

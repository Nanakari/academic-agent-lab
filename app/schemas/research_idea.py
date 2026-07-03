"""Schema for a candidate research idea."""

from dataclasses import asdict, dataclass, field


@dataclass
class ResearchIdea:
    """A testable research idea grounded in retrieved evidence."""

    title: str
    hypothesis: str
    motivation: str
    method: str
    evidence_refs: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    rank_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

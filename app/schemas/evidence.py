"""Structured schemas for locally traceable paper evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


SUPPORT_LEVELS = ("strong", "moderate", "weak", "insufficient")


def support_level_for_score(score: float) -> str:
    """Map a normalized evidence score to the MVP support taxonomy."""
    if score >= 0.6:
        return "strong"
    if score >= 0.35:
        return "moderate"
    if score >= 0.15:
        return "weak"
    return "insufficient"


@dataclass(frozen=True)
class PaperDocument:
    """A local paper and its source-level metadata."""

    paper_id: str
    title: str
    source_path: str
    file_type: str
    text: str


@dataclass(frozen=True)
class PaperChunk:
    """A paper fragment retaining its section and page location."""

    chunk_id: str | int
    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class EvidenceChunk:
    """A ranked paper chunk with an explainable, local citation."""

    paper_id: str
    title: str
    source_path: str
    file_type: str
    chunk_id: str | int
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    matched_keywords: list[str] = field(default_factory=list)
    supporting_claim: str | None = None
    support_level: str = "insufficient"

    def to_dict(self) -> dict:
        return asdict(self)

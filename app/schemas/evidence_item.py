"""Source-neutral evidence records for optional external retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceType = Literal[
    "local_paper", "arxiv", "github_repo", "github_code", "github_issue"
]
ReliabilityLevel = Literal["high", "medium", "low", "unknown"]
EvidenceStatus = Literal["retrieved", "filtered", "verified", "rejected"]


@dataclass
class EvidenceItem:
    source_type: SourceType
    title: str
    summary: str
    url: str | None = None
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None
    updated_at: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    query: str | None = None
    retrieved_at: str | None = None
    relevance_score: float = 0.0
    reliability_level: ReliabilityLevel = "unknown"
    evidence_status: EvidenceStatus = "retrieved"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceItem":
        return cls(**value)


@dataclass
class ExternalEvidenceResult:
    enabled: bool
    query: str
    retrieved_at: str
    run_at: str | None = None
    cache_loaded_at: str | None = None
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    queries_used: dict[str, str] = field(default_factory=dict)
    retrieved_at_by_source: dict[str, str] = field(default_factory=dict)
    cache_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence_items": [item.to_dict() for item in self.evidence_items],
        }

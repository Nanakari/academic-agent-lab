"""Common output schema shared by all verifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class VerificationResult:
    """A scored verification decision with actionable feedback."""

    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    supported_claims: list[dict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    evidence_used: list[dict] = field(default_factory=list)
    support_level: str | None = None
    domain_consistency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

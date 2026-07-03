"""Common output schema shared by all verifiers."""

from dataclasses import asdict, dataclass, field


@dataclass
class VerificationResult:
    """A scored verification decision with actionable feedback."""

    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

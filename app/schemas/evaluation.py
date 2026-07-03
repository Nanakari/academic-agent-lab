"""Schemas for fixture-based AI Scientific Agent evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


DEFAULT_EXPERIMENT_FIELDS = ["datasets", "baselines", "metrics", "ablation", "risks"]


@dataclass
class EvalCase:
    """Expected behavior for one deterministic scientific-agent scenario."""

    case_id: str
    topic: str
    papers_dir: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_sections: list[str] | None = None
    expected_min_evidence_count: int = 0
    expected_required_experiment_fields: list[str] = field(
        default_factory=lambda: list(DEFAULT_EXPERIMENT_FIELDS)
    )
    should_pass_evidence_verifier: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        return cls(
            case_id=data["case_id"],
            topic=data["topic"],
            papers_dir=data["papers_dir"],
            expected_keywords=list(data.get("expected_keywords", [])),
            expected_sections=data.get("expected_sections"),
            expected_min_evidence_count=int(
                data.get("expected_min_evidence_count", 0)
            ),
            expected_required_experiment_fields=list(
                data.get(
                    "expected_required_experiment_fields",
                    DEFAULT_EXPERIMENT_FIELDS,
                )
            ),
            should_pass_evidence_verifier=bool(
                data.get("should_pass_evidence_verifier", False)
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalMetrics:
    """Small, interpretable metric set for one evaluation case."""

    evidence_count: int
    keyword_hit_rate: float
    section_hit_rate: float
    verifier_pass_match: bool
    experiment_completeness: float
    citation_completeness: float
    overall_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalCaseResult:
    """Evaluation outcome and failure reasons for one case."""

    case_id: str
    topic: str
    passed: bool
    metrics: EvalMetrics
    issues: list[str] = field(default_factory=list)
    agent_output_paths: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

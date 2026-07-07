"""Schema for a reproducible experiment proposal."""

from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentPlan:
    """Minimum information required to execute and evaluate an idea."""

    idea_title: str
    method: str
    datasets: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    attack_scenarios: list[str] = field(default_factory=list)
    tool_schemas: list[str] = field(default_factory=list)
    ablation: list[str] = field(default_factory=list)
    failure_taxonomy: list[str] = field(default_factory=list)
    expected_results: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    implementation_notes: list[str] = field(default_factory=list)
    reproducibility_notes: list[str] = field(default_factory=list)
    output_format: str = "JSON metrics and Markdown analysis"

    def to_dict(self) -> dict:
        return asdict(self)

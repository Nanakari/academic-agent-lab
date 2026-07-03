"""Reproducibility checks for proposed experiments."""

from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.verification_result import VerificationResult


class ReproducibilityVerifier:
    """Check data, method, comparisons, evaluation, ablation, and output format."""

    def verify(self, plan: ExperimentPlan) -> VerificationResult:
        checks = {
            "dataset": bool(plan.datasets),
            "model or method": bool(plan.method),
            "comparison baseline": bool(plan.baselines),
            "evaluation metric": bool(plan.metrics),
            "ablation": bool(plan.ablation),
            "expected output format": bool(plan.output_format),
            "execution details": bool(plan.implementation_notes),
        }
        missing = [name for name, present in checks.items() if not present]
        issues = [f"Reproducibility information missing: {name}." for name in missing]
        suggestions = [f"Document the {name} before running experiments." for name in missing]
        score = sum(checks.values()) / len(checks)
        return VerificationResult(not missing, round(score, 3), issues, suggestions)

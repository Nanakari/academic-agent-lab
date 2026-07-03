"""Completeness checks for experiment plans."""

from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.verification_result import VerificationResult


class ExperimentVerifier:
    REQUIRED_FIELDS = ("datasets", "baselines", "metrics", "ablation", "risks")

    def verify(self, plan: ExperimentPlan) -> VerificationResult:
        missing = [field for field in self.REQUIRED_FIELDS if not getattr(plan, field, None)]
        issues = [f"Missing experiment field: {field}." for field in missing]
        suggestions = [f"Specify at least one concrete {field} entry." for field in missing]
        score = (len(self.REQUIRED_FIELDS) - len(missing)) / len(self.REQUIRED_FIELDS)
        return VerificationResult(not missing, round(score, 3), issues, suggestions)

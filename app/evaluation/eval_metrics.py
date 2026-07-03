"""Interpretable metrics for scientific-agent fixture evaluation."""

from __future__ import annotations

from app.schemas.evaluation import EvalCase, EvalMetrics


class EvalMetricsCalculator:
    """Calculate retrieval, verification, experiment, and citation metrics."""

    WEIGHTS = {
        "keyword_hit_rate": 0.25,
        "verifier_pass_match": 0.25,
        "experiment_completeness": 0.25,
        "citation_completeness": 0.25,
    }

    def calculate(self, case: EvalCase, result: dict) -> EvalMetrics:
        evidence = result.get("evidence_context", [])
        evidence_count = len(evidence)
        keyword_hit_rate = self._keyword_hit_rate(case.expected_keywords, evidence)
        section_hit_rate = self._section_hit_rate(case.expected_sections, evidence)

        evidence_verification = result.get("verification", {}).get("evidence", {})
        verifier_pass_match = (
            bool(evidence_verification.get("passed"))
            == case.should_pass_evidence_verifier
        )
        experiment_completeness = self._experiment_completeness(
            case.expected_required_experiment_fields,
            result.get("experiment_plan", {}),
        )
        citation_completeness = self._citation_completeness(result)
        overall_score = (
            self.WEIGHTS["keyword_hit_rate"] * keyword_hit_rate
            + self.WEIGHTS["verifier_pass_match"] * float(verifier_pass_match)
            + self.WEIGHTS["experiment_completeness"] * experiment_completeness
            + self.WEIGHTS["citation_completeness"] * citation_completeness
        )
        return EvalMetrics(
            evidence_count=evidence_count,
            keyword_hit_rate=round(keyword_hit_rate, 3),
            section_hit_rate=round(section_hit_rate, 3),
            verifier_pass_match=verifier_pass_match,
            experiment_completeness=round(experiment_completeness, 3),
            citation_completeness=round(citation_completeness, 3),
            overall_score=round(overall_score, 3),
        )

    @staticmethod
    def collect_issues(case: EvalCase, metrics: EvalMetrics) -> list[str]:
        issues = []
        if metrics.evidence_count < case.expected_min_evidence_count:
            issues.append(
                f"no evidence found or evidence_count below "
                f"{case.expected_min_evidence_count}"
            )
        if metrics.keyword_hit_rate < 1.0:
            issues.append("insufficient keyword match")
        if case.expected_sections and metrics.section_hit_rate < 1.0:
            issues.append("expected paper sections were not retrieved")
        if not metrics.verifier_pass_match:
            issues.append("evidence verifier result did not match expectation")
        if metrics.experiment_completeness < 1.0:
            issues.append("experiment plan is missing required fields")
        if metrics.citation_completeness < 1.0:
            issues.append("citation fields are incomplete")
        return issues

    @staticmethod
    def _keyword_hit_rate(expected_keywords: list[str], evidence: list[dict]) -> float:
        if not expected_keywords:
            return 1.0
        searchable_text = " ".join(
            [
                " ".join(item.get("matched_keywords", []))
                + " "
                + str(item.get("text", ""))
                for item in evidence
            ]
        ).casefold()
        hits = sum(keyword.casefold() in searchable_text for keyword in expected_keywords)
        return hits / len(expected_keywords)

    @staticmethod
    def _section_hit_rate(
        expected_sections: list[str] | None,
        evidence: list[dict],
    ) -> float:
        if not expected_sections:
            return 1.0
        actual_sections = {
            str(item.get("section") or "").casefold() for item in evidence
        }
        hits = sum(section.casefold() in actual_sections for section in expected_sections)
        return hits / len(expected_sections)

    @staticmethod
    def _experiment_completeness(
        required_fields: list[str],
        experiment_plan: dict,
    ) -> float:
        if not required_fields:
            return 1.0
        present = sum(bool(experiment_plan.get(field)) for field in required_fields)
        return present / len(required_fields)

    @staticmethod
    def _citation_completeness(result: dict) -> float:
        evidence_verification = result.get("verification", {}).get("evidence", {})
        has_evidence = bool(result.get("evidence_context"))
        evidence_used = result.get("evidence_used")
        unsupported_claims = result.get("unsupported_claims")
        supported_claims = evidence_verification.get("supported_claims")
        verifier_evidence_used = evidence_verification.get("evidence_used")
        checks = [
            isinstance(evidence_used, list)
            and (bool(evidence_used) if has_evidence else True),
            isinstance(unsupported_claims, list)
            and (True if has_evidence else bool(unsupported_claims)),
            isinstance(supported_claims, list)
            and isinstance(verifier_evidence_used, list)
            and (
                bool(supported_claims) and bool(verifier_evidence_used)
                if has_evidence
                else bool(evidence_verification.get("unsupported_claims"))
            ),
        ]
        return sum(checks) / len(checks)

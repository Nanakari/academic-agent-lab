"""Rule-based planning-readiness synthesis for one selected direction."""

from __future__ import annotations

from app.schemas.feasibility_assessment import FeasibilityAssessment


class FeasibilityService:
    """Assess readiness without modifying plans or claiming empirical feasibility."""

    REQUIRED_EXPERIMENT_FIELDS = (
        "datasets",
        "baselines",
        "metrics",
        "ablation",
        "risks",
        "implementation_notes",
    )

    def assess(
        self,
        *,
        selected_direction,
        selected_idea,
        experiment_plan,
        evidence_assessment: dict,
        verification: dict,
    ) -> FeasibilityAssessment:
        evidence_readiness = self._evidence_readiness(
            selected_direction,
            verification,
        )
        experiment_readiness = self._experiment_readiness(
            experiment_plan,
            verification,
        )
        reproducibility_readiness = self._reproducibility_readiness(
            verification
        )
        resource_requirement = self._resource_requirement(
            selected_idea,
            experiment_plan,
        )
        implementation_readiness = self._implementation_readiness(
            experiment_readiness,
            reproducibility_readiness,
        )
        dataset_clarity = self._clarity(
            self._value(experiment_plan, "datasets")
        )
        baseline_clarity = self._clarity(
            self._value(experiment_plan, "baselines")
        )
        metric_clarity = self._clarity(
            self._value(experiment_plan, "metrics")
        )
        score = self._readiness_score(
            evidence_readiness,
            experiment_readiness,
            reproducibility_readiness,
        )
        recommendation = self._recommendation(
            evidence_readiness=evidence_readiness,
            experiment_readiness=experiment_readiness,
            reproducibility_readiness=reproducibility_readiness,
            direction_priority=(
                self._value(selected_direction, "recommended_priority")
                or "exploratory"
            ),
            score=score,
        )
        risks = self._main_risks(
            selected_direction=selected_direction,
            experiment_plan=experiment_plan,
            evidence_assessment=evidence_assessment,
            evidence_readiness=evidence_readiness,
            experiment_readiness=experiment_readiness,
            reproducibility_readiness=reproducibility_readiness,
        )
        mitigations = self._mitigations(
            selected_direction=selected_direction,
            evidence_readiness=evidence_readiness,
            experiment_readiness=experiment_readiness,
            reproducibility_readiness=reproducibility_readiness,
        )
        return FeasibilityAssessment(
            direction_title=(
                self._value(selected_direction, "title")
                or "Selected research direction"
            ),
            source_idea_index=self._value(
                selected_direction,
                "source_idea_index",
            ),
            overall_score=score,
            recommendation=recommendation,
            evidence_readiness=evidence_readiness,
            experiment_readiness=experiment_readiness,
            reproducibility_readiness=reproducibility_readiness,
            resource_requirement=resource_requirement,
            implementation_readiness=implementation_readiness,
            dataset_clarity=dataset_clarity,
            baseline_clarity=baseline_clarity,
            metric_clarity=metric_clarity,
            main_risks=risks,
            mitigation_strategies=mitigations,
            minimum_viable_experiment=self._minimum_viable_experiment(
                experiment_plan
            ),
            assessment_note=(
                "This planning-readiness score summarizes local evidence, "
                "verifier outputs, and checklist completeness for the selected "
                "experiment plan. It is not a probability of feasibility and "
                "does not prove scientific validity or empirical performance."
            ),
        )

    def _evidence_readiness(
        self,
        selected_direction,
        verification: dict,
    ) -> str:
        evidence = verification.get("evidence", {})
        if not evidence.get("passed", False):
            return "insufficient"
        level = self._value(selected_direction, "evidence_support_level")
        return {
            "strong": "ready",
            "moderate": "partial",
            "weak": "weak",
            "insufficient": "insufficient",
        }.get(level, "insufficient")

    def _experiment_readiness(
        self,
        experiment_plan,
        verification: dict,
    ) -> str:
        present = sum(
            bool(self._value(experiment_plan, field))
            for field in self.REQUIRED_EXPERIMENT_FIELDS
        )
        complete = present == len(self.REQUIRED_EXPERIMENT_FIELDS)
        if verification.get("experiment", {}).get("passed", False) and complete:
            return "ready"
        ratio = present / len(self.REQUIRED_EXPERIMENT_FIELDS)
        return "partial" if ratio >= 0.5 else "weak"

    @staticmethod
    def _reproducibility_readiness(verification: dict) -> str:
        return (
            "ready"
            if verification.get("reproducibility", {}).get("passed", False)
            else "weak"
        )

    def _resource_requirement(
        self,
        selected_idea,
        experiment_plan,
    ) -> str:
        text = " ".join(
            [
                str(self._value(selected_idea, "method") or ""),
                *self._strings(self._value(experiment_plan, "risks")),
                *self._strings(
                    self._value(experiment_plan, "implementation_notes")
                ),
            ]
        ).casefold()
        high_markers = (
            "multi-gpu",
            "multi gpu",
            "large-scale",
            "large scale",
            "high compute",
            "expensive training",
        )
        medium_markers = (
            "gpu",
            "compute",
            "latency",
            "cost",
            "large model",
        )
        low_markers = ("lightweight", "cpu", "small model", "local fixture")
        if any(marker in text for marker in high_markers):
            return "high"
        if any(marker in text for marker in medium_markers):
            return "medium"
        if any(marker in text for marker in low_markers):
            return "low"
        return "unknown"

    @staticmethod
    def _implementation_readiness(
        experiment_readiness: str,
        reproducibility_readiness: str,
    ) -> str:
        if (
            experiment_readiness == "ready"
            and reproducibility_readiness == "ready"
        ):
            return "ready"
        if experiment_readiness == "weak":
            return "weak"
        return "partial"

    @staticmethod
    def _clarity(values) -> str:
        items = FeasibilityService._strings(values)
        if not items:
            return "missing"
        placeholders = ("tbd", "unknown", "to be determined", "unspecified")
        if any(
            placeholder in item.casefold()
            for item in items
            for placeholder in placeholders
        ):
            return "partial"
        return "specified"

    @staticmethod
    def _readiness_score(
        evidence_readiness: str,
        experiment_readiness: str,
        reproducibility_readiness: str,
    ) -> float:
        values = {
            "ready": 1.0,
            "partial": 0.6,
            "weak": 0.25,
            "insufficient": 0.0,
        }
        score = (
            0.4 * values[evidence_readiness]
            + 0.4 * values[experiment_readiness]
            + 0.2 * values[reproducibility_readiness]
        )
        return round(score, 3)

    @staticmethod
    def _recommendation(
        *,
        evidence_readiness: str,
        experiment_readiness: str,
        reproducibility_readiness: str,
        direction_priority: str,
        score: float,
    ) -> str:
        if evidence_readiness == "insufficient":
            return "needs_more_evidence"
        if direction_priority == "exploratory":
            return "exploratory_only"
        if (
            experiment_readiness != "ready"
            or reproducibility_readiness != "ready"
        ):
            return "proceed_with_caution"
        if score >= 0.75:
            return "ready_for_pilot_planning"
        return "proceed_with_caution"

    def _main_risks(
        self,
        *,
        selected_direction,
        experiment_plan,
        evidence_assessment: dict,
        evidence_readiness: str,
        experiment_readiness: str,
        reproducibility_readiness: str,
    ) -> list[str]:
        risks = self._strings(self._value(experiment_plan, "risks"))
        if evidence_readiness == "insufficient":
            risks.append(
                "Local evidence does not sufficiently support the selected direction."
            )
            gaps = evidence_assessment.get("gaps", [])
            if gaps:
                risks.append(f"Evidence gap: {gaps[0]}")
        if experiment_readiness != "ready":
            risks.append(
                "The experiment plan requires clearer or more complete execution details."
            )
        if reproducibility_readiness != "ready":
            risks.append(
                "Seeds, versions, environment, logging, or other reproducibility "
                "details may be incomplete."
            )
        if self._value(selected_direction, "novelty_risk") == "high":
            risks.append(
                "The selected direction may overlap with saved or candidate ideas."
            )
        fallbacks = (
            "Planning assumptions require human review before execution.",
            "Empirical performance remains unknown until a pilot is run.",
        )
        for fallback in fallbacks:
            if len(self._deduplicate(risks)) >= 2:
                break
            risks.append(fallback)
        return self._deduplicate(risks)

    def _mitigations(
        self,
        *,
        selected_direction,
        evidence_readiness: str,
        experiment_readiness: str,
        reproducibility_readiness: str,
    ) -> list[str]:
        strategies = []
        if evidence_readiness in {"insufficient", "weak"}:
            strategies.append(
                "Add relevant local papers before treating the direction as grounded."
            )
        if experiment_readiness != "ready":
            strategies.append(
                "Specify datasets, baselines, metrics, ablations, risks, and "
                "implementation notes."
            )
        if reproducibility_readiness != "ready":
            strategies.append(
                "Record dependency versions, random seeds, environment, and logs."
            )
        if self._value(selected_direction, "novelty_risk") == "high":
            strategies.append(
                "Compare against saved and candidate ideas to reduce duplication."
            )
        strategies.append(
            "Require human review of the minimum viable experiment before execution."
        )
        if len(self._deduplicate(strategies)) < 2:
            strategies.append(
                "Start with one bounded pilot and review its artifacts before scaling."
            )
        return self._deduplicate(strategies)

    def _minimum_viable_experiment(self, experiment_plan) -> list[str]:
        dataset = self._first(
            self._value(experiment_plan, "datasets"),
            "one small benchmark or local fixture task",
        )
        baseline = self._first(
            self._value(experiment_plan, "baselines"),
            "one simple baseline",
        )
        metric = self._first(
            self._value(experiment_plan, "metrics"),
            "one primary metric",
        )
        ablation = self._first(
            self._value(experiment_plan, "ablation"),
            "one ablation of the main proposed component",
        )
        return [
            f"Use {dataset}.",
            f"Compare against {baseline}.",
            f"Evaluate with {metric}.",
            f"Run {ablation}.",
            "Record seeds, dependency versions, environment, and output artifacts.",
        ]

    @staticmethod
    def _first(values, fallback: str) -> str:
        items = FeasibilityService._strings(values)
        return items[0] if items else fallback

    @staticmethod
    def _strings(values) -> list[str]:
        if not values:
            return []
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values if value]

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _value(item, name: str):
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

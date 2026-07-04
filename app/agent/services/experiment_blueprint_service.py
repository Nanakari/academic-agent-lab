"""Build a human-reviewable pilot planning protocol."""

from __future__ import annotations

from app.schemas.experiment_blueprint import ExperimentBlueprint


class ExperimentBlueprintService:
    """Organize existing planning outputs without executing or mutating them."""

    PLANNING_ARTIFACTS = (
        "result.json",
        "report.md",
        "experiment_blueprint section in the planning report",
        "planning_readiness_assessment",
        "agent_trace",
    )
    EXPERIMENT_ARTIFACTS = (
        "experiment_config.yaml or equivalent configuration notes",
        "raw_predictions_or_outputs",
        "evaluation_metrics",
        "failure_cases",
        "variance_or_repeated_run_summary",
        "environment_and_dependency_notes",
        "random_seed_record",
        "execution_logs",
    )

    def build(
        self,
        *,
        selected_direction,
        selected_idea,
        experiment_plan,
        feasibility_assessment,
        verification: dict,
        evidence_assessment: dict,
    ) -> ExperimentBlueprint:
        datasets = self._strings(self._value(experiment_plan, "datasets"))
        baselines = self._strings(self._value(experiment_plan, "baselines"))
        metrics = self._strings(self._value(experiment_plan, "metrics"))
        ablations = self._strings(self._value(experiment_plan, "ablation"))
        recommendation = (
            self._value(feasibility_assessment, "recommendation") or ""
        )
        blockers = self._pre_execution_blockers(
            datasets=datasets,
            baselines=baselines,
            metrics=metrics,
            ablations=ablations,
            recommendation=recommendation,
            verification=verification,
        )
        pilot_planning_ready = (
            recommendation == "ready_for_pilot_planning"
            and all(
                verification.get(name, {}).get("passed", False)
                for name in ("evidence", "experiment", "reproducibility")
            )
            and bool(datasets and baselines and metrics and ablations)
        )
        return ExperimentBlueprint(
            direction_title=(
                self._value(selected_direction, "title")
                or "Selected research direction"
            ),
            source_idea_index=self._value(
                selected_direction,
                "source_idea_index",
            ),
            objective=self._objective(selected_direction),
            hypothesis=(
                self._value(selected_direction, "hypothesis")
                or self._value(selected_idea, "hypothesis")
                or (
                    "A planning-stage hypothesis should be validated before "
                    "being treated as a result."
                )
            ),
            pilot_planning_ready=pilot_planning_ready,
            minimum_viable_experiment=self._minimum_viable_experiment(
                feasibility_assessment,
                experiment_plan,
            ),
            datasets=datasets,
            baselines=baselines,
            metrics=metrics,
            ablations=ablations,
            planning_artifacts=list(self.PLANNING_ARTIFACTS),
            experiment_artifacts=list(self.EXPERIMENT_ARTIFACTS),
            success_criteria=self._success_criteria(
                datasets,
                baselines,
                metrics,
            ),
            failure_criteria=self._failure_criteria(),
            reproducibility_checklist=self._reproducibility_checklist(
                verification
            ),
            pre_execution_checklist=self._pre_execution_checklist(
                recommendation,
                verification,
                evidence_assessment,
            ),
            pre_execution_blockers=blockers,
            blueprint_note=(
                "This blueprint is a human-reviewable pilot planning protocol. "
                "It does not execute experiments, download datasets, train "
                "models, or prove empirical performance. Human approval is "
                "required before any experiment execution, and execution "
                "remains disabled because this agent has no approval input or "
                "execution subsystem."
            ),
        )

    def _objective(self, selected_direction) -> str:
        core_problem = (
            self._value(selected_direction, "core_problem")
            or "the stated research problem"
        )
        method = (
            self._value(selected_direction, "method_sketch")
            or "the selected method sketch"
        )
        return (
            "Specify how a human-approved bounded pilot would evaluate the "
            "selected method sketch against the stated core problem. "
            f"Core problem: {core_problem} Method sketch: {method}"
        )

    def _minimum_viable_experiment(
        self,
        feasibility_assessment,
        experiment_plan,
    ) -> list[str]:
        existing = self._strings(
            self._value(
                feasibility_assessment,
                "minimum_viable_experiment",
            )
        )
        if existing:
            return existing
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
    def _success_criteria(
        datasets: list[str],
        baselines: list[str],
        metrics: list[str],
    ) -> list[str]:
        criteria = [
            "The selected metric can be computed consistently.",
            "The baseline comparison is fully reported.",
            (
                "A target delta or decision threshold is defined by a human "
                "before execution."
            ),
            (
                "Failure cases and variance are reported alongside aggregate "
                "metrics."
            ),
            "All outputs can be linked to saved experiment artifacts.",
        ]
        if metrics and datasets:
            criteria.append(
                f"The pilot reports {metrics[0]} on {datasets[0]}."
            )
        if baselines:
            criteria.append(
                f"The proposed method is compared against {baselines[0]}."
            )
        return criteria

    @staticmethod
    def _failure_criteria() -> list[str]:
        return [
            "The selected metric cannot be computed or interpreted.",
            "The method cannot be compared against the chosen baseline.",
            "Required experiment artifacts are missing.",
            "The executed protocol deviates from the approved pilot plan.",
            "Outputs cannot be traced back to saved artifacts.",
        ]

    @staticmethod
    def _reproducibility_checklist(verification: dict) -> list[str]:
        checklist = [
            "Record random seeds.",
            "Record dependency versions.",
            "Record hardware or runtime environment.",
            "Save raw outputs and evaluation logs.",
            "Save configuration notes.",
            "Document dataset split or input selection.",
            "Document any manual intervention during execution.",
        ]
        if not verification.get("reproducibility", {}).get("passed", False):
            checklist.append(
                "Resolve reproducibility verifier issues before any experiment execution."
            )
        return checklist

    @staticmethod
    def _pre_execution_checklist(
        recommendation: str,
        verification: dict,
        evidence_assessment: dict,
    ) -> list[str]:
        checklist = [
            "Confirm local evidence is sufficient.",
            "Confirm dataset, baseline, metric, and ablation are defined.",
            "Confirm required planning and experiment artifacts are planned.",
            "Confirm human approval before any experiment execution.",
            "Confirm no automatic training or data download is triggered by this agent.",
            "Confirm the pilot protocol is reviewed before execution.",
        ]
        if verification.get("evidence", {}).get("passed", False):
            checklist.append(
                "Confirm evidence-related claims remain grounded in cited local evidence."
            )
        if recommendation == "needs_more_evidence":
            checklist.append("Add more local evidence before pilot planning.")
        if recommendation == "exploratory_only":
            checklist.append(
                "Treat this as exploratory planning until evidence and verifier "
                "issues improve."
            )
        if evidence_assessment.get("gaps"):
            checklist.append("Review every recorded evidence gap.")
        return checklist

    @staticmethod
    def _pre_execution_blockers(
        *,
        datasets: list[str],
        baselines: list[str],
        metrics: list[str],
        ablations: list[str],
        recommendation: str,
        verification: dict,
    ) -> list[str]:
        blockers = ["Human approval is required before experiment execution."]
        if recommendation == "needs_more_evidence":
            blockers.append(
                "More local evidence is required before pilot planning."
            )
        if not verification.get("evidence", {}).get("passed", False):
            blockers.append("Evidence verifier failed.")
        if not verification.get("experiment", {}).get("passed", False):
            blockers.append(
                "Experiment verifier reported incomplete experimental design."
            )
        if not verification.get("reproducibility", {}).get("passed", False):
            blockers.append(
                "Reproducibility verifier reported incomplete reproducibility details."
            )
        if recommendation == "exploratory_only":
            blockers.append("Selected direction is exploratory only.")
        for values, label in (
            (datasets, "Dataset"),
            (baselines, "Baseline"),
            (metrics, "Metric"),
            (ablations, "Ablation"),
        ):
            if not values:
                blockers.append(f"{label} is not specified.")
        return list(dict.fromkeys(blockers))

    @staticmethod
    def _first(values, fallback: str) -> str:
        items = ExperimentBlueprintService._strings(values)
        return items[0] if items else fallback

    @staticmethod
    def _strings(values) -> list[str]:
        if not values:
            return []
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values if value]

    @staticmethod
    def _value(item, name: str):
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

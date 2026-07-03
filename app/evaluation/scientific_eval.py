"""Evaluation runner for the AI Scientific Agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.agent.ai_scientific_agent import AIScientificAgent
from app.evaluation.eval_metrics import EvalMetricsCalculator
from app.memory.scientific_memory import ScientificMemory
from app.schemas.evaluation import EvalCase, EvalCaseResult


class ScientificEvaluator:
    """Run isolated fixture cases and aggregate their outcomes."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )
        self.metrics_calculator = EvalMetricsCalculator()

    def evaluate(self, cases: list[EvalCase], output_dir: str | Path) -> dict:
        output_path = Path(output_dir)
        case_results = [
            self.evaluate_case(case, output_path / "cases" / case.case_id)
            for case in cases
        ]
        passed_cases = sum(result.passed for result in case_results)
        average_score = (
            sum(result.metrics.overall_score for result in case_results)
            / len(case_results)
            if case_results
            else 0.0
        )
        return {
            "summary": {
                "total_cases": len(case_results),
                "passed_cases": passed_cases,
                "average_overall_score": round(average_score, 3),
            },
            "cases": [result.to_dict() for result in case_results],
        }

    def evaluate_case(self, case: EvalCase, output_dir: str | Path) -> EvalCaseResult:
        papers_dir = self._resolve_project_path(case.papers_dir)
        with tempfile.TemporaryDirectory(prefix=f"scientific-eval-{case.case_id}-") as temp:
            memory = ScientificMemory(Path(temp) / "memory")
            agent = AIScientificAgent(
                project_root=self.project_root,
                papers_dir=papers_dir,
                output_dir=output_dir,
                top_k=max(5, case.expected_min_evidence_count),
                memory=memory,
            )
            result = agent.run(case.topic)

        metrics = self.metrics_calculator.calculate(case, result)
        issues = self.metrics_calculator.collect_issues(case, metrics)
        return EvalCaseResult(
            case_id=case.case_id,
            topic=case.topic,
            passed=not issues,
            metrics=metrics,
            issues=issues,
            agent_output_paths=result.get("output_paths", {}),
        )

    def _resolve_project_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

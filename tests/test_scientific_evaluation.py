"""Tests for the lightweight scientific-agent evaluation framework."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.evaluation.eval_cases import load_eval_cases
from app.evaluation.eval_metrics import EvalMetricsCalculator
from app.scientific_eval_demo import run_evaluation
from app.tools.paper_corpus import PaperCorpusIndexer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "eval_cases.json"


def minimal_result(
    evidence: list[dict] | None = None,
    verifier_passed: bool = True,
    experiment_plan: dict | None = None,
) -> dict:
    """Build the smallest result contract consumed by evaluation metrics."""
    return {
        "evidence_context": evidence or [],
        "evidence_used": [],
        "unsupported_claims": [],
        "verification": {
            "evidence": {
                "passed": verifier_passed,
                "supported_claims": [],
                "evidence_used": [],
            }
        },
        "experiment_plan": experiment_plan or {},
    }


class ScientificEvaluationTests(unittest.TestCase):
    def test_eval_cases_load(self) -> None:
        cases = load_eval_cases(CASES_PATH)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].case_id, "lvlm_hallucination_fixture")
        self.assertFalse(cases[1].should_pass_evidence_verifier)

    def test_fixture_keyword_hit_rate_is_positive(self) -> None:
        case = load_eval_cases(CASES_PATH)[0]
        corpus = PaperCorpusIndexer(PROJECT_ROOT / case.papers_dir)
        evidence = [item.to_dict() for item in corpus.search(case.topic, top_k=5)]
        result = minimal_result(evidence=evidence, verifier_passed=True)

        metrics = EvalMetricsCalculator().calculate(case, result)

        self.assertGreater(metrics.keyword_hit_rate, 0.0)

    def test_empty_corpus_expected_failure_matches_verifier(self) -> None:
        case = load_eval_cases(CASES_PATH)[1]
        result = minimal_result(evidence=[], verifier_passed=False)

        metrics = EvalMetricsCalculator().calculate(case, result)

        self.assertTrue(metrics.verifier_pass_match)

    def test_experiment_completeness_detects_complete_plan(self) -> None:
        case = load_eval_cases(CASES_PATH)[0]
        complete_plan = {
            field: ["configured"] for field in case.expected_required_experiment_fields
        }
        result = minimal_result(
            verifier_passed=True,
            experiment_plan=complete_plan,
        )

        metrics = EvalMetricsCalculator().calculate(case, result)

        self.assertEqual(metrics.experiment_completeness, 1.0)

    def test_demo_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "evaluation_report.md"

            result = run_evaluation(
                CASES_PATH,
                report_path,
                project_root=PROJECT_ROOT,
            )

            json_path = Path(result["output_paths"]["json"])
            self.assertTrue(json_path.exists())
            self.assertTrue(report_path.exists())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# AI Scientific Agent Evaluation Report", report)
            self.assertIn("## Failure Analysis", report)


if __name__ == "__main__":
    unittest.main()

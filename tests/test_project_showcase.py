"""Tests for real-paper validation and GitHub showcase documentation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.evaluation.validation_report import RealPaperValidationReportWriter
from app.real_paper_validation_demo import (
    EMPTY_CORPUS_MESSAGE,
    run_real_paper_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealPaperValidationTests(unittest.TestCase):
    def test_empty_papers_directory_exits_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "papers").mkdir(parents=True)

            result = run_real_paper_validation(
                topic="LVLM hallucination mitigation",
                papers_dir="data/papers",
                output_dir=root / "outputs",
                project_root=root,
            )

            self.assertEqual(result["status"], "no_papers")
            self.assertEqual(result["message"], EMPTY_CORPUS_MESSAGE)

    def test_validation_summary_contains_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "validation_summary.md"
            agent_result = {
                "selected_idea": {"title": "Evidence-aware LVLM verifier"},
                "experiment_plan": {
                    "datasets": ["POPE"],
                    "baselines": ["Base LVLM"],
                    "metrics": ["Hallucination rate"],
                },
                "verification_passed": True,
                "evidence_context": [
                    {
                        "title": "Grounded LVLM",
                        "section": "Method",
                        "chunk_id": "C2",
                        "score": 0.9,
                        "support_level": "strong",
                        "matched_keywords": ["hallucination", "lvlm"],
                        "text": "LVLM visual evidence reduces hallucination.",
                    }
                ],
                "unsupported_claims": [],
                "evidence_gaps": [],
                "verification": {
                    "evidence": {
                        "passed": False,
                        "domain_consistency": {
                            "passed": True,
                            "matched_topic_concepts": [
                                "lvlm hallucination"
                            ],
                            "issues": [],
                        },
                    }
                },
            }
            metrics = {
                "keyword_hit_rate": 1.0,
                "verifier_pass_match": True,
                "experiment_completeness": 1.0,
                "citation_completeness": 1.0,
                "overall_score": 1.0,
            }

            RealPaperValidationReportWriter().write(
                topic="LVLM hallucination mitigation",
                papers_dir="data/papers",
                top_k=8,
                scanned_papers=2,
                agent_result=agent_result,
                metrics=metrics,
                output_path=output_path,
            )

            report = output_path.read_text(encoding="utf-8")
            for heading in (
                "# Real Paper Validation Summary",
                "## Input",
                "## Agent Output Summary",
                "## Evidence Summary",
                "### Evidence support distribution",
                "### Topic keyword coverage",
                "### Strongest evidence detail",
                "### Domain consistency",
                "## Evaluation Summary",
                "## Notes",
            ):
                self.assertIn(heading, report)
            self.assertIn(
                "Evidence coverage may be incomplete; consider increasing top_k",
                report,
            )

    def test_readme_contains_showcase_modes(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("AI Scientific Agent Mode", readme)
        self.assertIn("Evaluation Mode", readme)
        self.assertIn("Real Paper Validation", readme)
        self.assertIn("Structured Evidence Citation", readme)
        self.assertTrue(
            (PROJECT_ROOT / "examples" / "resume_description.md").exists()
        )


if __name__ == "__main__":
    unittest.main()

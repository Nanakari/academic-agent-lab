"""JSON and Markdown reporting for scientific-agent evaluation."""

from __future__ import annotations

import json
from pathlib import Path


class EvaluationReportWriter:
    """Serialize aggregate evaluation results without extra dependencies."""

    def write_json(self, evaluation: dict, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_markdown(self, evaluation: dict, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = evaluation["summary"]
        lines = [
            "# AI Scientific Agent Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Total cases: {summary['total_cases']}",
            f"- Passed cases: {summary['passed_cases']}",
            f"- Average overall score: {summary['average_overall_score']:.3f}",
            "",
            "## Case Results",
            "",
        ]
        for case in evaluation["cases"]:
            metrics = case["metrics"]
            lines.extend([
                f"### {case['case_id']}",
                "",
                f"- Topic: {case['topic']}",
                f"- Passed: {case['passed']}",
                f"- Evidence count: {metrics['evidence_count']}",
                f"- Keyword hit rate: {metrics['keyword_hit_rate']:.3f}",
                f"- Section hit rate: {metrics['section_hit_rate']:.3f}",
                f"- Verifier pass match: {metrics['verifier_pass_match']}",
                f"- Experiment completeness: {metrics['experiment_completeness']:.3f}",
                f"- Citation completeness: {metrics['citation_completeness']:.3f}",
                f"- Overall score: {metrics['overall_score']:.3f}",
                "- Main issues: "
                + ("; ".join(case["issues"]) if case["issues"] else "None"),
                "",
            ])
        lines.extend(["## Failure Analysis", ""])
        failed_cases = [case for case in evaluation["cases"] if not case["passed"]]
        if failed_cases:
            for case in failed_cases:
                for issue in case["issues"]:
                    lines.append(f"- {case['case_id']}: {issue}")
        else:
            lines.append("- No evaluation case failures.")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

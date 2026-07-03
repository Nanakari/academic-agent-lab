"""Command-line entry point for fixture-based scientific-agent evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.eval_cases import load_eval_cases
from app.evaluation.eval_report import EvaluationReportWriter
from app.evaluation.scientific_eval import ScientificEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AIScientificAgent on local fixture cases."
    )
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "tests" / "fixtures" / "eval_cases.json"),
        help="JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.md"),
        help="Markdown evaluation report path.",
    )
    return parser


def run_evaluation(
    cases_path: str | Path,
    report_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict:
    """Run all cases and write both required evaluation artifacts."""
    report_path = Path(report_path)
    cases = load_eval_cases(cases_path)
    evaluator = ScientificEvaluator(project_root=project_root)
    evaluation = evaluator.evaluate(cases, output_dir=report_path.parent)
    writer = EvaluationReportWriter()
    json_path = writer.write_json(
        evaluation,
        report_path.parent / "evaluation_result.json",
    )
    markdown_path = writer.write_markdown(evaluation, report_path)
    return {
        "evaluation": evaluation,
        "output_paths": {
            "json": str(json_path.resolve()),
            "markdown": str(markdown_path.resolve()),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_evaluation(args.cases, args.output)
    print(json.dumps(
        {
            **result["evaluation"]["summary"],
            "output_paths": result["output_paths"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

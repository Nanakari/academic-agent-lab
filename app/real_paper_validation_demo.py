"""Validate AIScientificAgent against a user-provided local paper collection."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.ai_scientific_agent import AIScientificAgent
from app.cli.scientific import build_default_llm
from app.evaluation.eval_metrics import EvalMetricsCalculator
from app.evaluation.eval_report import EvaluationReportWriter
from app.evaluation.validation_report import RealPaperValidationReportWriter
from app.memory.scientific_memory import ScientificMemory
from app.schemas.evaluation import EvalCase
from app.tools.paper_corpus import PaperCorpusIndexer, keyword_tokens

EMPTY_CORPUS_MESSAGE = (
    "Please add TXT/MD/PDF papers into data/papers/ "
    "before running real paper validation."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AIScientificAgent and evaluate it on real local papers."
    )
    parser.add_argument("--topic", required=True, help="AI research topic to validate.")
    parser.add_argument(
        "--papers-dir",
        default="data/papers",
        help="Directory containing local TXT/MD/PDF papers.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Maximum evidence chunks to retrieve.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "real_paper_validation"),
        help="Directory for validation artifacts.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable LLM tool-decision calls for offline regression checks.",
    )
    return parser


def run_real_paper_validation(
    *,
    topic: str,
    papers_dir: str | Path,
    top_k: int = 8,
    output_dir: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    offline: bool = False,
) -> dict:
    """Run agent plus metrics, or return a friendly empty-corpus result."""
    root = Path(project_root).resolve()
    requested_papers = Path(papers_dir)
    resolved_papers = (
        requested_papers.resolve()
        if requested_papers.is_absolute()
        else (root / requested_papers).resolve()
    )
    output_path = (
        Path(output_dir)
        if output_dir
        else root / "outputs" / "real_paper_validation"
    )
    corpus = PaperCorpusIndexer(resolved_papers)
    documents = corpus.scan_papers()
    if not documents:
        return {
            "status": "no_papers",
            "message": EMPTY_CORPUS_MESSAGE,
            "scanned_papers": 0,
        }

    llm = None if offline else build_default_llm()
    with tempfile.TemporaryDirectory(prefix="real-paper-validation-") as temp:
        agent = AIScientificAgent(
            project_root=root,
            papers_dir=resolved_papers,
            output_dir=output_path,
            top_k=top_k,
            memory=ScientificMemory(Path(temp) / "memory"),
            llm=llm,
            llm_tool_decision_enabled=not offline,
            domain_mode="strict",
        )
        agent_result = agent.run(topic)

    eval_case = EvalCase(
        case_id="real_paper_validation",
        topic=topic,
        papers_dir=str(resolved_papers),
        expected_keywords=sorted(keyword_tokens(topic)),
        expected_sections=None,
        expected_min_evidence_count=1,
        should_pass_evidence_verifier=True,
    )
    calculator = EvalMetricsCalculator()
    metrics = calculator.calculate(eval_case, agent_result)
    issues = calculator.collect_issues(eval_case, metrics)
    evaluation = {
        "case": eval_case.to_dict(),
        "metrics": metrics.to_dict(),
        "issues": issues,
        "passed": not issues,
    }
    evaluation_path = EvaluationReportWriter().write_json(
        evaluation,
        output_path / "evaluation_result.json",
    )
    summary_path = RealPaperValidationReportWriter().write(
        topic=topic,
        papers_dir=resolved_papers,
        top_k=top_k,
        scanned_papers=len(documents),
        agent_result=agent_result,
        metrics=metrics.to_dict(),
        output_path=output_path / "validation_summary.md",
    )
    return {
        "status": "completed",
        "scanned_papers": len(documents),
        "evaluation": evaluation,
        "output_paths": {
            **agent_result["output_paths"],
            "evaluation_json": str(evaluation_path.resolve()),
            "validation_summary": str(summary_path.resolve()),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_real_paper_validation(
        topic=args.topic,
        papers_dir=args.papers_dir,
        top_k=max(1, args.top_k),
        output_dir=args.output_dir,
        offline=args.offline,
    )
    if result["status"] == "no_papers":
        print(result["message"])
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

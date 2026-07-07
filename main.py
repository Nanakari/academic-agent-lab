"""Top-level CLI for academic-agent-lab.

The default entry point is the LLM-driven, verifier-bounded AIScientificAgent.
The original LLM tool-calling AcademicAgent remains available as an explicit
legacy demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.cli.scientific import (
    build_scientific_parser,
    print_result_summary,
    run_scientific_from_args,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TOPIC = "LVLM hallucination mitigation"
DEFAULT_FIXTURE_PAPERS = "tests/fixtures/papers"


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the legacy AcademicAgent RAG demo. Requires Gemini API key."
    )
    parser.add_argument(
        "--paper-path",
        default="data/demo_paper.pdf",
        help="Local TXT/MD/PDF paper path for the legacy RAG tool.",
    )
    parser.add_argument(
        "--question",
        default="What is the core method of this paper?",
        help="Question to ask about the paper.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum legacy tool-calling steps.",
    )
    return parser


def run_legacy_academic(args: argparse.Namespace) -> None:
    from app.agent.academic_agent import AcademicAgent
    from app.config import load_config
    from app.llm import LLM

    config = load_config()
    llm = LLM(config)
    agent = AcademicAgent(llm=llm, max_steps=args.max_steps)
    user_request = (
        "Legacy AcademicAgent demo. Answer this paper question using tools in "
        "this order: rag_search, paper_qa, terminate. "
        f"Question: {args.question} "
        f"Paper path: {args.paper_path}"
    )

    result = agent.run(user_request)
    print("Legacy AcademicAgent result:")
    print(result)
    print("Final answer:")
    print(agent.final_answer)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "legacy-academic":
        parser = build_legacy_parser()
        run_legacy_academic(parser.parse_args(argv[1:]))
        return

    if argv and argv[0] == "scientific":
        argv = argv[1:]
    parser = build_scientific_parser(
        description="Run the LLM-driven AI Scientific Agent demo.",
        default_topic=DEFAULT_TOPIC,
        topic_required=False,
        default_output_dir=str(PROJECT_ROOT / "outputs" / "ai_scientific_agent"),
        default_papers_dir=DEFAULT_FIXTURE_PAPERS,
    )
    result = run_scientific_from_args(parser.parse_args(argv), project_root=PROJECT_ROOT)
    print_result_summary(result, include_mode=True)


if __name__ == "__main__":
    main()

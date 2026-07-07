"""Shared CLI helpers for the LLM-driven scientific agent."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


SUPPORTED_EXTERNAL_SOURCES = {"arxiv", "github"}
SUPPORTED_LLM_STAGES = {
    "tool_decision",
    "literature_analysis",
    "idea_generation",
    "experiment_design",
    "reflection",
}


class LLMConfigurationError(RuntimeError):
    """Raised when the default LLM client cannot be created."""


def parse_external_sources(value: str) -> list[str]:
    """Parse and validate a comma-separated external source list."""
    sources = [
        source.strip().casefold()
        for source in value.split(",")
        if source.strip()
    ]
    invalid = sorted(set(sources) - SUPPORTED_EXTERNAL_SOURCES)
    if invalid:
        raise ValueError(
            "Unsupported --external-sources value(s): " + ", ".join(invalid)
        )
    return sources


def parse_llm_stages(value: str) -> list[str]:
    """Parse and validate a comma-separated LLM stage list."""
    cleaned = str(value or "all").strip().casefold()
    if cleaned == "all":
        return sorted(SUPPORTED_LLM_STAGES)
    if cleaned in {"none", "off"}:
        return []
    stages = [
        stage.strip()
        for stage in cleaned.split(",")
        if stage.strip()
    ]
    invalid = sorted(set(stages) - SUPPORTED_LLM_STAGES)
    if invalid:
        raise ValueError("Unsupported --llm-stages value(s): " + ", ".join(invalid))
    return stages


def build_scientific_parser(
    *,
    description: str,
    default_topic: str | None = None,
    topic_required: bool = False,
    default_output_dir: str,
    default_papers_dir: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--topic",
        required=topic_required,
        default=default_topic,
        help=(
            "AI research direction or task."
            if default_topic is None
            else f"AI research direction or task (default: {default_topic!r})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory for result.json and report.md.",
    )
    parser.add_argument(
        "--papers-dir",
        default=default_papers_dir,
        help=f"Local .txt, .md, and .pdf paper corpus (default: {default_papers_dir}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of evidence chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Disable LLM tool-decision calls and use the deterministic local "
            "workflow. Intended for CI and offline regression checks."
        ),
    )
    parser.add_argument(
        "--llm-stages",
        default="all",
        help=(
            "Comma-separated LLM stages to enable, or all/none. Supported: "
            "tool_decision,literature_analysis,idea_generation,experiment_design,reflection."
        ),
    )
    external = parser.add_mutually_exclusive_group()
    external.add_argument(
        "--use-external-search",
        action="store_true",
        help="Force controlled arXiv/GitHub metadata retrieval to be available.",
    )
    external.add_argument(
        "--no-external-search",
        action="store_true",
        help="Disallow external arXiv/GitHub metadata retrieval.",
    )
    parser.add_argument(
        "--external-sources",
        default="arxiv,github",
        help="Comma-separated subset of arxiv,github (default: both).",
    )
    parser.add_argument(
        "--external-max-results",
        type=int,
        default=5,
        help="Maximum external results per requested source (default: 5).",
    )
    parser.add_argument(
        "--external-force-refresh",
        action="store_true",
        help="Ignore existing external evidence cache entries.",
    )
    return parser


def run_scientific_from_args(
    args: argparse.Namespace,
    *,
    project_root: str | Path,
) -> dict:
    """Run AIScientificAgent from parsed CLI arguments."""
    from app.agent.ai_scientific_agent import AIScientificAgent
    from app.memory.scientific_memory import ScientificMemory

    try:
        external_sources = parse_external_sources(args.external_sources)
        llm_stages = parse_llm_stages(args.llm_stages)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.no_external_search:
        external_sources = []
    topic = str(args.topic or "").strip()
    if not topic:
        raise SystemExit("--topic must not be empty.")

    try:
        llm = None if args.offline else build_default_llm()
    except LLMConfigurationError as error:
        raise SystemExit(str(error)) from error
    if args.offline:
        llm_stages = []
    root = Path(project_root)
    with tempfile.TemporaryDirectory(prefix="ai-scientific-cli-") as temp:
        agent = AIScientificAgent(
            project_root=root,
            output_dir=Path(args.output_dir),
            papers_dir=Path(args.papers_dir),
            top_k=args.top_k,
            memory=ScientificMemory(Path(temp) / "memory"),
            llm=llm,
            llm_tool_decision_enabled=not args.offline,
            llm_stages=llm_stages,
            external_search_enabled=(
                args.use_external_search and not args.no_external_search
            ),
            external_search_sources=external_sources,
            external_max_results_per_source=args.external_max_results,
            external_force_refresh=args.external_force_refresh,
        )
        return agent.run(topic)


def build_default_llm():
    """Create the default LLM client for user-facing agent runs."""
    try:
        from app.config import load_config
        from app.llm import LLM

        return LLM(load_config())
    except Exception as exc:
        raise LLMConfigurationError(
            "LLM tool-decision mode is the default for this scientific agent. "
            "Install runtime dependencies with `python -m pip install -r requirements.txt`. "
            "Set GEMINI_API_KEY or write api_key in config.toml. "
            "Use --offline only for CI/offline regression runs. "
            f"Original error: {exc}"
        ) from exc


def summarize_result(result: dict, *, include_mode: bool = False) -> dict:
    summary = {
        "task_type": result["task_type"],
        "evidence_status": result["evidence_status"],
        "evidence_count": len(result["evidence_context"]),
        "external_search_status": result["external_search_status"],
        "external_evidence_count": len(result["external_evidence"]),
        "verification_passed": result["verification_passed"],
        "scientific_readiness": result.get("scientific_readiness"),
        "final_recommendation": result.get("final_recommendation"),
        "llm_call_count": result.get("llm_call_count", 0),
        "llm_call_stages": result.get("llm_call_stages", []),
        "llm_fallback_stages": result.get("llm_fallback_stages", []),
        "tool_decision": result.get("tool_decision", {}),
        "output_paths": result["output_paths"],
    }
    if include_mode:
        return {"mode": "ai_scientific_agent", **summary}
    return summary


def print_result_summary(result: dict, *, include_mode: bool = False) -> None:
    print(
        json.dumps(
            summarize_result(result, include_mode=include_mode),
            ensure_ascii=False,
            indent=2,
        )
    )

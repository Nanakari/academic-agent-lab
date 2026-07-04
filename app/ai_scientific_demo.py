"""Command-line demo for the AI Scientific Agent MVP."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Support the documented invocation: python app/ai_scientific_demo.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.ai_scientific_agent import AIScientificAgent
from app.memory.scientific_memory import ScientificMemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local AI Scientific Agent MVP.")
    parser.add_argument("--topic", required=True, help="AI research direction or task.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "ai_scientific_agent"),
        help="Directory for result.json and report.md.",
    )
    parser.add_argument(
        "--papers-dir",
        default="data/papers",
        help="Local .txt, .md, and .pdf paper corpus (default: data/papers).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of evidence chunks to retrieve (default: 5).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Demo runs are intentionally isolated so repeated Quick Start commands are
    # reproducible and do not pollute the project's long-lived scientific memory.
    with tempfile.TemporaryDirectory(prefix="ai-scientific-demo-") as temp:
        agent = AIScientificAgent(
            project_root=PROJECT_ROOT,
            output_dir=Path(args.output_dir),
            papers_dir=Path(args.papers_dir),
            top_k=args.top_k,
            memory=ScientificMemory(Path(temp) / "memory"),
        )
        result = agent.run(args.topic)
    print(json.dumps(
        {
            "task_type": result["task_type"],
            "evidence_status": result["evidence_status"],
            "evidence_count": len(result["evidence_context"]),
            "verification_passed": result["verification_passed"],
            "output_paths": result["output_paths"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

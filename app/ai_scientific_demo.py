"""Command-line demo for the AI Scientific Agent MVP."""

import argparse
import json
import sys
from pathlib import Path

# Support the documented invocation: python app/ai_scientific_demo.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.ai_scientific_agent import AIScientificAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local AI Scientific Agent MVP.")
    parser.add_argument("--topic", required=True, help="AI research direction or task.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "ai_scientific_agent"),
        help="Directory for result.json and report.md.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = AIScientificAgent(
        project_root=PROJECT_ROOT,
        output_dir=Path(args.output_dir),
    )
    result = agent.run(args.topic)
    print(json.dumps(
        {
            "task_type": result["task_type"],
            "verification_passed": result["verification_passed"],
            "output_paths": result["output_paths"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

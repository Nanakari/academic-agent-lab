"""Command-line demo for the AI Scientific Agent MVP."""

import sys
from pathlib import Path

# Support the documented invocation: python app/ai_scientific_demo.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cli.scientific import (
    build_scientific_parser,
    print_result_summary,
    run_scientific_from_args,
)


def main() -> None:
    parser = build_scientific_parser(
        description="Run the local AI Scientific Agent MVP.",
        topic_required=True,
        default_output_dir=str(PROJECT_ROOT / "outputs" / "ai_scientific_agent"),
        default_papers_dir="data/papers",
    )
    result = run_scientific_from_args(parser.parse_args(), project_root=PROJECT_ROOT)
    print_result_summary(result)


if __name__ == "__main__":
    main()

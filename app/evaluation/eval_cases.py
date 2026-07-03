"""Loading and validation helpers for evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.evaluation import EvalCase


def load_eval_cases(cases_path: str | Path) -> list[EvalCase]:
    """Load cases from either a JSON list or a {"cases": [...]} object."""
    path = Path(cases_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("cases", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Evaluation cases JSON must contain a list of cases.")

    cases = [EvalCase.from_dict(record) for record in records]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case_id values must be unique.")
    return cases

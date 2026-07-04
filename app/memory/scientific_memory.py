"""Small, dependency-free JSONL memory for research artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ScientificMemory:
    """Persist papers, ideas, experiments, and verification logs locally."""

    FILES = {
        "paper_note": "paper_notes.jsonl",
        "idea": "ideas.jsonl",
        "experiment": "experiments.jsonl",
        "verification_log": "verification_logs.jsonl",
    }

    def __init__(self, memory_dir: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.memory_dir = Path(memory_dir) if memory_dir else root / "data" / "research_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.warnings: list[str] = []

    def save_paper_note(self, note: Any) -> bool:
        """Save a note unless the same topic/source/chunk is already present."""
        return self._append_unique(
            "paper_note",
            note,
            keys=("source", "chunk_id", "topic"),
        )

    def save_idea(self, idea: Any) -> bool:
        """Save an idea unless its topic, title, and method exactly match."""
        return self._append_unique(
            "idea",
            idea,
            keys=("topic", "title", "method"),
        )

    def save_experiment(self, experiment: Any) -> None:
        self._append("experiment", experiment)

    def save_verification_log(self, log: Any) -> None:
        self._append("verification_log", log)

    def load_recent_ideas(self, limit: int = 10) -> list[dict]:
        return self._read_records("idea")[-max(0, limit):]

    def search_memory(self, keyword: str) -> list[dict]:
        query = keyword.casefold().strip()
        if not query:
            return []
        matches = []
        for record_type in self.FILES:
            for record in self._read_records(record_type):
                if query in json.dumps(record, ensure_ascii=False).casefold():
                    matches.append({"memory_type": record_type, **record})
        return matches

    def _append_unique(
        self,
        record_type: str,
        value: Any,
        keys: tuple[str, ...],
    ) -> bool:
        payload = self._to_dict(value)
        identity = tuple(payload.get(key) for key in keys)
        if any(
            tuple(record.get(key) for key in keys) == identity
            for record in self._read_records(record_type)
        ):
            return False
        self._append(record_type, payload)
        return True

    def _append(self, record_type: str, value: Any) -> None:
        payload = self._to_dict(value)
        payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        path = self.memory_dir / self.FILES[record_type]
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_records(self, record_type: str) -> list[dict]:
        path = self.memory_dir / self.FILES[record_type]
        if not path.exists():
            return []
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                warning = (
                    f"Skipped malformed JSONL record in {path} at line "
                    f"{line_number}: {error.msg}"
                )
                if warning not in self.warnings:
                    self.warnings.append(warning)
                    LOGGER.warning(warning)
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _to_dict(value: Any) -> dict:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return dict(value)
        raise TypeError("ScientificMemory records must be dictionaries or expose to_dict().")

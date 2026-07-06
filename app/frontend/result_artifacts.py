"""Read persisted frontend artifacts without recreating canonical outputs."""

from __future__ import annotations

from pathlib import Path


def read_result_json_bytes(path: str | Path) -> bytes | None:
    """Return the exact persisted result.json bytes, or None when unavailable."""
    result_path = Path(path)
    try:
        return result_path.read_bytes() if result_path.is_file() else None
    except OSError:
        return None

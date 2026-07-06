"""JSON cache for auditable external retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvidenceCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, source: str, query: str) -> Path:
        digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / source / f"{digest}.json"

    def load(self, source: str, query: str) -> tuple[dict[str, Any] | None, str | None]:
        path = self.path_for(source, query)
        if not path.exists():
            return None, None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("query") != query or value.get("source") != source:
                return None, f"Ignored mismatched {source} cache entry: {path}"
            return value, None
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"Could not read {source} cache; refreshing: {exc}"

    def save(
        self,
        source: str,
        query: str,
        retrieved_at: str,
        items: list[dict],
        warnings: list[str],
    ) -> str | None:
        path = self.path_for(source, query)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "query": query,
                "source": source,
                "retrieved_at": retrieved_at,
                "items": items,
                "warnings": warnings,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return None
        except OSError as exc:
            return f"Could not write {source} cache: {exc}"

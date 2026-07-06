"""JSON cache for auditable external retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvidenceCache:
    SCHEMA_VERSION = 2

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, source: str, query: str, max_results: int) -> Path:
        raw_key = json.dumps(
            {
                "source": source,
                "query": query.strip(),
                "max_results": int(max_results),
                "schema_version": self.SCHEMA_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / source / f"{digest}.json"

    def load(
        self,
        source: str,
        query: str,
        max_results: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        path = self.path_for(source, query, max_results)
        if not path.exists():
            return None, None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("query") != query
                or value.get("source") != source
                or value.get("max_results") != int(max_results)
                or value.get("schema_version") != self.SCHEMA_VERSION
                or value.get("status") != "ok"
            ):
                return None, f"Ignored mismatched {source} cache entry: {path}"
            return value, None
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"Could not read {source} cache; refreshing: {exc}"

    def save(
        self,
        source: str,
        query: str,
        max_results: int,
        retrieved_at: str,
        items: list[dict],
        warnings: list[str],
    ) -> str | None:
        path = self.path_for(source, query, max_results)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "schema_version": self.SCHEMA_VERSION,
                "status": "ok",
                "query": query,
                "source": source,
                "max_results": int(max_results),
                "retrieved_at": retrieved_at,
                "items": items,
                "warnings": warnings,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return None
        except OSError as exc:
            return f"Could not write {source} cache: {exc}"

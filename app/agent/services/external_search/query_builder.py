"""Deterministic short-query construction for external providers."""

from __future__ import annotations

import re


class ExternalSearchQueryBuilder:
    def build_queries(self, topic: str, max_queries: int = 3) -> list[str]:
        cleaned = re.sub(r"[^\w\s+\-]", " ", topic, flags=re.UNICODE)
        cleaned = " ".join(cleaned.split())
        words = cleaned.split()[:12]
        if not words:
            return []
        base = " ".join(words)
        candidates = [base]
        if len(words) > 6:
            candidates.append(" ".join(words[:6]))
        return candidates[:max(1, min(3, int(max_queries)))]

    def for_source(self, query: str, source: str) -> str:
        if source == "github":
            return f"{query} implementation"
        return query

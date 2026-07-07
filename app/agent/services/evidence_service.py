"""Local-paper retrieval with scientific-memory fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from app.memory.scientific_memory import ScientificMemory
from app.schemas.evidence import support_level_for_score
from app.tools.paper_corpus import (
    PaperCorpusIndexer,
    infer_supporting_claim,
    keyword_tokens,
)


class EvidenceService:
    """Retrieve evidence and normalize it for the agent result contract."""

    def __init__(
        self,
        project_root: str | Path,
        paper_corpus: PaperCorpusIndexer,
        memory: ScientificMemory,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.paper_corpus = paper_corpus
        self.memory = memory
        self.last_deduplicated_count = 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        use_local_papers: bool = True,
        use_scientific_memory: bool = True,
    ) -> list[dict]:
        """Search papers first, then fill available slots from paper-note memory."""
        limit = max(1, int(top_k))
        selected = []
        if use_local_papers:
            for evidence in self.paper_corpus.search(query, top_k=limit * 3):
                item = evidence.to_dict()
                source_path = Path(item["source_path"])
                try:
                    display_source = source_path.relative_to(self.project_root).as_posix()
                except ValueError:
                    display_source = str(source_path)
                item.update(
                    {
                        "source": display_source,
                        "excerpt": item["text"],
                        "kind": "local_paper",
                    }
                )
                selected.append(item)

        selected, local_duplicates = self._deduplicate(selected)
        selected = selected[:limit]
        remaining = limit - len(selected)
        if use_scientific_memory and remaining > 0:
            selected.extend(self._search_memory(query, remaining))
        selected, combined_duplicates = self._deduplicate(selected)
        selected = selected[:limit]
        self.last_deduplicated_count = (
            local_duplicates + combined_duplicates
        )

        for index, item in enumerate(selected, start=1):
            item["evidence_id"] = f"E{index}"
        return selected

    @classmethod
    def _deduplicate(cls, evidence_items: list[dict]) -> tuple[list[dict], int]:
        """Deduplicate result records without touching source files."""
        positions: dict[tuple, int] = {}
        deduplicated: list[dict] = []
        duplicate_count = 0
        for item in evidence_items:
            key = cls._deduplication_key(item)
            if key not in positions:
                positions[key] = len(deduplicated)
                deduplicated.append(item)
                continue
            duplicate_count += 1
            position = positions[key]
            if float(item.get("score", 0.0)) > float(
                deduplicated[position].get("score", 0.0)
            ):
                deduplicated[position] = item
        deduplicated.sort(
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return deduplicated, duplicate_count

    @staticmethod
    def _deduplication_key(item: dict) -> tuple:
        normalized_title = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+",
            "",
            str(item.get("title") or "").casefold(),
        )
        section = str(item.get("section") or "").strip().casefold()
        claim = str(
            item.get("supporting_claim")
            or item.get("excerpt")
            or item.get("text")
            or ""
        )
        normalized_claim = re.sub(r"\s+", " ", claim).strip().casefold()
        claim_hash = hashlib.sha256(
            normalized_claim.encode("utf-8")
        ).hexdigest()[:16]
        return (
            normalized_title,
            item.get("page"),
            section,
            claim_hash,
        )

    def _search_memory(self, query: str, limit: int) -> list[dict]:
        records = []
        seen = set()
        for keyword in [query, *sorted(keyword_tokens(query))]:
            for record in self.memory.search_memory(keyword):
                # Only paper-derived notes are valid fallback scientific evidence.
                if (
                    record.get("memory_type") != "paper_note"
                    or str(record.get("source", "")).startswith("memory:")
                ):
                    continue
                record_key = (
                    record.get("saved_at"),
                    record.get("title"),
                    record.get("source"),
                    record.get("chunk_id"),
                )
                if record_key not in seen:
                    seen.add(record_key)
                    records.append(record)

        candidates = []
        for record in records:
            excerpt = str(
                record.get("summary")
                or record.get("motivation")
                or record.get("hypothesis")
                or record
            )
            title = str(record.get("title") or record.get("topic") or "Scientific memory")
            score = self.paper_corpus.score_text(query, excerpt, title)
            if score <= 0:
                continue
            candidates.append(
                {
                    "paper_id": "memory-paper-note",
                    "title": title,
                    "source_path": "memory:paper_note",
                    "file_type": "memory",
                    "page": record.get("page"),
                    "section": record.get("section"),
                    "chunk_id": str(record.get("evidence_id") or "memory-record"),
                    "text": excerpt[:800],
                    "source": "memory:paper_note",
                    "excerpt": excerpt[:800],
                    "score": round(score, 3),
                    "matched_keywords": self.paper_corpus.matched_keywords(
                        query,
                        excerpt,
                        title,
                    ),
                    "supporting_claim": infer_supporting_claim(query, excerpt),
                    "support_level": support_level_for_score(score),
                    "kind": "scientific_memory",
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

"""Local-paper retrieval with scientific-memory fallback."""

from __future__ import annotations

from pathlib import Path

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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Search papers first, then fill available slots from paper-note memory."""
        limit = max(1, int(top_k))
        selected = []
        for evidence in self.paper_corpus.search(query, top_k=limit):
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

        remaining = limit - len(selected)
        if remaining > 0:
            selected.extend(self._search_memory(query, remaining))

        for index, item in enumerate(selected, start=1):
            item["evidence_id"] = f"E{index}"
        return selected

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

"""Lightweight local paper corpus indexing and lexical evidence retrieval."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.rag.document_loader import load_document_text

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]+|[\u4e00-\u9fff]{2,}")
STOPWORDS = {
    "about", "after", "also", "and", "for", "from", "into", "of", "the",
    "this", "that", "using", "with", "研究", "方法", "论文",
}


def keyword_tokens(text: str) -> set[str]:
    """Tokenize English terms and Chinese character bigrams for local search."""
    tokens: set[str] = set()
    for token in TOKEN_PATTERN.findall(text.casefold()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(token[index:index + 2] for index in range(len(token) - 1))
        elif len(token) >= 3 and token not in STOPWORDS:
            tokens.add(token)
    return tokens


@dataclass(frozen=True)
class PaperDocument:
    paper_id: str
    title: str
    source_path: str
    text: str


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class EvidenceChunk:
    paper_id: str
    title: str
    source_path: str
    chunk_id: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class PaperCorpusIndexer:
    """Scan local papers, split them, and rank chunks without a vector database."""

    def __init__(
        self,
        papers_dir: str | Path,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> None:
        self.papers_dir = Path(papers_dir).resolve()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: list[PaperDocument] = []
        self.index: list[tuple[PaperDocument, PaperChunk]] = []
        self.warnings: list[str] = []

    def scan_papers(self, papers_dir: Path | None = None) -> list[PaperDocument]:
        """Read supported papers while isolating failures to individual files."""
        directory = (papers_dir or self.papers_dir).resolve()
        self.warnings = []
        if not directory.exists():
            return []

        documents = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                continue
            try:
                text = load_document_text(path).strip()
                if not text:
                    raise ValueError("document contains no extractable text")
            except (OSError, RuntimeError, ValueError) as error:
                warning = f"Skipped unreadable paper {path}: {error}"
                self.warnings.append(warning)
                LOGGER.warning(warning)
                continue

            relative_path = path.relative_to(directory).as_posix()
            documents.append(
                PaperDocument(
                    paper_id=hashlib.sha1(
                        relative_path.casefold().encode("utf-8")
                    ).hexdigest()[:12],
                    title=self._extract_title(text, path.stem),
                    source_path=str(path),
                    text=text,
                )
            )
        return documents

    def split_document(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> list[PaperChunk]:
        """Split text into deterministic overlapping character chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

        normalized = re.sub(r"[ \t]+", " ", text).strip()
        if not normalized:
            return []

        chunks = []
        step = chunk_size - overlap
        for index, start in enumerate(range(0, len(normalized), step), start=1):
            chunk_text = normalized[start:start + chunk_size].strip()
            if chunk_text:
                chunks.append(PaperChunk(chunk_id=f"C{index}", text=chunk_text))
            if start + chunk_size >= len(normalized):
                break
        return chunks

    def build_or_refresh_index(self) -> int:
        """Rebuild the in-memory index from the current directory contents."""
        self.documents = self.scan_papers(self.papers_dir)
        self.index = [
            (document, chunk)
            for document in self.documents
            for chunk in self.split_document(
                document.text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
        ]
        return len(self.index)

    def search(self, query: str, top_k: int = 5) -> list[EvidenceChunk]:
        """Return the highest-scoring paper chunks for a query."""
        if top_k <= 0:
            return []
        self.build_or_refresh_index()
        ranked = []
        for document, chunk in self.index:
            score = self.score_text(query, chunk.text, document.title)
            if score > 0:
                ranked.append(
                    EvidenceChunk(
                        paper_id=document.paper_id,
                        title=document.title,
                        source_path=document.source_path,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=round(score, 3),
                    )
                )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def score_text(query: str, text: str, title: str = "") -> float:
        """Score lexical overlap on a stable 0-1 scale."""
        query_terms = keyword_tokens(query)
        if not query_terms:
            return 0.0
        text_terms = keyword_tokens(text)
        title_terms = keyword_tokens(title)
        body_coverage = len(query_terms & text_terms) / len(query_terms)
        title_coverage = len(query_terms & title_terms) / len(query_terms)
        phrase_bonus = 0.1 if query.casefold().strip() in text.casefold() else 0.0
        return min(1.0, 0.8 * body_coverage + 0.2 * title_coverage + phrase_bonus)

    @staticmethod
    def _extract_title(text: str, fallback: str) -> str:
        for line in text.splitlines()[:10]:
            candidate = re.sub(r"^\s*(?:#|title\s*:)\s*", "", line, flags=re.I).strip()
            if 4 <= len(candidate) <= 200:
                return candidate
        return fallback.replace("_", " ").replace("-", " ").strip()

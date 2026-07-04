"""Lightweight local paper parsing and explainable evidence retrieval."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from app.rag.document_loader import load_document_text
from app.schemas.evidence import (
    EvidenceChunk,
    PaperChunk,
    PaperDocument,
    support_level_for_score,
)

LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]+|[\u4e00-\u9fff]{2,}")
PDF_PAGE_PATTERN = re.compile(r"(?m)^\[Page\s+(\d+)\]\s*$")
NUMBERED_HEADING_PATTERN = re.compile(
    r"^\d+(?:\.\d+)*[.)]?\s+([A-Za-z][^.!?]{1,100})$"
)
KNOWN_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experiment",
    "evaluation",
    "results",
    "discussion",
    "limitations",
    "limitation",
    "conclusion",
    "conclusions",
    "future work",
}
STOPWORDS = {
    "about", "after", "also", "and", "before", "can", "for", "from", "into",
    "may", "not", "of", "our", "the", "this", "that", "through", "using",
    "with", "研究", "方法", "论文",
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


def infer_supporting_claim(query_or_idea: str, chunk_text: str) -> str:
    """Select the sentence containing the most query keywords."""
    query_terms = keyword_tokens(query_or_idea)
    normalized_text = re.sub(r"\s+", " ", chunk_text).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", normalized_text)
        if len(sentence.strip()) >= 15
    ]
    if not sentences:
        return normalized_text[:400]
    best_sentence = max(
        sentences,
        key=lambda sentence: len(query_terms & keyword_tokens(sentence)),
    )
    return best_sentence[:400]


class PaperCorpusIndexer:
    """Parse local papers and rank section/page-aware chunks."""

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
        self._index_signature: tuple | None = None

    def _corpus_signature(self) -> tuple:
        """Return cheap file metadata used to detect corpus changes."""
        files = []
        if self.papers_dir.exists():
            for path in sorted(self.papers_dir.rglob("*")):
                if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    # A concurrent deletion will be reflected by the next signature.
                    continue
                files.append(
                    (
                        path.relative_to(self.papers_dir).as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                        path.suffix.casefold(),
                    )
                )
        return (self.chunk_size, self.overlap, tuple(files))

    def index_is_stale(self) -> bool:
        """Report whether the in-memory index needs to be rebuilt."""
        return (
            not self.index
            or self._index_signature is None
            or self._index_signature != self._corpus_signature()
        )

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
                    file_type=path.suffix.casefold().lstrip("."),
                    text=text,
                )
            )
        return documents

    def split_document(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 120,
        page: int | None = None,
        section: str | None = None,
    ) -> list[PaperChunk]:
        """Split text while carrying page and section metadata into each chunk."""
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
                chunks.append(
                    PaperChunk(
                        chunk_id=f"C{index}",
                        text=chunk_text,
                        page=page,
                        section=section,
                    )
                )
            if start + chunk_size >= len(normalized):
                break
        return chunks

    def build_or_refresh_index(self, force: bool = False) -> int:
        """Build the index only when corpus metadata or chunk settings changed."""
        if not force and not self.index_is_stale():
            return len(self.index)

        self.documents = self.scan_papers(self.papers_dir)
        self.index = []
        for document in self.documents:
            chunk_number = 1
            current_section = None
            for page, page_text in self._page_segments(document):
                section_segments, current_section = self._section_segments(
                    page_text,
                    current_section,
                )
                for section, section_text in section_segments:
                    for chunk in self.split_document(
                        section_text,
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        page=page,
                        section=section,
                    ):
                        structured_chunk = PaperChunk(
                            chunk_id=f"C{chunk_number}",
                            text=chunk.text,
                            page=chunk.page,
                            section=chunk.section,
                        )
                        self.index.append((document, structured_chunk))
                        chunk_number += 1
        self._index_signature = self._corpus_signature()
        return len(self.index)

    def search(
        self,
        query: str,
        top_k: int = 5,
        force_refresh: bool = False,
    ) -> list[EvidenceChunk]:
        """Return explainable evidence chunks ranked by keyword coverage."""
        if top_k <= 0:
            return []
        self.build_or_refresh_index(force=force_refresh)
        query_terms = keyword_tokens(query)
        if not query_terms:
            return []

        ranked = []
        for document, chunk in self.index:
            # Body overlap is the primary signal. A matching title contributes
            # at most 0.1, preventing every chunk in a well-titled paper from
            # receiving strong support.
            matched_keywords = sorted(query_terms & keyword_tokens(chunk.text))
            title_matches = query_terms & keyword_tokens(document.title)
            text_score = len(matched_keywords) / len(query_terms)
            title_bonus = 0.1 * len(title_matches) / len(query_terms)
            score = min(1.0, text_score + title_bonus)
            if score <= 0:
                continue
            ranked.append(
                EvidenceChunk(
                    paper_id=document.paper_id,
                    title=document.title,
                    source_path=document.source_path,
                    file_type=document.file_type,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=round(score, 3),
                    matched_keywords=matched_keywords,
                    supporting_claim=infer_supporting_claim(query, chunk.text),
                    support_level=support_level_for_score(score),
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.paper_id, str(item.chunk_id)))
        return ranked[:top_k]

    @staticmethod
    def score_text(query: str, text: str, title: str = "") -> float:
        """Return body coverage plus a title bonus capped at 0.1."""
        query_terms = keyword_tokens(query)
        if not query_terms:
            return 0.0
        text_matches = query_terms & keyword_tokens(text)
        title_matches = query_terms & keyword_tokens(title)
        text_score = len(text_matches) / len(query_terms)
        title_bonus = 0.1 * len(title_matches) / len(query_terms)
        return min(1.0, text_score + title_bonus)

    @staticmethod
    def matched_keywords(query: str, text: str, title: str = "") -> list[str]:
        """Return body-matched terms; title-only matches are not evidence."""
        query_terms = keyword_tokens(query)
        return sorted(query_terms & keyword_tokens(text))

    @staticmethod
    def _page_segments(document: PaperDocument) -> list[tuple[int | None, str]]:
        if document.file_type != "pdf":
            return [(None, document.text)]
        matches = list(PDF_PAGE_PATTERN.finditer(document.text))
        if not matches:
            return [(None, document.text)]
        pages = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
            pages.append((int(match.group(1)), document.text[match.end():end].strip()))
        return pages

    @classmethod
    def _section_segments(
        cls,
        text: str,
        initial_section: str | None = None,
    ) -> tuple[list[tuple[str | None, str]], str | None]:
        segments: list[tuple[str | None, str]] = []
        current_section = initial_section
        buffer: list[str] = []
        for line in text.splitlines():
            if line.strip().casefold().startswith("title:"):
                continue
            heading = cls._detect_section_heading(line)
            if heading is not None:
                if any(part.strip() for part in buffer):
                    segments.append((current_section, "\n".join(buffer).strip()))
                current_section = heading
                buffer = []
            else:
                buffer.append(line)
        if any(part.strip() for part in buffer):
            segments.append((current_section, "\n".join(buffer).strip()))
        return segments, current_section

    @staticmethod
    def _detect_section_heading(line: str) -> str | None:
        stripped = line.strip()
        if not stripped or stripped.casefold().startswith("title:"):
            return None
        markdown_match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", stripped)
        if markdown_match:
            return markdown_match.group(1).strip()
        numbered_match = NUMBERED_HEADING_PATTERN.match(stripped)
        if numbered_match:
            return numbered_match.group(1).strip()
        plain = stripped.rstrip(":").strip()
        if plain.casefold() in KNOWN_SECTIONS:
            return plain
        return None

    @staticmethod
    def _extract_title(text: str, fallback: str) -> str:
        for line in text.splitlines()[:10]:
            stripped = line.strip()
            if stripped.casefold().startswith("title:"):
                return stripped.split(":", maxsplit=1)[1].strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return fallback.replace("_", " ").replace("-", " ").strip()

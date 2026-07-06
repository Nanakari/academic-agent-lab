"""Metadata-only arXiv API retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from app.schemas.evidence_item import EvidenceItem

from .base import open_url


class ArxivSearchService:
    API_URL = "https://export.arxiv.org/api/query"
    ATOM = "http://www.w3.org/2005/Atom"

    def __init__(
        self,
        timeout: float = 10.0,
        delay_seconds: float = 0.0,
        opener=open_url,
    ) -> None:
        self.timeout = timeout
        self.delay_seconds = max(0.0, delay_seconds)
        self.opener = opener
        self.last_warnings: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[EvidenceItem]:
        self.last_warnings = []
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, int(max_results)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        request = urllib.request.Request(
            f"{self.API_URL}?{params}",
            headers={"User-Agent": "academic-agent-lab/1.0"},
        )
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            with self.opener(request, self.timeout) as response:
                payload = response.read()
            return self._parse_feed(payload, query)
        except Exception as exc:
            self.last_warnings.append(f"arXiv retrieval failed: {exc}")
            return []

    @classmethod
    def _parse_feed(cls, payload: bytes, query: str) -> list[EvidenceItem]:
        root = ET.fromstring(payload)
        now = datetime.now(timezone.utc).isoformat()
        items = []
        ns = {"atom": cls.ATOM}
        for entry in root.findall("atom:entry", ns):
            source_url = cls._text(entry, "atom:id", ns)
            source_id = source_url.rsplit("/", 1)[-1] if source_url else None
            html_url = source_url
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("rel") == "alternate":
                    html_url = link.attrib.get("href", html_url)
                    break
            items.append(EvidenceItem(
                source_type="arxiv",
                title=cls._clean(cls._text(entry, "atom:title", ns)),
                summary=cls._clean(cls._text(entry, "atom:summary", ns)),
                url=html_url,
                authors=[
                    cls._text(author, "atom:name", ns)
                    for author in entry.findall("atom:author", ns)
                ],
                published_at=cls._text(entry, "atom:published", ns) or None,
                updated_at=cls._text(entry, "atom:updated", ns) or None,
                source_id=source_id,
                metadata={
                    "categories": [
                        category.attrib.get("term", "")
                        for category in entry.findall("atom:category", ns)
                    ],
                    "retrieval_scope": "metadata_and_abstract",
                    "evidence_role": "literature_discovery",
                },
                query=query,
                retrieved_at=now,
                reliability_level="unknown",
            ))
        return items

    @staticmethod
    def _text(element, path: str, namespaces: dict[str, str]) -> str:
        found = element.find(path, namespaces)
        return found.text.strip() if found is not None and found.text else ""

    @staticmethod
    def _clean(value: str) -> str:
        return " ".join(value.split())

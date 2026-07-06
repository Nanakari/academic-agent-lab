"""Repository-only GitHub API retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import urllib.parse
import urllib.request

from app.schemas.evidence_item import EvidenceItem

from .base import open_url


class GitHubSearchService:
    API_URL = "https://api.github.com/search/repositories"

    def __init__(self, timeout: float = 10.0, opener=open_url) -> None:
        self.timeout = timeout
        self.opener = opener
        self.last_warnings: list[str] = []
        self.last_attempt_succeeded = True

    def search_repositories(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[EvidenceItem]:
        self.last_warnings = []
        self.last_attempt_succeeded = True
        token = os.getenv("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "academic-agent-lab/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            self.last_warnings.append(
                "GITHUB_TOKEN is not set; unauthenticated GitHub rate limits apply."
            )
        params = urllib.parse.urlencode({
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": max(1, min(100, int(max_results))),
        })
        request = urllib.request.Request(f"{self.API_URL}?{params}", headers=headers)
        try:
            with self.opener(request, self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return self._normalize(payload.get("items", []), query)
        except Exception as exc:
            self.last_attempt_succeeded = False
            self.last_warnings.append(f"GitHub retrieval failed: {exc}")
            return []

    @staticmethod
    def _normalize(records: list[dict], query: str) -> list[EvidenceItem]:
        now = datetime.now(timezone.utc).isoformat()
        items = []
        for record in records:
            license_data = record.get("license") or {}
            full_name = str(record.get("full_name") or "Unnamed repository")
            items.append(EvidenceItem(
                source_type="github_repo",
                title=full_name,
                summary=str(record.get("description") or "No description provided."),
                url=record.get("html_url"),
                updated_at=record.get("updated_at"),
                source_id=full_name,
                metadata={
                    "stargazers_count": record.get("stargazers_count", 0),
                    "forks_count": record.get("forks_count", 0),
                    "language": record.get("language"),
                    "license": license_data.get("spdx_id"),
                    "open_issues_count": record.get("open_issues_count", 0),
                    "archived": bool(record.get("archived", False)),
                    "evidence_role": "implementation_availability",
                    "scientific_validation": False,
                },
                query=query,
                retrieved_at=now,
                reliability_level="medium",
            ))
        return items

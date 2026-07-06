"""Independent orchestration of arXiv and GitHub retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.schemas.evidence_item import EvidenceItem, ExternalEvidenceResult

from .arxiv_search_service import ArxivSearchService
from .evidence_cache import EvidenceCache
from .github_search_service import GitHubSearchService


class ExternalEvidenceService:
    def __init__(
        self,
        cache_dir: str | Path,
        arxiv_service: ArxivSearchService | None = None,
        github_service: GitHubSearchService | None = None,
        cache_enabled: bool = True,
        force_refresh: bool = False,
    ) -> None:
        self.arxiv = arxiv_service or ArxivSearchService()
        self.github = github_service or GitHubSearchService()
        self.cache = EvidenceCache(cache_dir)
        self.cache_enabled = cache_enabled
        self.force_refresh = force_refresh

    def retrieve(
        self,
        query: str,
        use_arxiv: bool = True,
        use_github: bool = True,
        max_results_per_source: int = 5,
        source_queries: dict[str, str] | None = None,
    ) -> ExternalEvidenceResult:
        run_at = datetime.now(timezone.utc).isoformat()
        result = ExternalEvidenceResult(
            enabled=use_arxiv or use_github,
            query=query,
            retrieved_at="",
            run_at=run_at,
        )
        for source, enabled, service, method_name in (
            ("arxiv", use_arxiv, self.arxiv, "search"),
            ("github", use_github, self.github, "search_repositories"),
        ):
            if not enabled:
                continue
            source_query = (source_queries or {}).get(source, query)
            result.queries_used[source] = source_query
            cached = None
            if self.cache_enabled and not self.force_refresh:
                cached, warning = self.cache.load(
                    source,
                    source_query,
                    max_results_per_source,
                )
                if warning:
                    result.warnings.append(warning)
            if cached is not None:
                cached_items = [
                    EvidenceItem.from_dict(item)
                    for item in cached.get("items", [])
                ]
                result.evidence_items.extend(cached_items)
                result.warnings.extend(cached.get("warnings", []))
                if cached_items:
                    result.sources_used.append(source)
                cached_retrieved_at = str(cached.get("retrieved_at") or "")
                result.retrieved_at_by_source[source] = cached_retrieved_at
                result.cache_used = True
                result.cache_loaded_at = run_at
                continue
            try:
                items = getattr(service, method_name)(
                    source_query,
                    max_results_per_source,
                )
                warnings = list(getattr(service, "last_warnings", []))
                succeeded = bool(
                    getattr(service, "last_attempt_succeeded", True)
                )
            except Exception as exc:
                items = []
                warnings = [f"{source} retrieval failed: {exc}"]
                succeeded = False
            result.evidence_items.extend(items)
            result.warnings.extend(warnings)
            if items:
                result.sources_used.append(source)
            if succeeded:
                result.retrieved_at_by_source[source] = run_at
            if self.cache_enabled and succeeded:
                warning = self.cache.save(
                    source,
                    source_query,
                    max_results_per_source,
                    run_at,
                    [item.to_dict() for item in items],
                    warnings,
                )
                if warning:
                    result.warnings.append(warning)
        source_times = [
            value
            for value in result.retrieved_at_by_source.values()
            if value
        ]
        if source_times:
            result.retrieved_at = max(source_times)
        return result

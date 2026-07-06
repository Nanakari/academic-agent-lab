"""Controlled, optional external evidence retrieval."""

from .arxiv_search_service import ArxivSearchService
from .evidence_cache import EvidenceCache
from .external_evidence_service import ExternalEvidenceService
from .github_search_service import GitHubSearchService
from .query_builder import ExternalSearchQueryBuilder

__all__ = [
    "ArxivSearchService",
    "EvidenceCache",
    "ExternalEvidenceService",
    "ExternalSearchQueryBuilder",
    "GitHubSearchService",
]

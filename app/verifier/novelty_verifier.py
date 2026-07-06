"""Conservative academic-novelty screening with local dedup diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.schemas.research_idea import ResearchIdea
from app.schemas.verification_result import VerificationResult
from app.verifier.claim_filter import is_verifiable_claim


@dataclass
class NoveltyVerificationResult(VerificationResult):
    """Novelty result with separate local and literature assessments."""

    novelty_assessment_version: int = 2
    local_memory_overlap: dict = field(default_factory=dict)
    literature_novelty: dict = field(default_factory=dict)


class NoveltyVerifier:
    """Treat local overlap as warning and literature evidence as the decision."""

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        min_related_papers: int = 2,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_related_papers = max(1, int(min_related_papers))

    def verify(
        self,
        idea: ResearchIdea,
        historical_ideas: list[dict],
        literature_analysis: dict | None = None,
        evidence_context: list[dict] | None = None,
        external_literature_evidence: list[dict] | None = None,
    ) -> NoveltyVerificationResult:
        local_overlap = self._local_memory_overlap(idea, historical_ideas)
        literature_novelty = self._literature_novelty(
            idea,
            literature_analysis or {},
            evidence_context or [],
            external_literature_evidence or [],
        )
        status = literature_novelty["status"]
        passed = status == "potentially_distinct"
        if status == "overlapping":
            issues = [
                "The proposed mechanism overlaps substantially with a method "
                "described in the topic-relevant literature."
            ]
        elif status == "insufficient_literature_evidence":
            issues = [
                "Insufficient literature evidence to assess academic novelty; "
                "related papers, benchmarks, methods, or surveys are missing."
            ]
        else:
            issues = []
        warnings = []
        if local_overlap["has_overlap"]:
            warnings.append(
                "Local memory overlap was detected. This is a draft "
                "deduplication warning, not evidence of academic duplication."
            )
        score = (
            max(0.0, 1.0 - literature_novelty["max_method_similarity"])
            if status != "insufficient_literature_evidence"
            else 0.0
        )
        suggestions = (
            []
            if passed
            else [
                "Compare against additional topic-relevant papers, surveys, "
                "benchmarks, and method descriptions before claiming novelty."
            ]
        )
        return NoveltyVerificationResult(
            passed=passed,
            score=round(score, 3),
            issues=issues,
            suggestions=suggestions,
            warnings=warnings,
            local_memory_overlap=local_overlap,
            literature_novelty=literature_novelty,
        )

    def _local_memory_overlap(
        self,
        idea: ResearchIdea,
        historical_ideas: list[dict],
    ) -> dict:
        current = self._tokens(idea.title + " " + idea.method)
        maximum = 0.0
        nearest = None
        for record in historical_ideas:
            previous = self._tokens(
                str(record.get("title", ""))
                + " "
                + str(record.get("method", ""))
            )
            similarity = self._jaccard(current, previous)
            if similarity > maximum:
                maximum = similarity
                nearest = str(
                    record.get("title", "untitled historical idea")
                )
        return {
            "has_overlap": maximum >= self.similarity_threshold,
            "max_similarity": round(maximum, 3),
            "matched_title": nearest,
            "effect": "warning_only",
        }

    def _literature_novelty(
        self,
        idea: ResearchIdea,
        literature_analysis: dict,
        evidence_context: list[dict],
        external_literature_evidence: list[dict],
    ) -> dict:
        compared_papers = []
        evidence_texts = []
        for item in [*evidence_context, *external_literature_evidence]:
            if (
                item.get("support_level") == "insufficient"
                and item.get("source_type") != "arxiv"
            ):
                continue
            title = str(item.get("title") or "").strip()
            if title and title not in compared_papers:
                compared_papers.append(title)
            evidence_texts.append(
                str(
                    item.get("text")
                    or item.get("excerpt")
                    or item.get("summary")
                    or ""
                )
            )

        related_methods = [
            str(method)
            for method in literature_analysis.get("existing_methods", [])
            if is_verifiable_claim(str(method))
        ]
        related_benchmarks = self._benchmark_signals(evidence_texts)
        idea_tokens = self._tokens(idea.method)
        method_similarities = [
            self._jaccard(idea_tokens, self._tokens(method))
            for method in related_methods
        ]
        maximum = max(method_similarities, default=0.0)

        enough_comparison = (
            len(compared_papers) >= self.min_related_papers
            and bool(related_methods)
            and bool(idea_tokens)
        )
        if enough_comparison and maximum >= self.similarity_threshold:
            status = "overlapping"
            risk = "high"
            mechanism_difference = (
                "No reliable mechanism difference was found by the lexical "
                "comparison."
            )
        elif enough_comparison:
            status = "potentially_distinct"
            risk = "medium"
            mechanism_difference = (
                "The proposed method has limited lexical overlap with the "
                "retrieved method descriptions; this is not proof of novelty."
            )
        else:
            status = "insufficient_literature_evidence"
            risk = "unknown"
            mechanism_difference = (
                "Insufficient topic-relevant literature was available for a "
                "mechanism comparison."
            )
        return {
            "status": status,
            "compared_papers": compared_papers,
            "related_methods": related_methods,
            "related_benchmarks": related_benchmarks,
            "mechanism_difference": mechanism_difference,
            "claimed_contribution": idea.method,
            "risk": risk,
            "max_method_similarity": round(maximum, 3),
            "minimum_related_papers": self.min_related_papers,
        }

    @staticmethod
    def _benchmark_signals(texts: list[str]) -> list[str]:
        signals = []
        for text in texts:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                lowered = sentence.casefold()
                if any(
                    cue in lowered
                    for cue in ("benchmark", "dataset", "evaluation", "survey")
                ):
                    cleaned = " ".join(sentence.split())[:300]
                    if cleaned and cleaned not in signals:
                        signals.append(cleaned)
        return signals[:5]

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left | right))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[\w-]+", text.casefold())
            if len(token) > 2
        }

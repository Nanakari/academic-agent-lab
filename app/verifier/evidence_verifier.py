"""Evidence coverage and overclaim checks."""

from __future__ import annotations

import re

from app.schemas.research_idea import ResearchIdea
from app.schemas.verification_result import VerificationResult
from app.tools.paper_corpus import keyword_tokens


class EvidenceVerifier:
    """Check evidence quality and lexical support for ideas and key claims."""

    OVERCLAIMS = ("state-of-the-art", "guarantee", "solves", "always", "完全解决", "保证")

    def __init__(self, minimum_evidence_score: float = 0.12) -> None:
        self.minimum_evidence_score = minimum_evidence_score

    def verify(
        self,
        idea: ResearchIdea,
        evidence_context: list[dict],
        claims: list[str] | None = None,
        ideas: list[ResearchIdea] | None = None,
    ) -> VerificationResult:
        issues: list[str] = []
        suggestions: list[str] = []
        evidence_ids = {
            item.get("evidence_id") for item in evidence_context if item.get("evidence_id")
        }
        top_score = max(
            (float(item.get("score", 0.0)) for item in evidence_context),
            default=0.0,
        )
        if not evidence_context:
            issues.append("unsupported: no local paper or memory evidence was retrieved.")
            suggestions.append("Add relevant .txt, .md, or .pdf papers to data/papers/.")
        elif top_score < self.minimum_evidence_score:
            issues.append(
                f"weakly_supported: top evidence score {top_score:.3f} is below "
                f"{self.minimum_evidence_score:.3f}."
            )
            suggestions.append("Add papers whose terminology more directly matches the topic.")

        checked_ideas = ideas or [idea]
        seen_titles = set()
        for candidate in checked_ideas:
            if candidate.title in seen_titles:
                continue
            seen_titles.add(candidate.title)
            cited_ids = set(candidate.evidence_refs)
            support = self._best_support(
                " ".join(
                    [
                        candidate.title,
                        candidate.hypothesis,
                        candidate.motivation,
                        candidate.method,
                    ]
                ),
                evidence_context,
            )
            if not cited_ids.intersection(evidence_ids) or support < 0.1:
                issues.append(f"unsupported idea: {candidate.title}")
                suggestions.append(
                    f"Ground '{candidate.title}' in a retrieved chunk or label it exploratory."
                )

        for index, claim in enumerate(claims or [], start=1):
            if self._best_support(claim, evidence_context) < 0.1:
                issues.append(f"unsupported claim {index}: {claim}")
                suggestions.append(
                    f"Add a supporting evidence chunk for key claim {index}."
                )

        generated_text = " ".join(
            candidate.hypothesis + " " + candidate.motivation
            for candidate in checked_ideas
        )
        found_overclaims = [
            term for term in self.OVERCLAIMS
            if self._contains_overclaim(generated_text, term)
        ]
        if found_overclaims:
            issues.append(f"Overclaiming language detected: {', '.join(found_overclaims)}.")
            suggestions.append("Use conditional, testable wording and state uncertainty.")

        total_checks = len(seen_titles) + len(claims or []) + 1
        failed_checks = len([
            issue for issue in issues
            if issue.startswith(("unsupported", "weakly_supported"))
        ])
        coverage = max(0.0, (total_checks - failed_checks) / max(1, total_checks))
        score = min(1.0, 0.5 * coverage + 0.5 * top_score)
        if found_overclaims:
            score = max(0.0, score - 0.2)
        return VerificationResult(
            passed=not issues,
            score=round(score, 3),
            issues=issues,
            suggestions=list(dict.fromkeys(suggestions)),
        )

    @staticmethod
    def _best_support(statement: str, evidence_context: list[dict]) -> float:
        statement_terms = keyword_tokens(statement)
        if not statement_terms:
            return 0.0
        denominator = min(len(statement_terms), 10)
        return max(
            (
                len(
                    statement_terms
                    & keyword_tokens(item.get("text") or item.get("excerpt", ""))
                )
                / max(1, denominator)
                for item in evidence_context
            ),
            default=0.0,
        )

    @staticmethod
    def _contains_overclaim(text: str, term: str) -> bool:
        normalized = text.casefold()
        if term == "always" and re.search(r"\b(?:not|never)\s+always\b", normalized):
            normalized = re.sub(r"\b(?:not|never)\s+always\b", "", normalized)
        if re.fullmatch(r"[a-z-]+", term):
            return re.search(rf"\b{re.escape(term)}\b", normalized) is not None
        return term in normalized

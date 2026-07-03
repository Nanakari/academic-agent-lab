"""Evidence coverage and overclaim checks."""

from __future__ import annotations

import re

from app.schemas.research_idea import ResearchIdea
from app.schemas.verification_result import VerificationResult


class EvidenceVerifier:
    """Check that an idea cites retrieved context and avoids absolute claims."""

    OVERCLAIMS = ("state-of-the-art", "guarantee", "solves", "always", "完全解决", "保证")

    def verify(
        self,
        idea: ResearchIdea,
        evidence_context: list[dict],
        claims: list[str] | None = None,
    ) -> VerificationResult:
        issues = []
        suggestions = []
        evidence_ids = {item.get("evidence_id") for item in evidence_context}
        cited_ids = set(idea.evidence_refs)
        if not evidence_context:
            issues.append("unsupported: no local paper or memory evidence was retrieved.")
            suggestions.append("Add relevant papers to data/ and rerun evidence retrieval.")
        elif not cited_ids.intersection(evidence_ids):
            issues.append("weakly_supported: the selected idea does not cite retrieved evidence IDs.")
            suggestions.append("Attach one or more evidence IDs to the idea motivation.")

        claim_text = " ".join(claims or []) + " " + idea.hypothesis + " " + idea.motivation
        found_overclaims = [term for term in self.OVERCLAIMS if term in claim_text.casefold()]
        if found_overclaims:
            issues.append(f"Overclaiming language detected: {', '.join(found_overclaims)}.")
            suggestions.append("Use conditional, testable wording and state uncertainty.")

        # Reward lexical contact between the idea rationale and retrieved evidence.
        evidence_words = self._tokens(" ".join(item.get("excerpt", "") for item in evidence_context))
        idea_words = self._tokens(idea.motivation + " " + idea.hypothesis)
        overlap = len(evidence_words & idea_words) / max(1, len(idea_words))
        score = min(1.0, 0.55 + overlap) if evidence_context else 0.2
        score -= 0.2 * len(found_overclaims)
        passed = bool(evidence_context) and bool(cited_ids.intersection(evidence_ids)) and not found_overclaims
        return VerificationResult(passed, round(max(0.0, score), 3), issues, suggestions)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[\w-]+", text.casefold()) if len(token) > 2}

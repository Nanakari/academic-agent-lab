"""Claim-level evidence coverage and overclaim checks."""

from __future__ import annotations

import re

from app.schemas.evidence import support_level_for_score
from app.schemas.research_idea import ResearchIdea
from app.schemas.verification_result import VerificationResult
from app.tools.paper_corpus import infer_supporting_claim, keyword_tokens
from app.verifier.topic_consistency import domain_consistency_score


class EvidenceVerifier:
    """Attach the best local citation to every generated idea and key claim."""

    OVERCLAIMS = ("state-of-the-art", "guarantee", "solves", "always", "完全解决", "保证")
    UNSUPPORTED_MARKERS = (
        "not explicitly stated",
        "evidence is insufficient",
        "cannot yet be established",
        "原文未明确说明",
        "证据不足",
    )

    def __init__(
        self,
        minimum_evidence_score: float = 0.15,
        strict_domain: bool = False,
    ) -> None:
        self.minimum_evidence_score = minimum_evidence_score
        self.strict_domain = strict_domain

    def verify(
        self,
        idea: ResearchIdea,
        evidence_context: list[dict],
        claims: list[str] | None = None,
        ideas: list[ResearchIdea] | None = None,
        topic: str | None = None,
    ) -> VerificationResult:
        issues: list[str] = []
        suggestions: list[str] = []
        supported_claims: list[dict] = []
        unsupported_claims: list[str] = []
        evidence_used: list[dict] = []
        support_scores: list[float] = []
        evidence_ids = {
            item.get("evidence_id") for item in evidence_context if item.get("evidence_id")
        }
        domain_consistency = {}

        if not evidence_context:
            issues.append("no evidence found in the local paper corpus or scientific memory.")
            suggestions.append("Add relevant .txt, .md, or .pdf papers to data/papers/.")
        else:
            top_score = max(float(item.get("score", 0.0)) for item in evidence_context)
            if top_score < self.minimum_evidence_score:
                issues.append(
                    f"top evidence score {top_score:.3f} is below "
                    f"{self.minimum_evidence_score:.3f}."
                )
                suggestions.append(
                    "Add papers whose terminology more directly matches the research topic."
                )

        if self.strict_domain and topic:
            domain_consistency = domain_consistency_score(topic, evidence_context)
            if not domain_consistency["passed"]:
                reason = "; ".join(domain_consistency["issues"])
                issues.append(
                    f"topic-domain consistency check failed: {reason}"
                )
                unsupported_claims.append(
                    "No retrieved evidence matched topic-critical concepts."
                )
                suggestions.append(
                    "Add papers containing the topic's specific concept combinations."
                )

        checked_ideas = self._unique_ideas(ideas or [idea])
        for candidate in checked_ideas:
            statement = " ".join(
                [
                    candidate.title,
                    candidate.hypothesis,
                    candidate.motivation,
                    candidate.method,
                ]
            )
            support = self._best_evidence(statement, evidence_context)
            support_scores.append(support["score"])
            has_reference = bool(set(candidate.evidence_refs) & evidence_ids)
            label = f"idea: {candidate.title}"
            if support["support_level"] != "insufficient" and has_reference:
                citation = self._citation(label, support)
                supported_claims.append(citation)
                evidence_used.append(citation)
            else:
                unsupported_claims.append(label)
                issues.append(
                    f"unsupported {label} "
                    f"(support={support['support_level']}, score={support['score']:.3f})."
                )
                suggestions.append(
                    f"Add direct paper evidence for '{candidate.title}' or label it speculative."
                )

        for index, claim in enumerate(claims or [], start=1):
            support = self._best_evidence(claim, evidence_context)
            support_scores.append(support["score"])
            label = f"key claim {index}: {claim}"
            if support["support_level"] != "insufficient":
                citation = self._citation(label, support)
                supported_claims.append(citation)
                evidence_used.append(citation)
            else:
                unsupported_claims.append(label)
                issues.append(
                    f"unsupported {label} "
                    f"(support={support['support_level']}, score={support['score']:.3f})."
                )
                suggestions.append(f"Retrieve a paper chunk supporting key claim {index}.")

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

        score = sum(support_scores) / len(support_scores) if support_scores else 0.0
        if found_overclaims:
            score = max(0.0, score - 0.2)
        return VerificationResult(
            passed=bool(evidence_context) and not issues,
            score=round(score, 3),
            issues=issues,
            suggestions=list(dict.fromkeys(suggestions)),
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            evidence_used=self._deduplicate_citations(evidence_used),
            support_level=support_level_for_score(score),
            domain_consistency=domain_consistency,
        )

    @classmethod
    def _best_evidence(cls, statement: str, evidence_context: list[dict]) -> dict:
        best = {
            "score": 0.0,
            "support_level": "insufficient",
            "matched_keywords": [],
            "supporting_claim": None,
            "evidence": None,
        }
        if any(marker in statement.casefold() for marker in cls.UNSUPPORTED_MARKERS):
            return best
        statement_terms = keyword_tokens(statement)
        if not statement_terms:
            return best

        denominator = min(len(statement_terms), 10)
        for evidence in evidence_context:
            text = evidence.get("text") or evidence.get("excerpt", "")
            matched = sorted(statement_terms & keyword_tokens(text))
            lexical_score = min(1.0, len(matched) / max(1, denominator))
            # Claim support cannot exceed the retrieval confidence assigned to
            # the source chunk.
            score = min(lexical_score, float(evidence.get("score", 0.0)))
            if score > best["score"]:
                best = {
                    "score": round(score, 3),
                    "support_level": support_level_for_score(score),
                    "matched_keywords": matched,
                    "supporting_claim": infer_supporting_claim(statement, text),
                    "evidence": evidence,
                }
        return best

    @staticmethod
    def _citation(claim: str, support: dict) -> dict:
        evidence = support["evidence"] or {}
        return {
            "claim": claim,
            "evidence_id": evidence.get("evidence_id"),
            "paper_id": evidence.get("paper_id"),
            "title": evidence.get("title"),
            "source_path": evidence.get("source_path"),
            "file_type": evidence.get("file_type"),
            "page": evidence.get("page"),
            "section": evidence.get("section"),
            "chunk_id": evidence.get("chunk_id"),
            "score": support["score"],
            "matched_keywords": support["matched_keywords"],
            "supporting_claim": support["supporting_claim"],
            "support_level": support["support_level"],
        }

    @staticmethod
    def _unique_ideas(ideas: list[ResearchIdea]) -> list[ResearchIdea]:
        unique = []
        seen = set()
        for candidate in ideas:
            if candidate.title not in seen:
                seen.add(candidate.title)
                unique.append(candidate)
        return unique

    @staticmethod
    def _deduplicate_citations(citations: list[dict]) -> list[dict]:
        unique = []
        seen = set()
        for citation in citations:
            key = (citation["claim"], citation["evidence_id"])
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        return unique

    @staticmethod
    def _contains_overclaim(text: str, term: str) -> bool:
        normalized = text.casefold()
        if term == "always" and re.search(r"\b(?:not|never)\s+always\b", normalized):
            normalized = re.sub(r"\b(?:not|never)\s+always\b", "", normalized)
        if re.fullmatch(r"[a-z-]+", term):
            return re.search(rf"\b{re.escape(term)}\b", normalized) is not None
        return term in normalized

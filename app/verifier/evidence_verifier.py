"""Claim-level evidence coverage and overclaim checks."""

from __future__ import annotations

import re

from app.schemas.evidence import support_level_for_score
from app.schemas.research_idea import ResearchIdea
from app.schemas.verification_result import VerificationResult
from app.tools.paper_corpus import infer_supporting_claim, keyword_tokens
from app.verifier.topic_consistency import (
    TopicConsistencyConfig,
    domain_consistency_score,
    resolve_domain_mode,
)


class EvidenceVerifier:
    """Attach the best local citation to every generated idea and key claim."""

    OVERCLAIMS = (
        "state-of-the-art",
        "guarantee",
        "solves",
        "always",
        "\u5b8c\u5168\u89e3\u51b3",
        "\u4fdd\u8bc1",
    )
    UNSUPPORTED_MARKERS = (
        "not explicitly stated",
        "evidence is insufficient",
        "cannot yet be established",
        "\u539f\u6587\u672a\u660e\u786e\u8bf4\u660e",
        "\u8bc1\u636e\u4e0d\u8db3",
    )

    def __init__(
        self,
        minimum_evidence_score: float = 0.15,
        strict_domain: bool | None = None,
        domain_mode: str = "off",
    ) -> None:
        self.minimum_evidence_score = minimum_evidence_score
        self.domain_mode = resolve_domain_mode(domain_mode, strict_domain)
        # Retain the legacy attribute for callers that inspect it.
        self.strict_domain = self.domain_mode == "strict"

    def verify(
        self,
        idea: ResearchIdea,
        evidence_context: list[dict],
        claims: list[str] | None = None,
        ideas: list[ResearchIdea] | None = None,
        topic: str | None = None,
    ) -> VerificationResult:
        """Run availability, domain, claim support, and language checks."""
        availability = self._check_evidence_availability(evidence_context)
        domain = self._check_domain_consistency(topic, evidence_context)
        evidence_ids = {
            item.get("evidence_id")
            for item in evidence_context
            if item.get("evidence_id")
        }
        idea_support = self._verify_idea_support(
            ideas or [idea],
            evidence_context,
            evidence_ids,
        )
        claim_support = self._verify_claim_support(
            claims or [],
            evidence_context,
        )
        overclaims = self._check_overclaims(ideas or [idea])

        sections = (
            availability,
            domain,
            idea_support,
            claim_support,
            overclaims,
        )
        issues = [
            issue for section in sections for issue in section.get("issues", [])
        ]
        warnings = [
            warning for section in sections for warning in section.get("warnings", [])
        ]
        suggestions = [
            suggestion
            for section in sections
            for suggestion in section.get("suggestions", [])
        ]
        supported_claims = (
            idea_support["supported_claims"]
            + claim_support["supported_claims"]
        )
        unsupported_claims = (
            domain["unsupported_claims"]
            + idea_support["unsupported_claims"]
            + claim_support["unsupported_claims"]
        )
        evidence_used = (
            idea_support["evidence_used"]
            + claim_support["evidence_used"]
        )
        support_scores = idea_support["scores"] + claim_support["scores"]
        score = self._aggregate_score(
            support_scores,
            overclaims["found_overclaims"],
        )
        return VerificationResult(
            passed=bool(evidence_context) and not issues,
            score=round(score, 3),
            issues=list(dict.fromkeys(issues)),
            suggestions=list(dict.fromkeys(suggestions)),
            supported_claims=supported_claims,
            unsupported_claims=list(dict.fromkeys(unsupported_claims)),
            evidence_used=self._deduplicate_citations(evidence_used),
            support_level=support_level_for_score(score),
            domain_consistency=domain["domain_consistency"],
            warnings=list(dict.fromkeys(warnings)),
        )

    def _check_evidence_availability(
        self,
        evidence_context: list[dict],
    ) -> dict:
        """Check whether evidence exists and clears the retrieval threshold."""
        issues = []
        suggestions = []
        if not evidence_context:
            issues.append(
                "no evidence found in the local paper corpus or scientific memory."
            )
            suggestions.append("Add relevant .txt, .md, or .pdf papers to data/papers/.")
        else:
            top_score = max(
                float(item.get("score", 0.0)) for item in evidence_context
            )
            if top_score < self.minimum_evidence_score:
                issues.append(
                    f"top evidence score {top_score:.3f} is below "
                    f"{self.minimum_evidence_score:.3f}."
                )
                suggestions.append(
                    "Add papers whose terminology more directly matches the research topic."
                )
        return {"issues": issues, "suggestions": suggestions}

    def _check_domain_consistency(
        self,
        topic: str | None,
        evidence_context: list[dict],
    ) -> dict:
        """Apply off, warning, or strict policy to raw concept consistency."""
        result = {
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "unsupported_claims": [],
            "domain_consistency": {},
        }
        if self.domain_mode == "off" or not topic:
            return result

        consistency = domain_consistency_score(
            topic,
            evidence_context,
            TopicConsistencyConfig(mode=self.domain_mode),
        )
        result["domain_consistency"] = consistency
        if consistency["passed"]:
            return result

        message = (
            "topic-domain consistency check failed: "
            f"{consistency['reason']}"
        )
        suggestion = (
            "Add papers containing the topic's specific concept combinations."
        )
        result["suggestions"].append(suggestion)
        if self.domain_mode == "warning":
            result["warnings"].append(message)
        else:
            result["issues"].append(message)
            result["unsupported_claims"].append(
                "No retrieved evidence matched topic-critical concepts."
            )
        return result

    def _verify_idea_support(
        self,
        ideas: list[ResearchIdea],
        evidence_context: list[dict],
        evidence_ids: set,
    ) -> dict:
        """Verify candidate ideas and their explicit evidence references."""
        result = self._empty_support_result()
        for candidate in self._unique_ideas(ideas):
            statement = " ".join([
                candidate.title,
                candidate.hypothesis,
                candidate.motivation,
                candidate.method,
            ])
            support = self._best_evidence(statement, evidence_context)
            result["scores"].append(support["score"])
            has_reference = bool(set(candidate.evidence_refs) & evidence_ids)
            label = f"idea: {candidate.title}"
            if support["support_level"] != "insufficient" and has_reference:
                citation = self._citation(label, support)
                result["supported_claims"].append(citation)
                result["evidence_used"].append(citation)
            else:
                result["unsupported_claims"].append(label)
                result["issues"].append(
                    f"unsupported {label} "
                    f"(support={support['support_level']}, "
                    f"score={support['score']:.3f})."
                )
                result["suggestions"].append(
                    f"Add direct paper evidence for '{candidate.title}' "
                    "or label it speculative."
                )
        return result

    def _verify_claim_support(
        self,
        claims: list[str],
        evidence_context: list[dict],
    ) -> dict:
        """Verify key claims and create claim-to-evidence citations."""
        result = self._empty_support_result()
        for index, claim in enumerate(claims, start=1):
            support = self._best_evidence(claim, evidence_context)
            result["scores"].append(support["score"])
            label = f"key claim {index}: {claim}"
            if support["support_level"] != "insufficient":
                citation = self._citation(label, support)
                result["supported_claims"].append(citation)
                result["evidence_used"].append(citation)
            else:
                result["unsupported_claims"].append(label)
                result["issues"].append(
                    f"unsupported {label} "
                    f"(support={support['support_level']}, "
                    f"score={support['score']:.3f})."
                )
                result["suggestions"].append(
                    f"Retrieve a paper chunk supporting key claim {index}."
                )
        return result

    def _check_overclaims(self, ideas: list[ResearchIdea]) -> dict:
        """Detect strong language that exceeds the lightweight evidence check."""
        checked_ideas = self._unique_ideas(ideas)
        generated_text = " ".join(
            candidate.hypothesis + " " + candidate.motivation
            for candidate in checked_ideas
        )
        found = [
            term for term in self.OVERCLAIMS
            if self._contains_overclaim(generated_text, term)
        ]
        return {
            "found_overclaims": found,
            "issues": (
                [f"Overclaiming language detected: {', '.join(found)}."]
                if found else []
            ),
            "suggestions": (
                ["Use conditional, testable wording and state uncertainty."]
                if found else []
            ),
        }

    @staticmethod
    def _aggregate_score(
        scores: list[float],
        found_overclaims: list[str],
    ) -> float:
        """Average support scores and apply the existing overclaim penalty."""
        score = sum(scores) / len(scores) if scores else 0.0
        if found_overclaims:
            score = max(0.0, score - 0.2)
        return score

    @staticmethod
    def _empty_support_result() -> dict:
        return {
            "issues": [],
            "suggestions": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "evidence_used": [],
            "scores": [],
        }

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

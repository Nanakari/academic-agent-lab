"""Topic-aware admission rules for external literature evidence."""

from __future__ import annotations

import re
from typing import Any

from app.tools.query_normalizer import (
    GENERIC_RESEARCH_TERMS,
    english_research_terms,
    research_query_expansions,
)
from app.verifier.topic_consistency import evidence_matches_concept


SECURITY_TRIGGER_TERMS = {
    "attack",
    "defense",
    "injection",
    "permission",
    "safety",
    "security",
}
SECURITY_CORE_PHRASES = {
    "ai control",
    "agent attack",
    "agent security",
    "attack surface",
    "code execution",
    "confused deputy",
    "executable tools",
    "indirect attack",
    "indirect prompt injection",
    "malicious instruction",
    "permission escalation",
    "prompt injection",
    "prompt-injected",
    "safety monitoring",
    "tool calling",
    "tool governance",
    "tool permission",
    "tool security",
    "tool use",
    "untrusted content",
}
EXCLUDED_DOMAIN_PHRASES = {
    "4d generation",
    "diffusion model",
    "fuzzy functions",
    "localization precision",
    "program-as-weights",
    "unlearning",
    "video world model",
    "x-to-4d",
}


def _value(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("‑", "-")).strip()


def _matched_phrases(text: str, phrases: set[str]) -> list[str]:
    return sorted(
        phrase
        for phrase in phrases
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
    )


def is_external_evidence_relevant_to_topic(
    item: Any,
    expanded_query: str,
    topic: str,
) -> tuple[bool, str]:
    """Return whether an external item may influence literature analysis."""
    source_type = str(_value(item, "source_type", ""))
    if source_type != "arxiv":
        return False, (
            "Rejected because only arXiv literature evidence may enter "
            "literature analysis; GitHub remains implementation evidence."
        )
    evidence_status = str(_value(item, "evidence_status", "retrieved"))
    if evidence_status != "retrieved":
        return False, f"Rejected because evidence_status={evidence_status!r}."
    relevance_score = float(_value(item, "relevance_score", 0.0) or 0.0)
    if relevance_score < 0.15:
        return False, (
            f"Rejected because relevance_score={relevance_score:.3f} is below "
            "the 0.150 admission threshold."
        )

    title = str(_value(item, "title", ""))
    summary = str(_value(item, "summary", ""))
    combined = _normalized_text(f"{title} {summary}")
    query_terms = english_research_terms(expanded_query)
    specific_query_terms = query_terms - GENERIC_RESEARCH_TERMS
    evidence_terms = english_research_terms(combined)
    matched_specific = sorted(specific_query_terms & evidence_terms)

    expansion_phrases = {
        phrase.casefold()
        for phrase in research_query_expansions(topic)
        if len(phrase.split()) >= 2
    }
    security_topic = bool(query_terms & SECURITY_TRIGGER_TERMS)
    active_core_phrases = set(expansion_phrases)
    if security_topic:
        active_core_phrases.update(SECURITY_CORE_PHRASES)
    matched_core = _matched_phrases(combined, active_core_phrases)
    excluded = _matched_phrases(combined, EXCLUDED_DOMAIN_PHRASES)

    concept_match = evidence_matches_concept(summary, title, expanded_query)
    required_specific_matches = min(
        2,
        max(1, len(specific_query_terms)),
    )
    dynamic_concept_match = (
        concept_match["domain_consistent"]
        and len(matched_specific) >= required_specific_matches
    )
    has_core_match = bool(matched_core or dynamic_concept_match)
    if excluded and not has_core_match:
        return False, (
            "Rejected because it matches excluded domain signal(s): "
            f"{', '.join(excluded)}, and lacks topic-core concepts."
        )
    if not has_core_match:
        matched_text = ", ".join(matched_specific) or "none"
        return False, (
            "Rejected because generic lexical overlap is insufficient; "
            f"matched topic-specific terms: {matched_text}."
        )

    reasons = []
    if matched_core:
        reasons.append(f"core phrase(s): {', '.join(matched_core)}")
    if dynamic_concept_match:
        reasons.append(
            "dynamic topic concepts with specific terms: "
            + ", ".join(matched_specific)
        )
    return True, "Accepted based on " + "; ".join(reasons) + "."


def has_excluded_topic_drift(text: str, expanded_query: str, topic: str) -> bool:
    """Defensive check for an analysis sentence that drifts to excluded domains."""
    probe = {
        "source_type": "arxiv",
        "title": "",
        "summary": text,
        "relevance_score": 1.0,
        "evidence_status": "retrieved",
    }
    accepted, reason = is_external_evidence_relevant_to_topic(
        probe,
        expanded_query,
        topic,
    )
    return not accepted and "excluded domain signal" in reason

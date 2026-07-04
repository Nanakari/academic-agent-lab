"""Explainable lexical topic-concept checks for evidence validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.tools.paper_corpus import keyword_tokens

DomainConsistencyMode = Literal["off", "warning", "strict"]
ALLOWED_DOMAIN_MODES = {"off", "warning", "strict"}

GENERIC_TERMS = {
    "model",
    "method",
    "system",
    "network",
    "data",
    "task",
    "prediction",
    "learning",
    "based",
    "using",
}
DOMAIN_HEADS = {"prediction", "segmentation", "consensus"}
KNOWN_RELATION_PHRASES = {
    "planning failure",
    "hallucination mitigation",
    "tool use",
    "context drift",
    "multi agent collaboration",
    "hallucination in action",
}
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


@dataclass(frozen=True)
class TopicConsistencyConfig:
    """Thresholds and diagnostic mode for lexical concept checking."""

    mode: DomainConsistencyMode = "off"
    min_domain_consistent_evidence: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", resolve_domain_mode(self.mode))
        if self.min_domain_consistent_evidence < 1:
            raise ValueError("min_domain_consistent_evidence must be at least 1")


def resolve_domain_mode(
    domain_mode: str = "off",
    strict_domain: bool | None = None,
) -> DomainConsistencyMode:
    """Resolve the legacy boolean and validate the canonical mode."""
    if strict_domain is True:
        return "strict"
    if strict_domain is False:
        return "off"
    normalized = str(domain_mode).casefold()
    if normalized not in ALLOWED_DOMAIN_MODES:
        allowed = ", ".join(sorted(ALLOWED_DOMAIN_MODES))
        raise ValueError(
            f"Invalid domain_mode {domain_mode!r}; allowed values are: {allowed}."
        )
    return normalized  # type: ignore[return-value]


def _normalize_token(token: str) -> str:
    token = token.casefold()
    if (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def _ordered_tokens(text: str) -> list[str]:
    # The word regex naturally normalizes hyphenated expressions into tokens.
    return [_normalize_token(token) for token in WORD_PATTERN.findall(text)]


def extract_topic_concepts(topic: str) -> dict:
    """Extract conservative critical anchors plus broad diagnostic n-grams."""
    tokens = _ordered_tokens(topic)
    candidate_phrases: list[list[str]] = []
    if len(tokens) >= 2:
        leading_size = 3 if (
            len(tokens) >= 3
            and (
                tokens[2] in DOMAIN_HEADS
                or tokens[:3] == ["graph", "neural", "network"]
            )
        ) else 2
        candidate_phrases.append(tokens[:leading_size])

    for index, token in enumerate(tokens):
        if token in DOMAIN_HEADS and index > 0:
            candidate_phrases.append(tokens[index - 1:index + 1])
        if (
            token == "network"
            and index >= 2
            and tokens[index - 2:index + 1] == ["graph", "neural", "network"]
        ):
            candidate_phrases.append(tokens[index - 2:index + 1])

    for size in (3, 2):
        for start in range(len(tokens) - size + 1):
            phrase_tokens = tokens[start:start + size]
            if " ".join(phrase_tokens) in KNOWN_RELATION_PHRASES:
                candidate_phrases.append(phrase_tokens)

    concept_phrases: list[str] = []
    critical_groups: list[set[str]] = []
    seen_groups = set()
    for phrase_tokens in candidate_phrases:
        phrase = " ".join(phrase_tokens)
        if phrase not in concept_phrases:
            concept_phrases.append(phrase)
        group = frozenset(phrase_tokens)
        if group not in seen_groups:
            seen_groups.add(group)
            critical_groups.append(set(group))

    # Broad n-grams are diagnostic only. They must never create a strict pass.
    coverage_phrases = []
    for size in (3, 2):
        for start in range(len(tokens) - size + 1):
            phrase_tokens = tokens[start:start + size]
            if all(token in GENERIC_TERMS for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            if phrase not in coverage_phrases:
                coverage_phrases.append(phrase)

    return {
        "raw_terms": keyword_tokens(topic),
        "concept_phrases": concept_phrases,
        "critical_groups": critical_groups,
        "coverage_phrases": coverage_phrases,
        "generic_terms": set(GENERIC_TERMS),
    }


def evidence_matches_concept(
    evidence_text: str,
    title: str,
    topic: str,
) -> dict:
    """Describe critical and broad lexical matches for one evidence item."""
    concepts = extract_topic_concepts(topic)
    combined_tokens = _ordered_tokens(f"{title} {evidence_text}")
    combined_text = " ".join(combined_tokens)
    evidence_terms = set(combined_tokens)
    matched_phrases = [
        phrase
        for phrase in concepts["concept_phrases"]
        if re.search(rf"\b{re.escape(phrase)}\b", combined_text)
    ]
    matched_groups = [
        sorted(group)
        for group in concepts["critical_groups"]
        if group.issubset(evidence_terms)
    ]
    matched_coverage_phrases = [
        phrase
        for phrase in concepts["coverage_phrases"]
        if re.search(rf"\b{re.escape(phrase)}\b", combined_text)
    ]
    matched_raw = set(_ordered_tokens(topic)) & evidence_terms
    specific_matches = matched_raw - concepts["generic_terms"]
    generic_only = bool(matched_raw) and not specific_matches
    domain_consistent = bool(matched_phrases or matched_groups) and not generic_only
    if domain_consistent:
        reason = "Evidence contains a topic-critical phrase or token group."
    elif matched_coverage_phrases or specific_matches:
        reason = (
            "Some related lexical coverage was found, but no direct "
            "topic-critical concept matched."
        )
    elif generic_only:
        reason = "Evidence matched only generic topic terms."
    else:
        reason = "No direct topic-critical concept matched."
    return {
        "matched_phrases": matched_phrases,
        "matched_critical_groups": matched_groups,
        "matched_coverage_phrases": matched_coverage_phrases,
        "matched_non_generic_terms": sorted(specific_matches),
        "generic_only": generic_only,
        "domain_consistent": domain_consistent,
        "reason": reason,
    }


def domain_consistency_score(
    topic: str,
    evidence_items: list[dict],
    config: TopicConsistencyConfig | None = None,
) -> dict:
    """Return raw concept consistency; verifier mode decides enforcement."""
    settings = config or TopicConsistencyConfig()
    concepts = extract_topic_concepts(topic)
    critical_phrases = concepts["concept_phrases"]
    matched_critical = set()
    consistent_ids = []
    coverage_matches = []
    generic_only_matches = []

    for item in evidence_items:
        evidence_id = item.get("evidence_id")
        match = evidence_matches_concept(
            str(item.get("text") or item.get("excerpt") or ""),
            str(item.get("title") or ""),
            topic,
        )
        item_critical = set(match["matched_phrases"])
        for phrase, group in zip(
            critical_phrases,
            concepts["critical_groups"],
        ):
            if sorted(group) in match["matched_critical_groups"]:
                item_critical.add(phrase)
        if match["domain_consistent"]:
            consistent_ids.append(evidence_id)
            matched_critical.update(item_critical)
        if match["matched_coverage_phrases"] or match["matched_non_generic_terms"]:
            coverage_matches.append({
                "evidence_id": evidence_id,
                "phrases": match["matched_coverage_phrases"],
                "terms": match["matched_non_generic_terms"],
            })
        if match["generic_only"]:
            generic_only_matches.append(evidence_id)

    required = settings.min_domain_consistent_evidence
    passed = len(consistent_ids) >= required
    missing_concepts = [
        phrase for phrase in critical_phrases if phrase not in matched_critical
    ]
    best_score = (
        len(matched_critical) / len(critical_phrases)
        if critical_phrases
        else 0.0
    )
    if passed:
        reason = (
            f"{len(consistent_ids)} evidence item(s) matched reliable "
            "topic-critical concepts."
        )
        issues = []
        warnings = []
    elif coverage_matches:
        reason = (
            "Some related lexical coverage was found, but no direct "
            "topic-critical concept matched."
        )
        issues = [reason]
        warnings = [
            reason
            + " Consider adding lexically closer papers or using a broader topic."
        ]
    else:
        reason = "No direct topic-critical concept matched."
        issues = [reason]
        warnings = [
            reason
            + " Lexical rules cannot infer semantic similarity from different wording."
        ]

    return {
        "passed": passed,
        "mode": settings.mode,
        "best_score": round(best_score, 3),
        "domain_consistent_evidence_ids": [
            evidence_id for evidence_id in consistent_ids if evidence_id
        ],
        "matched_topic_concepts": sorted(matched_critical),
        "missing_topic_concepts": missing_concepts,
        "generic_only_matches": [
            evidence_id for evidence_id in generic_only_matches if evidence_id
        ],
        "coverage_matches": coverage_matches,
        "issues": issues,
        "warnings": warnings,
        "reason": reason,
    }

"""Explainable topic-concept checks for strict real-paper validation."""

from __future__ import annotations

import re

from app.tools.paper_corpus import keyword_tokens

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
    return [_normalize_token(token) for token in WORD_PATTERN.findall(text)]


def extract_topic_concepts(topic: str) -> dict:
    """Extract ordered n-grams and token groups without an NLP dependency."""
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
    return {
        "raw_terms": keyword_tokens(topic),
        "concept_phrases": concept_phrases,
        "critical_groups": critical_groups,
        "generic_terms": set(GENERIC_TERMS),
    }


def evidence_matches_concept(
    evidence_text: str,
    title: str,
    topic: str,
) -> dict:
    """Check whether one evidence item contains a topic-critical combination."""
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
    matched_raw = concepts["raw_terms"] & keyword_tokens(f"{title} {evidence_text}")
    specific_matches = matched_raw - concepts["generic_terms"]
    generic_only = bool(matched_raw) and not specific_matches
    domain_consistent = bool(matched_phrases or matched_groups) and not generic_only
    if domain_consistent:
        reason = "Evidence contains a topic-critical phrase or token group."
    elif generic_only:
        reason = "Evidence matched only generic topic terms."
    else:
        reason = "No evidence item matched a topic-critical phrase or token group."
    return {
        "matched_phrases": matched_phrases,
        "matched_critical_groups": matched_groups,
        "generic_only": generic_only,
        "domain_consistent": domain_consistent,
        "reason": reason,
    }


def domain_consistency_score(topic: str, evidence_items: list[dict]) -> dict:
    """Aggregate per-item concept matches into an explainable strict decision."""
    consistent_ids = []
    matched_concepts = []
    best_score = 0.0
    reasons = []
    concepts = extract_topic_concepts(topic)
    denominator = max(1, len(concepts["concept_phrases"]))
    for item in evidence_items:
        match = evidence_matches_concept(
            str(item.get("text") or item.get("excerpt") or ""),
            str(item.get("title") or ""),
            topic,
        )
        item_concepts = list(match["matched_phrases"])
        item_concepts.extend(
            " + ".join(group) for group in match["matched_critical_groups"]
        )
        best_score = max(best_score, min(1.0, len(item_concepts) / denominator))
        if match["domain_consistent"]:
            consistent_ids.append(item.get("evidence_id"))
            matched_concepts.extend(item_concepts)
        else:
            reasons.append(match["reason"])

    passed = bool(consistent_ids)
    issues = [] if passed else [
        reasons[0] if reasons else "No evidence was available for domain checking."
    ]
    return {
        "passed": passed,
        "best_score": round(best_score, 3),
        "domain_consistent_evidence_ids": [
            evidence_id for evidence_id in consistent_ids if evidence_id
        ],
        "matched_topic_concepts": list(dict.fromkeys(matched_concepts)),
        "issues": issues,
    }

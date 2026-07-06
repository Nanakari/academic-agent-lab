"""Offline query expansion and lexical relevance scoring."""

from __future__ import annotations

import re


CHINESE_RESEARCH_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("间接提示注入", ("indirect prompt injection", "prompt injection")),
    ("长期记忆", ("long-term memory", "memory")),
    ("计算机使用", ("computer use",)),
    ("浏览器代理", ("web agent", "browser agent")),
    ("上下文漂移", ("context drift",)),
    ("提示注入", ("prompt injection",)),
    ("工具调用", ("tool calling", "tool use")),
    ("工具使用", ("tool use",)),
    ("智能体", ("agent",)),
    ("大模型", ("LLM",)),
    ("记忆", ("memory",)),
    ("安全", ("security", "safety")),
    ("攻击", ("attack",)),
    ("防御", ("defense",)),
    ("代理", ("agent",)),
    ("MCP", ("Model Context Protocol",)),
)

ENGLISH_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]*")
RELEVANCE_STOPWORDS = {
    "about",
    "after",
    "and",
    "for",
    "from",
    "into",
    "of",
    "on",
    "the",
    "this",
    "under",
    "with",
}
GENERIC_RESEARCH_TERMS = {"agent", "agents", "ai", "llm", "model", "models"}


def research_query_expansions(query: str) -> list[str]:
    """Return deterministic English phrases triggered by Chinese terms."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []
    lowered = normalized_query.casefold()
    expansions: list[str] = []
    seen = set()
    for chinese_term, english_phrases in CHINESE_RESEARCH_EXPANSIONS:
        if chinese_term.casefold() not in lowered:
            continue
        for phrase in english_phrases:
            key = phrase.casefold()
            if key in seen or key in lowered:
                continue
            seen.add(key)
            expansions.append(phrase)
    return expansions


def normalize_research_query(query: str) -> str:
    """Append mapped English keywords without replacing the original query."""
    original = str(query or "").strip()
    if not original:
        return ""
    expansions = research_query_expansions(original)
    return " ".join([original, *expansions])


def english_research_terms(query: str) -> set[str]:
    """Extract English terms used by the lightweight relevance scorer."""
    return {
        token.casefold()
        for token in ENGLISH_TOKEN_PATTERN.findall(str(query or ""))
        if len(token) >= 2 and token.casefold() not in RELEVANCE_STOPWORDS
    }


def score_query_relevance(query: str, title: str, summary: str) -> float:
    """Score query coverage while discounting generic LLM/agent-only matches."""
    query_terms = english_research_terms(query)
    specific_terms = query_terms - GENERIC_RESEARCH_TERMS
    scoring_terms = specific_terms or query_terms
    if not scoring_terms:
        return 0.0

    summary_terms = english_research_terms(summary)
    title_terms = english_research_terms(title)
    summary_score = len(scoring_terms & summary_terms) / len(scoring_terms)
    title_bonus = 0.1 * len(scoring_terms & title_terms) / len(scoring_terms)
    return round(min(1.0, summary_score + title_bonus), 3)

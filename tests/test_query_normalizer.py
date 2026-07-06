"""Tests for offline Chinese research-query expansion."""

from __future__ import annotations

import unittest

from app.tools.paper_corpus import PaperCorpusIndexer
from app.tools.query_normalizer import (
    normalize_research_query,
    score_query_relevance,
)
from app.verifier.topic_consistency import (
    TopicConsistencyConfig,
    domain_consistency_score,
)


class QueryNormalizerTests(unittest.TestCase):
    def test_indirect_prompt_injection_is_expanded(self) -> None:
        expanded = normalize_research_query("间接提示注入攻击")

        self.assertIn("indirect prompt injection", expanded)
        self.assertIn("prompt injection", expanded)
        self.assertIn("attack", expanded)

    def test_tool_calling_security_is_expanded(self) -> None:
        expanded = normalize_research_query("工具调用安全")

        self.assertIn("tool calling", expanded)
        self.assertIn("tool use", expanded)
        self.assertIn("security", expanded)
        self.assertIn("safety", expanded)

    def test_english_query_is_not_changed(self) -> None:
        query = "indirect prompt injection tool calling security"

        self.assertEqual(normalize_research_query(query), query)

    def test_empty_query_is_safe(self) -> None:
        self.assertEqual(normalize_research_query(""), "")
        self.assertEqual(normalize_research_query("   "), "")

    def test_expansion_improves_english_paper_matching(self) -> None:
        query = "LLM Agent 在间接提示注入攻击下的工具调用安全"
        text = (
            "Indirect prompt injection attacks compromise tool calling and "
            "tool use security in LLM agents."
        )

        raw_score = PaperCorpusIndexer.score_text(query, text)
        expanded_score = PaperCorpusIndexer.score_text(
            normalize_research_query(query),
            text,
        )

        self.assertGreater(expanded_score, raw_score)
        self.assertGreaterEqual(expanded_score, 0.15)

    def test_relevance_scorer_rejects_generic_llm_match(self) -> None:
        query = normalize_research_query(
            "LLM Agent 在间接提示注入攻击下的工具调用安全"
        )

        relevant = score_query_relevance(
            query,
            "Indirect Prompt Injection Against Tool-Using Agents",
            "We evaluate tool calling security under prompt injection attacks.",
        )
        unrelated = score_query_relevance(
            query,
            "Unlearning in Large Language Models",
            "We study efficient removal of memorized training examples.",
        )

        self.assertGreaterEqual(relevant, 0.15)
        self.assertLess(unrelated, 0.15)

    def test_expanded_security_concept_is_visible_to_domain_check(self) -> None:
        expanded = normalize_research_query(
            "LLM Agent 在间接提示注入攻击下的工具调用安全"
        )

        consistency = domain_consistency_score(
            expanded,
            [{
                "evidence_id": "E1",
                "title": "Tool-use security",
                "text": (
                    "Indirect prompt injection attacks compromise tool use "
                    "security in LLM agents."
                ),
            }],
            TopicConsistencyConfig(mode="strict"),
        )

        self.assertTrue(consistency["passed"])
        self.assertIn("tool use", consistency["matched_topic_concepts"])


if __name__ == "__main__":
    unittest.main()

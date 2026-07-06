"""Tests for topic-aware external literature admission."""

from __future__ import annotations

import unittest

from app.agent.ai_scientific_agent import AIScientificAgent
from app.agent.services.literature_analysis_service import LiteratureAnalysisService
from app.schemas.evidence_item import EvidenceItem
from app.tools.external_relevance import (
    is_external_evidence_relevant_to_topic,
)
from app.tools.paper_analyzer import PaperAnalyzer


TOPIC = "LLM Agent 在间接提示注入攻击下的工具调用安全"
EXPANDED = (
    f"{TOPIC} indirect prompt injection prompt injection tool calling "
    "tool use security safety attack"
)


def arxiv_item(
    title: str,
    summary: str,
    score: float = 0.3,
) -> dict:
    return {
        "source_type": "arxiv",
        "title": title,
        "summary": summary,
        "relevance_score": score,
        "evidence_status": "retrieved",
    }


class ExternalRelevanceTests(unittest.TestCase):
    def test_known_unrelated_domains_are_rejected(self) -> None:
        cases = (
            (
                "Alignment Is All You Need For X-to-4D Generation",
                "A diffusion model generates animated 4D assets.",
            ),
            (
                "LLM Unlearning",
                "We remove memorized examples from language models.",
            ),
            (
                "Program-as-Weights",
                "Fuzzy functions encode programs in model weights.",
            ),
        )
        for title, summary in cases:
            with self.subTest(title=title):
                accepted, reason = is_external_evidence_relevant_to_topic(
                    arxiv_item(title, summary, score=0.4),
                    EXPANDED,
                    TOPIC,
                )
                self.assertFalse(accepted)
                self.assertIn("Rejected", reason)

    def test_persistent_state_ai_control_attack_is_accepted(self) -> None:
        accepted, reason = is_external_evidence_relevant_to_topic(
            arxiv_item(
                "Distributed Attacks in Persistent-State AI Control",
                (
                    "We study distributed attacks against AI control monitors "
                    "for persistent agents."
                ),
            ),
            EXPANDED,
            TOPIC,
        )

        self.assertTrue(accepted)
        self.assertIn("core phrase", reason)

    def test_score_without_topic_core_is_rejected(self) -> None:
        accepted, reason = is_external_evidence_relevant_to_topic(
            arxiv_item(
                "Efficient Language Model Compression",
                "A model compression method improves inference throughput.",
                score=0.167,
            ),
            EXPANDED,
            TOPIC,
        )

        self.assertFalse(accepted)
        self.assertIn("generic lexical overlap is insufficient", reason)

    def test_security_core_phrases_are_accepted(self) -> None:
        summaries = (
            "Prompt injection compromises a tool use policy.",
            "We map the attack surface of tool calling agents.",
            "A safety monitoring method detects malicious instruction attacks.",
        )
        for summary in summaries:
            with self.subTest(summary=summary):
                accepted, _ = is_external_evidence_relevant_to_topic(
                    arxiv_item("Agent security study", summary),
                    EXPANDED,
                    TOPIC,
                )
                self.assertTrue(accepted)

    def test_dynamic_filter_also_supports_non_security_topics(self) -> None:
        accepted, reason = is_external_evidence_relevant_to_topic(
            arxiv_item(
                "Retrieval Memory for LLM Agents",
                "LLM agent memory retrieves relevant interaction records.",
            ),
            "LLM agent memory",
            "LLM agent memory",
        )

        self.assertTrue(accepted)
        self.assertIn("dynamic topic concepts", reason)

    def test_rejected_external_text_cannot_pollute_research_gap(self) -> None:
        local = [{
            "title": "Tool Security",
            "excerpt": (
                "Method: Tool permission checks constrain agent actions. "
                "Limitations: Indirect prompt injection remains challenging."
            ),
            "text": (
                "Method: Tool permission checks constrain agent actions. "
                "Limitations: Indirect prompt injection remains challenging."
            ),
            "section": "Limitations",
        }]
        irrelevant = EvidenceItem(
            source_type="arxiv",
            title="X-to-4D Generation",
            summary="A diffusion model generates 4D video worlds.",
            relevance_score=0.4,
        )

        context, used, rejected = AIScientificAgent._build_literature_context(
            local,
            [irrelevant],
            EXPANDED,
            TOPIC,
        )
        analysis = LiteratureAnalysisService(PaperAnalyzer()).analyze(context)

        self.assertEqual(used, [])
        self.assertEqual(len(rejected), 1)
        for forbidden in (
            "X-to-4D",
            "4D generation",
            "unlearning",
            "Program-as-Weights",
        ):
            self.assertNotIn(forbidden, analysis["research_gap"])


if __name__ == "__main__":
    unittest.main()

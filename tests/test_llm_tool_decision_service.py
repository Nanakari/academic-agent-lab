"""Tests for LLM-driven scientific tool decisions."""

from __future__ import annotations

import unittest

from app.agent.services.llm_tool_decision_service import LLMToolDecisionService
from app.schema import ToolCall


class FakeToolDecisionLLM:
    def __init__(self, arguments: dict) -> None:
        self.arguments = arguments

    def ask_tool(self, messages, tools):
        return "", [
            ToolCall(
                id="call-1",
                name=LLMToolDecisionService.TOOL_NAME,
                arguments=self.arguments,
            )
        ]


class LLMToolDecisionServiceTests(unittest.TestCase):
    def test_uses_llm_function_call_to_select_tools(self) -> None:
        service = LLMToolDecisionService(
            llm=FakeToolDecisionLLM(
                {
                    "use_local_evidence_search": True,
                    "use_scientific_memory": False,
                    "use_external_search": True,
                    "external_sources": ["arxiv"],
                    "top_k": 3,
                    "reason": "Need recent literature metadata.",
                }
            ),
            enabled=True,
        )

        decision = service.decide(
            topic="LVLM hallucination mitigation",
            default_top_k=5,
            external_search_enabled=False,
            external_sources=["arxiv", "github"],
        )

        self.assertTrue(decision.llm_used)
        self.assertEqual(decision.mode, "llm_tool_decision")
        self.assertEqual(decision.top_k, 3)
        self.assertFalse(decision.use_scientific_memory)
        self.assertTrue(decision.use_external_search)
        self.assertEqual(decision.external_sources, ["arxiv"])
        self.assertEqual(decision.llm_call_count, 1)
        self.assertEqual(decision.llm_call_stages, ["tool_decision"])

    def test_external_search_request_is_blocked_when_no_sources_allowed(self) -> None:
        service = LLMToolDecisionService(
            llm=FakeToolDecisionLLM(
                {
                    "use_local_evidence_search": True,
                    "use_scientific_memory": True,
                    "use_external_search": True,
                    "external_sources": ["arxiv"],
                    "top_k": 5,
                    "reason": "Try external search.",
                }
            ),
            enabled=True,
        )

        decision = service.decide(
            topic="LVLM hallucination mitigation",
            default_top_k=5,
            external_search_enabled=False,
            external_sources=[],
        )

        self.assertFalse(decision.use_external_search)
        self.assertEqual(decision.external_sources, [])
        self.assertTrue(any("no external sources" in item for item in decision.warnings))

    def test_falls_back_without_enabled_llm(self) -> None:
        service = LLMToolDecisionService(llm=None, enabled=False)

        decision = service.decide(
            topic="LVLM hallucination mitigation",
            default_top_k=5,
            external_search_enabled=False,
            external_sources=["arxiv", "github"],
        )

        self.assertFalse(decision.llm_used)
        self.assertEqual(decision.mode, "deterministic_fallback")
        self.assertFalse(decision.use_external_search)
        self.assertEqual(decision.llm_call_count, 0)
        self.assertEqual(decision.llm_call_stages, [])


if __name__ == "__main__":
    unittest.main()

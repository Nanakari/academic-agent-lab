"""Tests for AI scientific agent construction-time validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.agent.ai_scientific_agent import AIScientificAgent


class AIScientificAgentConfigurationTests(unittest.TestCase):
    def test_rejects_unsupported_external_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "pubmed"):
                AIScientificAgent(
                    project_root=Path(directory),
                    external_search_sources=["arxiv", "pubmed"],
                )

    def test_rejects_enabled_external_search_without_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one supported source"):
                AIScientificAgent(
                    project_root=Path(directory),
                    external_search_enabled=True,
                    external_search_sources=[],
                )

    def test_accepts_comma_separated_external_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = AIScientificAgent(
                project_root=Path(directory),
                external_search_sources="arxiv, github",
            )

            self.assertEqual(agent.external_search_sources, ["arxiv", "github"])


if __name__ == "__main__":
    unittest.main()

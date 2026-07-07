"""Tests for shared scientific-agent CLI helpers."""

from __future__ import annotations

import argparse
import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.cli.scientific import (
    LLMConfigurationError,
    build_default_llm,
    parse_llm_stages,
    parse_external_sources,
    run_scientific_from_args,
)


class ScientificCliTests(unittest.TestCase):
    def test_parse_external_sources_normalizes_values(self) -> None:
        self.assertEqual(
            parse_external_sources(" arxiv,GitHub "),
            ["arxiv", "github"],
        )

    def test_parse_llm_stages_supports_all_none_and_subsets(self) -> None:
        self.assertEqual(parse_llm_stages("none"), [])
        self.assertIn("reflection", parse_llm_stages("all"))
        self.assertEqual(
            parse_llm_stages(" literature_analysis,Reflection "),
            ["literature_analysis", "reflection"],
        )

    def test_run_scientific_from_args_exits_cleanly_for_bad_source(self) -> None:
        args = argparse.Namespace(
            topic="LVLM hallucination mitigation",
            output_dir="outputs/test",
            papers_dir="tests/fixtures/papers",
            top_k=5,
            offline=True,
            llm_stages="all",
            use_external_search=False,
            no_external_search=False,
            external_sources="pubmed",
            external_max_results=5,
            external_force_refresh=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "pubmed"):
                run_scientific_from_args(args, project_root=Path(directory))

    def test_build_default_llm_reports_missing_dependency_cleanly(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "app.llm":
                raise ImportError("cannot import name 'genai' from 'google'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(
                LLMConfigurationError,
                "Install runtime dependencies",
            ):
                build_default_llm()


if __name__ == "__main__":
    unittest.main()

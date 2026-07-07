"""Tests for shared scientific-agent CLI helpers."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from app.cli.scientific import parse_external_sources, run_scientific_from_args


class ScientificCliTests(unittest.TestCase):
    def test_parse_external_sources_normalizes_values(self) -> None:
        self.assertEqual(
            parse_external_sources(" arxiv,GitHub "),
            ["arxiv", "github"],
        )

    def test_run_scientific_from_args_exits_cleanly_for_bad_source(self) -> None:
        args = argparse.Namespace(
            topic="LVLM hallucination mitigation",
            output_dir="outputs/test",
            papers_dir="tests/fixtures/papers",
            top_k=5,
            offline=True,
            use_external_search=False,
            no_external_search=False,
            external_sources="pubmed",
            external_max_results=5,
            external_force_refresh=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "pubmed"):
                run_scientific_from_args(args, project_root=Path(directory))


if __name__ == "__main__":
    unittest.main()

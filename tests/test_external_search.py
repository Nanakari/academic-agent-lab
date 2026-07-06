"""Regression tests for controlled external evidence retrieval."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.agent.ai_scientific_agent import AIScientificAgent
from app.agent.services.external_search import (
    ArxivSearchService,
    ExternalEvidenceService,
    GitHubSearchService,
)
from app.memory.scientific_memory import ScientificMemory
from app.schemas.evidence_item import EvidenceItem


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class ExternalProviderTests(unittest.TestCase):
    def test_arxiv_atom_entry_becomes_evidence_item(self) -> None:
        payload = b"""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2601.00001v1</id>
            <updated>2026-01-02T00:00:00Z</updated>
            <published>2026-01-01T00:00:00Z</published>
            <title>Verifier driven agents</title>
            <summary>A verifier checks scientific agent claims.</summary>
            <author><name>Ada Researcher</name></author>
            <category term="cs.AI"/>
            <link rel="alternate" href="https://arxiv.org/abs/2601.00001"/>
          </entry>
        </feed>"""
        service = ArxivSearchService(
            delay_seconds=0,
            opener=lambda request, timeout: FakeResponse(payload),
        )

        items = service.search("scientific agents")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_type, "arxiv")
        self.assertEqual(items[0].authors, ["Ada Researcher"])
        self.assertEqual(items[0].metadata["categories"], ["cs.AI"])
        self.assertEqual(items[0].query, "scientific agents")
        self.assertTrue(items[0].retrieved_at)

    def test_github_repository_is_not_scientific_validation(self) -> None:
        payload = json.dumps({"items": [{
            "full_name": "example/scientific-agent",
            "description": "Reference implementation",
            "html_url": "https://github.com/example/scientific-agent",
            "updated_at": "2026-01-01T00:00:00Z",
            "stargazers_count": 42,
            "forks_count": 3,
            "language": "Python",
            "license": {"spdx_id": "MIT"},
            "open_issues_count": 2,
            "archived": False,
        }]}).encode()
        service = GitHubSearchService(
            opener=lambda request, timeout: FakeResponse(payload)
        )

        items = service.search_repositories("scientific agent")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_type, "github_repo")
        self.assertEqual(
            items[0].metadata["evidence_role"],
            "implementation_availability",
        )
        self.assertFalse(items[0].metadata["scientific_validation"])

    def test_provider_error_returns_warning_instead_of_raising(self) -> None:
        def fail(request, timeout):
            raise TimeoutError("offline")

        service = ArxivSearchService(opener=fail)

        self.assertEqual(service.search("agent"), [])
        self.assertIn("offline", service.last_warnings[0])


class StubProvider:
    def __init__(self, items=None, error: Exception | None = None) -> None:
        self.items = items or []
        self.error = error
        self.calls = 0
        self.last_warnings = []

    def search(self, query, max_results):
        self.calls += 1
        if self.error:
            raise self.error
        return self.items

    def search_repositories(self, query, max_results):
        return self.search(query, max_results)


class ExternalEvidenceServiceTests(unittest.TestCase):
    def test_arxiv_failure_does_not_prevent_github_result(self) -> None:
        github_item = EvidenceItem(
            source_type="github_repo",
            title="example/repo",
            summary="Implementation",
            metadata={"evidence_role": "implementation_availability"},
        )
        with tempfile.TemporaryDirectory() as directory:
            service = ExternalEvidenceService(
                directory,
                arxiv_service=StubProvider(error=TimeoutError("arxiv offline")),
                github_service=StubProvider([github_item]),
            )

            result = service.retrieve("agent")

        self.assertEqual([item.title for item in result.evidence_items], ["example/repo"])
        self.assertEqual(result.sources_used, ["github"])
        self.assertTrue(any("arxiv offline" in item for item in result.warnings))

    def test_github_failure_does_not_prevent_arxiv_result(self) -> None:
        arxiv_item = EvidenceItem(
            source_type="arxiv",
            title="Scientific paper",
            summary="Abstract",
        )
        with tempfile.TemporaryDirectory() as directory:
            service = ExternalEvidenceService(
                directory,
                arxiv_service=StubProvider([arxiv_item]),
                github_service=StubProvider(error=TimeoutError("github offline")),
            )

            result = service.retrieve("agent")

        self.assertEqual([item.title for item in result.evidence_items], ["Scientific paper"])
        self.assertEqual(result.sources_used, ["arxiv"])
        self.assertTrue(any("github offline" in item for item in result.warnings))

    def test_cache_hit_skips_second_provider_call(self) -> None:
        item = EvidenceItem(
            source_type="arxiv",
            title="Cached paper",
            summary="Abstract",
        )
        provider = StubProvider([item])
        with tempfile.TemporaryDirectory() as directory:
            first = ExternalEvidenceService(
                directory,
                arxiv_service=provider,
                github_service=StubProvider(),
            )
            first.retrieve("cache query", use_github=False)
            second = ExternalEvidenceService(
                directory,
                arxiv_service=provider,
                github_service=StubProvider(),
            )
            result = second.retrieve("cache query", use_github=False)

        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.cache_used)
        self.assertEqual(result.evidence_items[0].title, "Cached paper")


class AgentExternalIntegrationTests(unittest.TestCase):
    def test_disabled_status_preserves_offline_pipeline_and_report_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers = root / "data" / "papers"
            papers.mkdir(parents=True)
            (papers / "paper.txt").write_text(
                "Method: Agent memory retrieves relevant records. "
                "Limitations: Retrieval errors accumulate under domain shift.",
                encoding="utf-8",
            )
            output = root / "outputs"
            agent = AIScientificAgent(
                project_root=root,
                output_dir=output,
                memory=ScientificMemory(root / "memory"),
            )

            result = agent.run("agent memory retrieval")

            self.assertEqual(result["external_search_status"]["enabled"], False)
            self.assertEqual(result["external_evidence"], [])
            saved = json.loads(
                (output / "result.json").read_text(encoding="utf-8")
            )
            self.assertIn("external_search_status", saved)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("## External Evidence Retrieved", report)
            self.assertIn("## External Source Warnings", report)
            self.assertTrue(
                result["experiment_blueprint"]["human_approval_required"]
            )
            self.assertFalse(result["experiment_blueprint"]["execution_allowed"])

    def test_all_network_failures_do_not_abort_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers = root / "data" / "papers"
            papers.mkdir(parents=True)
            (papers / "paper.txt").write_text(
                "Method: Verifiers inspect agent claims. "
                "Limitations: Errors remain under distribution shift.",
                encoding="utf-8",
            )
            external = ExternalEvidenceService(
                root / "cache",
                arxiv_service=StubProvider(error=TimeoutError("arxiv offline")),
                github_service=StubProvider(error=TimeoutError("github offline")),
            )
            agent = AIScientificAgent(
                project_root=root,
                output_dir=root / "outputs",
                memory=ScientificMemory(root / "memory"),
                external_search_enabled=True,
                external_evidence_service=external,
            )

            result = agent.run("scientific agent verifier")

        self.assertEqual(result["external_evidence"], [])
        self.assertEqual(len(result["external_retrieval_warnings"]), 2)
        self.assertIn("verification_passed", result)

    def test_external_abstract_does_not_turn_failed_verifier_into_pass(self) -> None:
        arxiv_item = EvidenceItem(
            source_type="arxiv",
            title="Strong external result",
            summary=(
                "Method: A scientific verifier reduces unsupported claims. "
                "Limitations: Evaluation remains narrow."
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = ExternalEvidenceService(
                root / "cache",
                arxiv_service=StubProvider([arxiv_item]),
                github_service=StubProvider(),
            )
            agent = AIScientificAgent(
                project_root=root,
                output_dir=root / "outputs",
                papers_dir=root / "empty-papers",
                memory=ScientificMemory(root / "memory"),
                external_search_enabled=True,
                external_search_sources=["arxiv"],
                external_evidence_service=external,
            )

            result = agent.run("scientific verifier unsupported claims")

        self.assertEqual(len(result["external_evidence"]), 1)
        self.assertFalse(result["verification"]["evidence"]["passed"])
        self.assertFalse(result["verification_passed"])


if __name__ == "__main__":
    unittest.main()

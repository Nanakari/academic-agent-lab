"""Tests for the offline AI Scientific Agent MVP."""

import json
import tempfile
import unittest
from pathlib import Path

from app.agent.ai_scientific_agent import AIScientificAgent
from app.agent.services import (
    EvidenceService,
    PersistenceService,
    VerificationPipeline,
)
from app.memory.scientific_memory import ScientificMemory
from app.planner.research_planner import ResearchPlanner
from app.schemas.evidence import EvidenceChunk, support_level_for_score
from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.research_idea import ResearchIdea
from app.schemas.scientific_task import ScientificTaskType
from app.tools.paper_corpus import PaperCorpusIndexer
from app.tools.paper_analyzer import PaperAnalyzer
from app.verifier.evidence_verifier import EvidenceVerifier
from app.verifier.experiment_verifier import ExperimentVerifier
from app.verifier.novelty_verifier import NoveltyVerifier
from app.verifier.reproducibility_verifier import ReproducibilityVerifier


FIXTURE_PAPERS = Path(__file__).parent / "fixtures" / "papers"


def complete_experiment_plan() -> ExperimentPlan:
    return ExperimentPlan(
        idea_title="Evidence-aware validation",
        method="Route uncertain examples through a verifier.",
        datasets=["POPE"],
        baselines=["Base LVLM"],
        metrics=["Hallucination rate"],
        ablation=["Remove the verifier"],
        expected_results=["Lower hallucination rate"],
        risks=["Additional latency"],
        implementation_notes=["Pin versions and seeds"],
    )


class ResearchPlannerTests(unittest.TestCase):
    def test_topic_only_defaults_to_research_proposal(self) -> None:
        planner = ResearchPlanner()
        task_type = planner.classify_task("LVLM hallucination mitigation")
        plan = planner.create_plan("LVLM hallucination mitigation", task_type)

        self.assertEqual(task_type, ScientificTaskType.RESEARCH_PROPOSAL)
        self.assertGreaterEqual(len(plan.steps), 5)
        self.assertIn("experiment_designer", plan.required_tools)

    def test_explicit_experiment_request_is_classified(self) -> None:
        planner = ResearchPlanner()
        self.assertEqual(
            planner.classify_task("Design an experiment for agent memory"),
            ScientificTaskType.EXPERIMENT_DESIGN,
        )


class ScientificMemoryTests(unittest.TestCase):
    def test_jsonl_round_trip_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = ScientificMemory(directory)
            memory.save_idea({"title": "Adaptive agent memory", "method": "routing"})

            self.assertEqual(len(memory.load_recent_ideas()), 1)
            self.assertEqual(memory.search_memory("adaptive")[0]["memory_type"], "idea")

    def test_duplicate_idea_is_not_appended(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = ScientificMemory(directory)
            idea = {
                "topic": "agent memory",
                "title": "Adaptive retrieval",
                "method": "Route records by relevance.",
            }

            self.assertTrue(memory.save_idea(idea))
            self.assertFalse(memory.save_idea(idea))
            self.assertEqual(len(memory.load_recent_ideas()), 1)

    def test_malformed_jsonl_warns_and_preserves_valid_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = ScientificMemory(directory)
            ideas_path = Path(directory) / "ideas.jsonl"
            ideas_path.write_text(
                '{"title": "valid idea", "method": "routing"}\n'
                '{"title": broken}\n',
                encoding="utf-8",
            )

            records = memory.load_recent_ideas()

            self.assertEqual([record["title"] for record in records], ["valid idea"])
            self.assertTrue(memory.warnings)
            self.assertIn("line 2", memory.warnings[0])


class PaperCorpusTests(unittest.TestCase):
    def test_fixture_papers_produce_ranked_evidence(self) -> None:
        corpus = PaperCorpusIndexer(FIXTURE_PAPERS)

        documents = corpus.scan_papers()
        evidence = corpus.search("LVLM hallucination mitigation", top_k=2)

        self.assertEqual(len(documents), 3)
        self.assertTrue(evidence)
        self.assertTrue(
            any(item.title == "Grounded LVLM Hallucination Mitigation" for item in evidence)
        )
        self.assertGreater(evidence[0].score, 0.12)
        self.assertTrue(evidence[0].paper_id)
        self.assertTrue(evidence[0].chunk_id)

    def test_markdown_section_and_matched_keywords_are_preserved(self) -> None:
        corpus = PaperCorpusIndexer(FIXTURE_PAPERS)

        evidence = corpus.search(
            "LVLM hallucination visual evidence",
            top_k=3,
        )

        method_evidence = next(
            item for item in evidence
            if item.title == "Structured Evidence for Reliable LVLMs"
        )
        self.assertEqual(method_evidence.section, "Method")
        self.assertIsNone(method_evidence.page)
        self.assertTrue(method_evidence.matched_keywords)
        self.assertIn("hallucination", method_evidence.matched_keywords)
        self.assertTrue(method_evidence.supporting_claim)

    def test_split_document_accepts_page_metadata(self) -> None:
        corpus = PaperCorpusIndexer(FIXTURE_PAPERS)

        chunks = corpus.split_document(
            "A visual evidence verifier reduces hallucination in LVLM outputs.",
            page=3,
            section="Method",
        )

        self.assertEqual(chunks[0].page, 3)
        self.assertEqual(chunks[0].section, "Method")

    def test_title_only_match_is_not_strong_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory) / "title_only.md"
            paper.write_text(
                "# LVLM Hallucination Visual Evidence\n\n"
                "## Method\n\n"
                "We study deterministic sorting algorithms and runtime complexity.",
                encoding="utf-8",
            )

            evidence = PaperCorpusIndexer(directory).search(
                "LVLM hallucination visual evidence",
                top_k=1,
            )

            self.assertTrue(evidence)
            self.assertEqual(evidence[0].matched_keywords, [])
            self.assertLessEqual(evidence[0].score, 0.1)
            self.assertEqual(evidence[0].support_level, "insufficient")

    def test_index_cache_refreshes_only_when_needed(self) -> None:
        class CountingCorpus(PaperCorpusIndexer):
            def __init__(self, papers_dir) -> None:
                super().__init__(papers_dir)
                self.scan_count = 0

            def scan_papers(self, papers_dir=None):
                self.scan_count += 1
                return super().scan_papers(papers_dir)

        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory)
            (corpus_path / "one.txt").write_text(
                "LVLM hallucination mitigation uses visual evidence.",
                encoding="utf-8",
            )
            corpus = CountingCorpus(corpus_path)

            corpus.search("LVLM hallucination")
            corpus.search("visual evidence")
            self.assertEqual(corpus.scan_count, 1)

            (corpus_path / "two.md").write_text(
                "# Method\nA verifier checks grounded answers.",
                encoding="utf-8",
            )
            corpus.search("grounded verifier")
            self.assertEqual(corpus.scan_count, 2)

            corpus.search("grounded verifier", force_refresh=True)
            self.assertEqual(corpus.scan_count, 3)


class EvidenceVerifierTests(unittest.TestCase):
    def test_empty_corpus_fails_honestly(self) -> None:
        idea = ResearchIdea(
            title="Unsupported LVLM idea",
            hypothesis="A new module may reduce hallucination.",
            motivation="Exploratory only.",
            method="Add a verification module.",
        )

        result = EvidenceVerifier().verify(idea, [], ideas=[idea])

        self.assertFalse(result.passed)
        self.assertEqual(result.support_level, "insufficient")
        self.assertTrue(any("no evidence found" in issue for issue in result.issues))
        self.assertTrue(result.unsupported_claims)

    def test_support_level_thresholds(self) -> None:
        cases = [
            (0.60, "strong"),
            (0.35, "moderate"),
            (0.15, "weak"),
            (0.14, "insufficient"),
        ]

        for score, expected in cases:
            evidence = EvidenceChunk(
                paper_id="paper",
                title="Test paper",
                source_path="paper.txt",
                file_type="txt",
                chunk_id="C1",
                text="test evidence",
                score=score,
                support_level=support_level_for_score(score),
            )
            self.assertEqual(evidence.support_level, expected)

    def test_low_score_evidence_cannot_be_strong_support(self) -> None:
        idea = ResearchIdea(
            title="LVLM hallucination verifier",
            hypothesis="Visual evidence reduces LVLM hallucination.",
            motivation="Ground the answer in image regions.",
            method="Verify answers against visual evidence.",
            evidence_refs=["E1"],
        )
        evidence = [{
            "evidence_id": "E1",
            "paper_id": "paper",
            "title": "Relevant title",
            "source_path": "paper.txt",
            "file_type": "txt",
            "chunk_id": "C1",
            "text": (
                "LVLM hallucination verification uses visual evidence to ground "
                "answers in image regions."
            ),
            "score": 0.1,
        }]

        result = EvidenceVerifier().verify(idea, evidence, ideas=[idea])

        self.assertFalse(result.passed)
        self.assertEqual(result.support_level, "insufficient")
        self.assertTrue(result.unsupported_claims)


class VerifierCoverageTests(unittest.TestCase):
    def test_novelty_verifier_rejects_similar_and_accepts_new_idea(self) -> None:
        existing = ResearchIdea(
            title="Adaptive LVLM verification",
            hypothesis="Use uncertainty for routing.",
            motivation="Reduce hallucination.",
            method="Route uncertain answers through visual verification.",
        )
        history = [existing.to_dict()]

        duplicate = NoveltyVerifier().verify(existing, history)
        new_idea = ResearchIdea(
            title="Causal memory compression for agents",
            hypothesis="Causal summaries improve long-horizon recall.",
            motivation="Reduce irrelevant memory retrieval.",
            method="Learn a causal graph over interaction summaries.",
        )
        novel = NoveltyVerifier().verify(new_idea, history)

        self.assertFalse(duplicate.passed)
        self.assertTrue(novel.passed)

    def test_experiment_verifier_checks_all_required_fields(self) -> None:
        verifier = ExperimentVerifier()
        complete = complete_experiment_plan()

        self.assertTrue(verifier.verify(complete).passed)
        for field in verifier.REQUIRED_FIELDS:
            values = complete.to_dict()
            values[field] = []
            with self.subTest(field=field):
                self.assertFalse(
                    verifier.verify(ExperimentPlan(**values)).passed
                )

    def test_reproducibility_verifier_rejects_missing_and_accepts_complete(self) -> None:
        missing = ExperimentPlan(
            idea_title="Incomplete plan",
            method="",
        )

        self.assertFalse(ReproducibilityVerifier().verify(missing).passed)
        self.assertTrue(
            ReproducibilityVerifier().verify(complete_experiment_plan()).passed
        )


class AIScientificAgentTests(unittest.TestCase):
    def test_default_memory_follows_custom_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = AIScientificAgent(project_root=root)

            self.assertEqual(
                agent.scientific_memory.memory_dir,
                root.resolve() / "data" / "research_memory",
            )

    def test_end_to_end_local_run_writes_both_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers_dir = root / "data" / "papers"
            papers_dir.mkdir(parents=True)
            (papers_dir / "memory_paper.txt").write_text(
                "Abstract: LLM agent memory improves long-horizon task continuity. "
                "Method: Retrieval memory selects relevant interaction records. "
                "Experiments: We compare no-memory and sliding-window baselines. "
                "Limitations: Retrieval errors accumulate under domain shift.",
                encoding="utf-8",
            )
            output_dir = root / "outputs"
            memory = ScientificMemory(root / "research_memory")
            agent = AIScientificAgent(
                project_root=root,
                output_dir=output_dir,
                memory=memory,
            )

            result = agent.run("LLM Agent Memory")

            self.assertEqual(result["task_type"], "research_proposal")
            self.assertEqual(len(result["candidate_ideas"]), 3)
            self.assertTrue(result["experiment_plan"]["datasets"])
            self.assertTrue(result["verification_passed"])
            self.assertEqual(result["evidence_status"], "sufficient")
            self.assertIn("evidence_used", result)
            self.assertIn("evidence_gaps", result)
            self.assertIn("unsupported_claims", result)
            self.assertTrue((output_dir / "result.json").exists())
            self.assertTrue((output_dir / "report.md").exists())
            saved = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_idea"]["title"], result["selected_idea"]["title"])
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Evidence Used", report)
            self.assertIn("## Evidence Gaps", report)
            self.assertIn("## Unsupported Claims", report)
            self.assertIn("Support Level", report)
            self.assertIn("Matched Keywords", report)

    def test_unreadable_pdf_does_not_abort_local_evidence_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers_dir = root / "data" / "papers"
            papers_dir.mkdir(parents=True)
            (papers_dir / "broken.pdf").write_bytes(b"not a valid PDF")
            (papers_dir / "valid.txt").write_text(
                "Method: Agent memory retrieves relevant historical interactions.",
                encoding="utf-8",
            )
            agent = AIScientificAgent(project_root=root)

            evidence = agent._retrieve_evidence("agent memory")

            self.assertTrue(evidence)
            self.assertTrue(agent.paper_corpus.warnings)
            self.assertTrue(all("broken.pdf" not in item["source"] for item in evidence))


class AgentServiceTests(unittest.TestCase):
    def test_evidence_service_assigns_evidence_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = ScientificMemory(Path(directory) / "memory")
            service = EvidenceService(
                Path(directory),
                PaperCorpusIndexer(FIXTURE_PAPERS),
                memory,
            )

            evidence = service.retrieve("LVLM hallucination", top_k=2)

            self.assertTrue(evidence)
            self.assertEqual(evidence[0]["evidence_id"], "E1")

    def test_verification_pipeline_returns_all_verifier_results(self) -> None:
        idea = ResearchIdea(
            title="Evidence-aware validation",
            hypothesis="Visual evidence may reduce hallucination.",
            motivation="Ground answers in retrieved image evidence.",
            method="Route uncertain answers through visual verification.",
            evidence_refs=["E1"],
        )
        evidence_item = PaperCorpusIndexer(FIXTURE_PAPERS).search(
            "LVLM hallucination visual evidence",
            top_k=1,
        )[0].to_dict()
        evidence_item["evidence_id"] = "E1"

        results = VerificationPipeline().verify(
            idea,
            complete_experiment_plan(),
            [evidence_item],
            {
                "research_gap": "Hallucination remains under visual uncertainty.",
                "existing_methods": ["Visual evidence verification."],
            },
            [],
            [idea],
        )

        self.assertEqual(
            set(results),
            {"evidence", "novelty", "experiment", "reproducibility"},
        )

    def test_persistence_service_uses_memory_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = ScientificMemory(directory)
            service = PersistenceService(memory, PaperAnalyzer())
            result = {
                "topic": "LVLM hallucination",
                "evidence_context": [{
                    "kind": "local_paper",
                    "source": "paper.txt",
                    "evidence_id": "E1",
                    "page": None,
                    "section": "Method",
                    "file_type": "txt",
                    "chunk_id": "C1",
                    "excerpt": "Visual evidence verifies an LVLM response.",
                }],
                "selected_idea": {
                    "title": "Evidence-aware validation",
                    "method": "Verify outputs against visual evidence.",
                },
                "experiment_plan": complete_experiment_plan().to_dict(),
                "verification_passed": True,
                "verification": {},
            }

            service.save(result)
            service.save(result)

            self.assertEqual(len(memory.load_recent_ideas()), 1)
            paper_notes = [
                item for item in memory.search_memory("paper.txt")
                if item["memory_type"] == "paper_note"
            ]
            self.assertEqual(len(paper_notes), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for the offline AI Scientific Agent MVP."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.agent.ai_scientific_agent import AIScientificAgent
from app.agent.services import (
    AgentDecisionPolicy,
    EvidenceService,
    FeasibilityService,
    LiteratureAnalysisService,
    PersistenceService,
    ResearchDirectionService,
    VerificationPipeline,
)
from app.memory.scientific_memory import ScientificMemory
from app.planner.research_planner import ResearchPlanner
from app.schemas.evidence import EvidenceChunk, support_level_for_score
from app.schemas.agent_trace import AgentTraceEntry
from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.feasibility_assessment import FeasibilityAssessment
from app.schemas.research_direction import ResearchDirection
from app.schemas.research_idea import ResearchIdea
from app.schemas.scientific_task import ScientificTaskType
from app.tools.paper_corpus import PaperCorpusIndexer
from app.tools.paper_analyzer import PaperAnalyzer
from app.verifier.evidence_verifier import EvidenceVerifier
from app.verifier.experiment_verifier import ExperimentVerifier
from app.verifier.novelty_verifier import NoveltyVerifier
from app.verifier.reproducibility_verifier import ReproducibilityVerifier
from app.verifier.claim_filter import is_verifiable_claim
from app.verifier.topic_consistency import (
    TopicConsistencyConfig,
    domain_consistency_score,
    evidence_matches_concept,
)


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

    def test_strict_domain_rejects_scattered_negative_topic_terms(self) -> None:
        idea = ResearchIdea(
            title="Graph traffic forecasting",
            hypothesis="A graph model predicts traffic.",
            motivation="Improve forecasts.",
            method="Use graph message passing.",
            evidence_refs=["E1"],
        )
        evidence = [{
            "evidence_id": "E1",
            "title": "Autonomous Driving Agents",
            "text": (
                "Autonomous driving agents operate in networked traffic "
                "environments and predict actions."
            ),
            "score": 0.4,
        }]

        result = EvidenceVerifier(strict_domain=True).verify(
            idea,
            evidence,
            ideas=[idea],
            topic="graph neural network traffic prediction",
        )

        self.assertFalse(result.passed)
        self.assertTrue(
            any(
                "topic-domain consistency check failed" in issue
                for issue in result.issues
            )
        )

    def test_domain_modes_and_legacy_boolean_compatibility(self) -> None:
        idea = ResearchIdea(
            title="Autonomous agent action prediction",
            hypothesis="Agents predict actions in traffic.",
            motivation="Autonomous agents need reliable action prediction.",
            method="Predict actions from networked traffic observations.",
            evidence_refs=["E1"],
        )
        evidence = [{
            "evidence_id": "E1",
            "title": "Autonomous Driving Agents",
            "text": (
                "Autonomous driving agents operate in networked traffic "
                "environments and predict actions reliably."
            ),
            "score": 0.4,
        }]
        topic = "graph neural network traffic prediction"

        off = EvidenceVerifier(domain_mode="off").verify(
            idea, evidence, ideas=[idea], topic=topic
        )
        warning = EvidenceVerifier(domain_mode="warning").verify(
            idea, evidence, ideas=[idea], topic=topic
        )
        strict = EvidenceVerifier(domain_mode="strict").verify(
            idea, evidence, ideas=[idea], topic=topic
        )

        self.assertTrue(off.passed)
        self.assertEqual(off.domain_consistency, {})
        self.assertTrue(warning.passed)
        self.assertTrue(warning.warnings)
        self.assertFalse(warning.domain_consistency["passed"])
        self.assertFalse(strict.passed)
        self.assertTrue(strict.issues)
        self.assertEqual(
            EvidenceVerifier(strict_domain=True).domain_mode,
            "strict",
        )
        self.assertEqual(
            EvidenceVerifier(
                strict_domain=False,
                domain_mode="strict",
            ).domain_mode,
            "off",
        )
        with self.assertRaisesRegex(ValueError, "allowed values"):
            EvidenceVerifier(domain_mode="invalid")

    def test_overclaim_and_key_claim_citation_are_preserved(self) -> None:
        idea = ResearchIdea(
            title="Visual evidence verifier",
            hypothesis="This always solves LVLM hallucination.",
            motivation="Visual evidence grounds generated answers.",
            method="Verify each answer against image evidence.",
            evidence_refs=["E1"],
        )
        evidence = [{
            "evidence_id": "E1",
            "paper_id": "paper",
            "title": "Grounded LVLM",
            "source_path": "paper.txt",
            "file_type": "txt",
            "page": None,
            "section": "Method",
            "chunk_id": "C1",
            "text": (
                "Visual evidence verification grounds LVLM generated answers "
                "and reduces measured hallucination."
            ),
            "score": 0.8,
        }]

        result = EvidenceVerifier().verify(
            idea,
            evidence,
            claims=["Visual evidence verification grounds LVLM answers."],
            ideas=[idea],
        )

        self.assertFalse(result.passed)
        self.assertTrue(
            any("Overclaiming language detected" in issue for issue in result.issues)
        )
        self.assertTrue(
            any(
                citation["claim"].startswith("key claim")
                for citation in result.supported_claims
            )
        )

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
            self.assertIn("agent_trace", result)
            self.assertGreaterEqual(len(result["agent_trace"]), 4)
            self.assertIn("research_directions", result)
            self.assertIn("selected_direction", result)
            self.assertGreaterEqual(len(result["research_directions"]), 1)
            self.assertEqual(
                result["selected_direction"]["source_idea_title"],
                result["selected_idea"]["title"],
            )
            self.assertEqual(
                result["selected_direction"]["assessment_status"],
                "verifier_assessed_selected",
            )
            self.assertEqual(
                result["selected_direction"]["source_idea_index"],
                0,
            )
            self.assertIn("feasibility_assessment", result)
            self.assertIn(
                "recommendation",
                result["feasibility_assessment"],
            )
            self.assertIn(
                "minimum_viable_experiment",
                result["feasibility_assessment"],
            )
            decisions = [entry["decision"] for entry in result["agent_trace"]]
            self.assertTrue(
                "skip_revision" in decisions
                or "trigger_bounded_revision" in decisions
            )
            for field in ("observation", "decision", "reason", "action"):
                self.assertIn(field, result["agent_trace"][0])
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
            self.assertIn("## Agent Decision Trace", report)
            self.assertIn("Observation:", report)
            self.assertIn("Decision:", report)
            self.assertIn("Action:", report)
            self.assertIn("Reason:", report)
            self.assertIn("## Candidate Research Directions", report)
            self.assertIn("## Selected Research Direction", report)
            self.assertIn("Evidence Support Level", report)
            self.assertIn("Recommended Priority", report)
            self.assertIn("## Feasibility Assessment", report)
            self.assertIn("Planning Readiness Score", report)
            self.assertIn("Recommendation", report)
            self.assertIn("Minimum Viable Experiment", report)

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

    def test_revision_trigger_reason_survives_when_reverification_passes(self) -> None:
        verification_template = {
            "score": 1.0,
            "issues": [],
            "suggestions": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "evidence_used": [],
            "support_level": "strong",
            "domain_consistency": {},
            "warnings": [],
        }
        initial = {
            name: {**verification_template, "passed": name != "novelty"}
            for name in ("evidence", "novelty", "experiment", "reproducibility")
        }
        initial["novelty"]["issues"] = ["Idea overlaps with memory."]
        final = {
            name: {**verification_template, "passed": True}
            for name in ("evidence", "novelty", "experiment", "reproducibility")
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = AIScientificAgent(
                project_root=root,
                output_dir=root / "outputs",
                papers_dir=FIXTURE_PAPERS,
                memory=ScientificMemory(root / "memory"),
            )
            agent.verification_pipeline.verify = Mock(
                side_effect=[initial, final]
            )

            result = agent.run("LVLM hallucination mitigation")

        self.assertTrue(result["revision_performed"])
        self.assertTrue(result["verification_passed"])
        self.assertEqual(
            result["agent_trace"][1]["decision"],
            "trigger_bounded_revision",
        )
        self.assertIn("novelty verifier failed", result["agent_trace"][1]["reason"])
        self.assertIn("novelty", result["agent_trace"][1]["observation"])
        self.assertEqual(
            result["agent_trace"][2]["decision"],
            "accept_current_plan",
        )


class AgentDecisionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = AgentDecisionPolicy()

    def test_trace_entry_serializes_all_stable_fields(self) -> None:
        entry = AgentTraceEntry(
            step=1,
            observation="Local evidence was inspected.",
            decision="continue_grounded_planning",
            reason="A concrete limitation was found.",
            action="generate_candidate_ideas",
            result="Planning continued.",
        )

        self.assertEqual(
            set(entry.to_dict()),
            {"step", "observation", "decision", "reason", "action", "result"},
        )

    def test_no_evidence_marks_insufficient_evidence(self) -> None:
        entry = self.policy.decide_after_evidence(
            step=1,
            topic="agent memory",
            evidence_context=[],
            literature_analysis={},
        )

        self.assertEqual(entry.decision, "mark_insufficient_evidence")

    def test_missing_research_gap_downgrades_confidence(self) -> None:
        entry = self.policy.decide_after_evidence(
            step=1,
            topic="agent memory",
            evidence_context=[{"evidence_id": "E1"}],
            literature_analysis={"research_gap_status": "insufficient_evidence"},
        )

        self.assertEqual(entry.decision, "downgrade_gap_confidence")

    def test_all_verifiers_pass_accepts_current_plan(self) -> None:
        verification = {
            name: {"passed": True, "issues": []}
            for name in ("evidence", "novelty", "experiment", "reproducibility")
        }

        entry = self.policy.decide_after_verification(
            step=2,
            verification=verification,
            revision_performed=False,
        )

        self.assertEqual(entry.decision, "accept_current_plan")

    def test_before_revision_skips_when_all_verifiers_pass(self) -> None:
        verification = {
            name: {"passed": True, "issues": []}
            for name in ("evidence", "novelty", "experiment", "reproducibility")
        }

        entry = self.policy.decide_before_revision(
            step=2,
            verification=verification,
        )

        self.assertEqual(entry.decision, "skip_revision")
        self.assertEqual(entry.action, "continue_to_final_assessment")

    def test_before_revision_records_all_failures_and_prioritizes_evidence(
        self,
    ) -> None:
        verification = {
            "evidence": {"passed": False, "issues": ["No grounded support."]},
            "experiment": {"passed": False, "issues": ["Missing baselines."]},
            "novelty": {"passed": True, "issues": []},
            "reproducibility": {"passed": True, "issues": []},
        }

        entry = self.policy.decide_before_revision(
            step=2,
            verification=verification,
        )

        self.assertEqual(entry.decision, "trigger_bounded_revision")
        self.assertEqual(entry.action, "perform_bounded_revision")
        self.assertIn("evidence verifier failed", entry.reason)
        self.assertIn("evidence, experiment", entry.observation)
        self.assertIn("Primary trigger: evidence", entry.observation)

    def test_before_revision_targets_experiment_failure(self) -> None:
        verification = {
            "evidence": {"passed": True, "issues": []},
            "experiment": {
                "passed": False,
                "issues": ["Missing baselines and metrics."],
            },
            "novelty": {"passed": True, "issues": []},
            "reproducibility": {"passed": True, "issues": []},
        }

        entry = self.policy.decide_before_revision(
            step=2,
            verification=verification,
        )

        self.assertEqual(entry.decision, "trigger_bounded_revision")
        self.assertIn("baseline, metric", entry.reason)
        self.assertIn("experiment-plan completeness", entry.result)

    def test_before_revision_targets_novelty_failure(self) -> None:
        verification = {
            "evidence": {"passed": True, "issues": []},
            "experiment": {"passed": True, "issues": []},
            "novelty": {
                "passed": False,
                "issues": ["Idea overlaps with memory."],
            },
            "reproducibility": {"passed": True, "issues": []},
        }

        entry = self.policy.decide_before_revision(
            step=2,
            verification=verification,
        )

        self.assertEqual(entry.decision, "trigger_bounded_revision")
        self.assertIn("novelty verifier failed", entry.reason)
        self.assertIn("idea differentiation", entry.result)

    def test_evidence_failure_is_explained_and_preserved(self) -> None:
        verification = {
            "evidence": {"passed": False, "issues": ["No local support."]},
            "novelty": {"passed": True, "issues": []},
            "experiment": {"passed": True, "issues": []},
            "reproducibility": {"passed": True, "issues": []},
        }

        entry = self.policy.decide_after_verification(
            step=2,
            verification=verification,
            revision_performed=True,
        )

        self.assertEqual(entry.decision, "revise_or_mark_evidence_gap")
        self.assertIn("evidence verifier failed", entry.reason)
        self.assertIn("failure is preserved", entry.result)

    def test_insufficient_final_evidence_uses_cautious_report(self) -> None:
        entry = self.policy.decide_before_report(
            step=3,
            evidence_status="evidence_insufficient",
            verification_passed=False,
            selected_idea={"title": "Exploratory agent memory"},
        )

        self.assertEqual(
            entry.decision,
            "report_as_exploratory_or_insufficient",
        )


class ResearchDirectionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ResearchDirectionService()
        self.ideas = [
            ResearchIdea(
                title="Selected evidence-aware verifier",
                hypothesis="Verification may reduce unsupported outputs.",
                motivation="Unsupported outputs remain under uncertainty.",
                method="Route uncertain outputs through a verifier.",
                evidence_refs=["E1", "E2"],
                rank_score=0.9,
            ),
            ResearchIdea(
                title="Unselected benchmark direction",
                hypothesis="Counterfactual pairs may reveal failures.",
                motivation="Static scores can hide sensitivity.",
                method="Build controlled counterfactual pairs.",
                evidence_refs=["E1"],
                rank_score=0.8,
            ),
        ]
        self.evidence = [
            {"evidence_id": "E1"},
            {"evidence_id": "E2"},
        ]
        self.analysis = {
            "research_gap": "Verification under uncertainty remains limited.",
            "research_gap_status": "evidence_supported",
        }

    @staticmethod
    def verification(
        *,
        evidence_passed: bool = True,
        novelty_passed: bool = True,
        experiment_passed: bool = True,
        reproducibility_passed: bool = True,
    ) -> dict:
        return {
            "evidence": {
                "passed": evidence_passed,
                "support_level": "strong",
                "issues": [],
            },
            "novelty": {"passed": novelty_passed, "issues": []},
            "experiment": {"passed": experiment_passed, "issues": []},
            "reproducibility": {
                "passed": reproducibility_passed,
                "issues": [],
            },
        }

    def test_research_direction_serializes(self) -> None:
        direction = ResearchDirection(
            title="Evidence-aware LVLM verification",
            source_idea_title="Evidence-aware verifier idea",
            source_idea_index=0,
            target_gap="Hallucination remains under visual uncertainty.",
            core_problem="How to reduce hallucination without retraining.",
            hypothesis="Evidence verification may reduce hallucination.",
            method_sketch="Route uncertain answers through a verifier.",
            supporting_evidence=["E1"],
            evidence_support_level="moderate",
            novelty_risk="low",
            feasibility_risk="medium",
            recommended_priority="medium",
            assessment_status="verifier_assessed_selected",
            next_steps=["Define the minimum viable experiment."],
        )

        data = direction.to_dict()

        self.assertEqual(data["title"], "Evidence-aware LVLM verification")
        self.assertEqual(data["supporting_evidence"], ["E1"])
        self.assertEqual(data["source_idea_title"], "Evidence-aware verifier idea")
        self.assertEqual(data["source_idea_index"], 0)

    def test_selected_direction_matches_selected_idea_only(self) -> None:
        directions, selected = self.service.plan(
            topic="LVLM reliability",
            literature_analysis=self.analysis,
            candidate_ideas=self.ideas,
            selected_idea=self.ideas[0],
            evidence_context=self.evidence,
            verification=self.verification(),
        )

        self.assertEqual(len(directions), 2)
        self.assertEqual(selected.source_idea_title, self.ideas[0].title)
        self.assertEqual(selected.source_idea_index, 0)
        self.assertEqual(
            selected.assessment_status,
            "verifier_assessed_selected",
        )
        self.assertEqual(selected.novelty_risk, "low")
        self.assertEqual(
            directions[1].assessment_status,
            "heuristic_unverified",
        )
        self.assertEqual(directions[1].novelty_risk, "unknown")
        self.assertEqual(directions[1].feasibility_risk, "unknown")

    def test_failed_evidence_forces_exploratory_selected_direction(self) -> None:
        _, selected = self.service.plan(
            topic="LVLM reliability",
            literature_analysis=self.analysis,
            candidate_ideas=self.ideas,
            selected_idea=self.ideas[0],
            evidence_context=self.evidence,
            verification=self.verification(evidence_passed=False),
        )

        self.assertEqual(selected.evidence_support_level, "insufficient")
        self.assertEqual(selected.recommended_priority, "exploratory")

    def test_failed_novelty_cannot_receive_high_priority(self) -> None:
        _, selected = self.service.plan(
            topic="LVLM reliability",
            literature_analysis=self.analysis,
            candidate_ideas=self.ideas,
            selected_idea=self.ideas[0],
            evidence_context=self.evidence,
            verification=self.verification(novelty_passed=False),
        )

        self.assertEqual(selected.novelty_risk, "high")
        self.assertNotEqual(selected.recommended_priority, "high")

    def test_empty_ideas_returns_conservative_fallback(self) -> None:
        directions, selected = self.service.plan(
            topic="LVLM reliability",
            literature_analysis={
                "research_gap_status": "insufficient_evidence",
            },
            candidate_ideas=[],
            selected_idea=None,
            evidence_context=[],
            verification=None,
        )

        self.assertEqual(len(directions), 1)
        self.assertIs(selected, directions[0])
        self.assertIsNone(selected.source_idea_title)
        self.assertEqual(selected.evidence_support_level, "insufficient")
        self.assertEqual(selected.recommended_priority, "exploratory")

    def test_selection_rejects_direction_that_does_not_map_to_selected_idea(
        self,
    ) -> None:
        direction = ResearchDirection(
            title="Direction A",
            source_idea_title="Idea A",
            target_gap="A local gap.",
            core_problem="A scoped problem.",
            hypothesis="A testable hypothesis.",
            method_sketch="A bounded method.",
        )

        with self.assertRaisesRegex(ValueError, "Idea B"):
            self.service.select_direction(
                [direction],
                selected_idea_title="Idea B",
            )

    def test_duplicate_titles_are_aligned_by_index(self) -> None:
        ideas = [
            ResearchIdea(
                title="Duplicate title",
                hypothesis="Hypothesis zero.",
                motivation="Motivation zero.",
                method="Method zero.",
            ),
            ResearchIdea(
                title="Duplicate title",
                hypothesis="Hypothesis one.",
                motivation="Motivation one.",
                method="Method one.",
            ),
        ]

        directions, selected = self.service.plan(
            topic="duplicate ideas",
            literature_analysis=self.analysis,
            candidate_ideas=ideas,
            selected_idea=ideas[1],
            selected_idea_index=1,
            evidence_context=self.evidence,
            verification=self.verification(),
        )

        self.assertEqual(selected.source_idea_index, 1)
        self.assertEqual(
            directions[0].assessment_status,
            "heuristic_unverified",
        )
        self.assertEqual(
            directions[1].assessment_status,
            "verifier_assessed_selected",
        )

    def test_out_of_range_selected_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.service.plan(
                topic="LVLM reliability",
                literature_analysis=self.analysis,
                candidate_ideas=self.ideas,
                selected_idea=self.ideas[0],
                selected_idea_index=5,
                evidence_context=self.evidence,
                verification=self.verification(),
            )

    def test_duplicate_title_fallback_requires_explicit_index(self) -> None:
        duplicate_ideas = [self.ideas[0], self.ideas[0]]

        with self.assertRaisesRegex(ValueError, "required"):
            self.service.generate_directions(
                topic="LVLM reliability",
                literature_analysis=self.analysis,
                candidate_ideas=duplicate_ideas,
                selected_idea_title=self.ideas[0].title,
                evidence_context=self.evidence,
                verification=self.verification(),
            )


class FeasibilityServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FeasibilityService()
        self.direction = ResearchDirection(
            title="Evidence-aware verification",
            source_idea_title="Evidence-aware verification",
            source_idea_index=0,
            target_gap="Verification remains incomplete.",
            core_problem="How to improve grounded reliability.",
            hypothesis="A verifier may reduce unsupported outputs.",
            method_sketch="Route uncertain outputs through a verifier.",
            supporting_evidence=["E1"],
            evidence_support_level="strong",
            novelty_risk="low",
            feasibility_risk="low",
            recommended_priority="high",
            assessment_status="verifier_assessed_selected",
        )
        self.idea = ResearchIdea(
            title="Evidence-aware verification",
            hypothesis="A verifier may reduce unsupported outputs.",
            motivation="Unsupported outputs remain under uncertainty.",
            method="Route uncertain outputs through a lightweight verifier.",
        )
        self.plan = complete_experiment_plan()
        self.evidence_assessment = {
            "status": "sufficient",
            "gaps": [],
        }

    @staticmethod
    def verification(
        *,
        evidence_passed: bool = True,
        experiment_passed: bool = True,
        reproducibility_passed: bool = True,
    ) -> dict:
        return {
            "evidence": {"passed": evidence_passed, "support_level": "strong"},
            "novelty": {"passed": True},
            "experiment": {"passed": experiment_passed},
            "reproducibility": {"passed": reproducibility_passed},
        }

    def assess(self, verification: dict) -> FeasibilityAssessment:
        return self.service.assess(
            selected_direction=self.direction,
            selected_idea=self.idea,
            experiment_plan=self.plan,
            evidence_assessment=self.evidence_assessment,
            verification=verification,
        )

    def test_feasibility_assessment_serializes(self) -> None:
        assessment = FeasibilityAssessment(
            direction_title="Evidence-aware verification",
            source_idea_index=0,
            overall_score=0.72,
            recommendation="proceed_with_caution",
            evidence_readiness="partial",
            experiment_readiness="partial",
            reproducibility_readiness="ready",
            resource_requirement="unknown",
            implementation_readiness="partial",
            dataset_clarity="specified",
            baseline_clarity="specified",
            metric_clarity="specified",
            main_risks=["Experiment design needs refinement."],
            mitigation_strategies=["Define a minimum viable experiment."],
            minimum_viable_experiment=["Run one baseline comparison."],
            assessment_note="Planning-stage only.",
        )

        data = assessment.to_dict()

        self.assertEqual(data["source_idea_index"], 0)
        self.assertEqual(data["recommendation"], "proceed_with_caution")
        self.assertEqual(data["implementation_readiness"], "partial")

    def test_evidence_failure_cannot_recommend_pilot(self) -> None:
        assessment = self.assess(
            self.verification(evidence_passed=False)
        )

        self.assertEqual(assessment.evidence_readiness, "insufficient")
        self.assertEqual(assessment.recommendation, "needs_more_evidence")
        self.assertLess(assessment.overall_score, 0.75)

    def test_experiment_failure_requires_caution(self) -> None:
        assessment = self.assess(
            self.verification(experiment_passed=False)
        )

        self.assertEqual(assessment.experiment_readiness, "partial")
        self.assertEqual(assessment.recommendation, "proceed_with_caution")

    def test_all_passed_is_ready_for_pilot_planning(self) -> None:
        assessment = self.assess(self.verification())

        self.assertEqual(
            assessment.recommendation,
            "ready_for_pilot_planning",
        )
        self.assertGreater(assessment.overall_score, 0.75)
        self.assertEqual(assessment.implementation_readiness, "ready")
        self.assertTrue(
            any("POPE" in step for step in assessment.minimum_viable_experiment)
        )
        self.assertTrue(
            any(
                "Base LVLM" in step
                for step in assessment.minimum_viable_experiment
            )
        )

    def test_readiness_score_does_not_double_count_direction_priority(
        self,
    ) -> None:
        high_priority = self.assess(self.verification())
        self.direction.recommended_priority = "low"
        low_priority = self.assess(self.verification())

        self.assertEqual(
            high_priority.overall_score,
            low_priority.overall_score,
        )

    def test_assessment_does_not_mutate_selected_inputs(self) -> None:
        direction_before = self.direction.to_dict()
        idea_before = self.idea.to_dict()
        plan_before = self.plan.to_dict()

        self.assess(self.verification())

        self.assertEqual(self.direction.to_dict(), direction_before)
        self.assertEqual(self.idea.to_dict(), idea_before)
        self.assertEqual(self.plan.to_dict(), plan_before)

    def test_resource_requirement_stays_unknown_without_explicit_signal(
        self,
    ) -> None:
        plan = complete_experiment_plan()
        plan.risks = ["Performance may vary across datasets."]
        plan.implementation_notes = ["Pin dependency versions and seeds."]
        idea = ResearchIdea(
            title="Routing method",
            hypothesis="Routing may improve reliability.",
            motivation="Reliability remains limited.",
            method="Route uncertain examples through a verifier.",
        )

        assessment = self.service.assess(
            selected_direction=self.direction,
            selected_idea=idea,
            experiment_plan=plan,
            evidence_assessment=self.evidence_assessment,
            verification=self.verification(),
        )

        self.assertEqual(assessment.resource_requirement, "unknown")


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


class ClaimFilteringAndDomainConsistencyTests(unittest.TestCase):
    def test_diagnostic_markers_are_not_verifiable_claims(self) -> None:
        self.assertFalse(
            is_verifiable_claim("Not explicitly stated in the local evidence")
        )
        self.assertFalse(
            is_verifiable_claim(
                "A defensible research gap cannot be established from the "
                "retrieved evidence."
            )
        )
        self.assertTrue(
            is_verifiable_claim(
                "Hierarchical planning reduces cascading failures in LLM agents."
            )
        )

    def test_pipeline_filters_insufficient_research_gap(self) -> None:
        idea = ResearchIdea(
            title="Planning error tracing",
            hypothesis="Tracing planning errors may reduce agent failures.",
            motivation="LLM agent failures propagate across planning steps.",
            method="Trace and verify each planning transition.",
            evidence_refs=["E1"],
        )
        evidence = [{
            "evidence_id": "E1",
            "paper_id": "paper",
            "title": "LLM agent planning failures",
            "source_path": "paper.txt",
            "file_type": "txt",
            "page": None,
            "section": "Results",
            "chunk_id": "C1",
            "text": (
                "LLM agents fail when planning errors propagate. Planning "
                "transition verification identifies cascading failures."
            ),
            "score": 0.8,
            "matched_keywords": ["agent", "llm", "planning"],
            "supporting_claim": "LLM agents fail when planning errors propagate.",
            "support_level": "strong",
        }]
        analysis = {
            "research_gap": (
                "A defensible research gap cannot be established from the "
                "retrieved evidence."
            ),
            "research_gap_status": "insufficient_evidence",
            "existing_methods": [
                "LLM agents fail when planning errors propagate."
            ],
        }

        pipeline = VerificationPipeline()
        verification = pipeline.verify(
            idea,
            complete_experiment_plan(),
            evidence,
            analysis,
            [],
            [idea],
            topic="LLM agent planning failure",
        )
        assessment = pipeline.build_evidence_assessment(
            [{**evidence[0], "kind": "local_paper"}],
            verification["evidence"],
            analysis,
        )

        self.assertFalse(
            any(
                "A defensible research gap cannot" in claim
                for claim in verification["evidence"]["unsupported_claims"]
            )
        )
        self.assertIn(
            "Research gap could not be established from retrieved evidence.",
            assessment["gaps"],
        )

    def test_topic_domain_consistency_negative_and_positive_cases(self) -> None:
        negative = evidence_matches_concept(
            (
                "Autonomous driving agents operate in networked traffic "
                "environments and predict actions."
            ),
            "Autonomous Driving",
            "graph neural network traffic prediction",
        )
        llm_positive = domain_consistency_score(
            "LLM agent planning failure and hallucination mitigation",
            [{
                "evidence_id": "E1",
                "text": (
                    "LLM-based web agents fail when planning logic is affected "
                    "by hallucination, causing error propagation."
                ),
            }],
        )
        agentic_positive = domain_consistency_score(
            "agentic AI reliability and hallucination in action",
            [{
                "evidence_id": "E2",
                "text": (
                    "Agentic AI systems face hallucination in action when "
                    "executing tool calls."
                ),
            }],
        )

        self.assertFalse(negative["domain_consistent"])
        self.assertTrue(llm_positive["passed"])
        self.assertTrue(agentic_positive["passed"])

    def test_hyphen_normalization_and_soft_fallback_diagnostics(self) -> None:
        hyphenated = domain_consistency_score(
            "multi-agent collaboration reliability",
            [{
                "evidence_id": "E1",
                "text": "Multi agent collaboration improves reliability.",
            }],
        )
        semantic_only = domain_consistency_score(
            "agent reliability under long-horizon tasks",
            [{
                "evidence_id": "E2",
                "text": "Robust autonomous systems operate over extended workflows.",
            }],
            TopicConsistencyConfig(mode="strict"),
        )

        self.assertTrue(hyphenated["passed"])
        self.assertFalse(semantic_only["passed"])
        self.assertTrue(semantic_only["warnings"])
        self.assertIn(
            "No direct topic-critical concept matched",
            semantic_only["reason"],
        )


class LiteratureAnalysisServiceTests(unittest.TestCase):
    class StubAnalyzer:
        def __init__(self, limitation: str) -> None:
            self.limitation = limitation

        def extract_problem_method_experiment_limitation(self, text: str) -> dict:
            return {
                "problem": ["Agent reliability"],
                "method": ["Hierarchical planning verifies transitions."],
                "experiment": ["A benchmark comparison."],
                "limitation": [self.limitation],
            }

    @staticmethod
    def evidence() -> list[dict]:
        return [{
            "excerpt": "A paper excerpt without explicit section headings.",
            "text": "A paper excerpt without explicit section headings.",
            "section": None,
        }]

    def test_no_evidence_returns_insufficient_status(self) -> None:
        service = LiteratureAnalysisService(
            self.StubAnalyzer("Planning errors propagate.")
        )

        result = service.analyze([])

        self.assertEqual(result["research_gap_status"], "insufficient_evidence")

    def test_placeholder_limitation_returns_insufficient_status(self) -> None:
        service = LiteratureAnalysisService(
            self.StubAnalyzer("Not explicitly stated in the local evidence")
        )

        result = service.analyze(self.evidence())

        self.assertEqual(result["research_gap_status"], "insufficient_evidence")

    def test_concrete_limitation_returns_supported_status(self) -> None:
        service = LiteratureAnalysisService(
            self.StubAnalyzer(
                "Planning errors propagate across long tool-use trajectories."
            )
        )

        result = service.analyze(self.evidence())

        self.assertEqual(result["research_gap_status"], "evidence_supported")


if __name__ == "__main__":
    unittest.main()

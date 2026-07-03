"""Tests for the offline AI Scientific Agent MVP."""

import json
import tempfile
import unittest
from pathlib import Path

from app.agent.ai_scientific_agent import AIScientificAgent
from app.memory.scientific_memory import ScientificMemory
from app.planner.research_planner import ResearchPlanner
from app.schemas.scientific_task import ScientificTaskType


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
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "memory_paper.txt").write_text(
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
            self.assertTrue((output_dir / "result.json").exists())
            self.assertTrue((output_dir / "report.md").exists())
            saved = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_idea"]["title"], result["selected_idea"]["title"])

    def test_unreadable_pdf_does_not_abort_local_evidence_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "broken.pdf").write_bytes(b"not a valid PDF")
            (data_dir / "valid.txt").write_text(
                "Method: Agent memory retrieves relevant historical interactions.",
                encoding="utf-8",
            )
            agent = AIScientificAgent(project_root=root)

            evidence = agent._retrieve_evidence("agent memory")

            self.assertTrue(evidence)
            self.assertTrue(all(item["source"] != "data\\broken.pdf" for item in evidence))


if __name__ == "__main__":
    unittest.main()

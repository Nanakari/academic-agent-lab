"""Tests for the deterministic Chinese presentation layer."""

from __future__ import annotations

from copy import deepcopy
import unittest

from app.frontend.chinese_renderer import (
    evidence_quality_messages,
    render_chinese_markdown_report,
    render_chinese_summary,
    verification_failure_summary,
    zh_bool,
    zh_decision,
    zh_status,
    zh_task_type,
    zh_verification_name,
)


def sample_result() -> dict:
    return {
        "topic": "科研 Agent 长期记忆去重",
        "task_type": "research_proposal",
        "evidence_status": "evidence_insufficient",
        "verification_passed": False,
        "revision_performed": True,
        "plan": {
            "steps": [{
                "name": "retrieve_evidence",
                "description": "Search local evidence.",
            }]
        },
        "evidence_context": [{
            "evidence_id": "E1",
            "title": "Agent Memory",
            "source": "data/papers/memory.md",
            "page": None,
            "section": "Method",
            "matched_keywords": ["agent", "memory"],
            "support_level": "moderate",
            "supporting_claim": "Memory deduplication reduces repeated records.",
            "excerpt": "A lightweight memory deduplication method is evaluated.",
        }],
        "external_search_status": {"enabled": False},
        "candidate_ideas": [{
            "title": "Memory deduplication",
            "hypothesis": "Deduplication may improve retrieval.",
            "motivation": "Repeated records introduce noise.",
            "method": "Compare deduplication policies.",
            "evidence_refs": ["E1"],
            "novelty_score": 0.2,
            "feasibility_score": 0.8,
            "rank_score": 0.5,
        }],
        "selected_idea": {"title": "Memory deduplication"},
        "experiment_plan": {
            "method": "Compare deterministic deduplication policies.",
            "datasets": ["Memory fixture"],
            "baselines": ["No deduplication"],
            "metrics": ["Duplicate rate"],
            "ablation": ["Remove recency key"],
            "expected_results": ["Lower duplicate rate"],
            "risks": ["Over-filtering"],
            "implementation_notes": ["Pin versions"],
        },
        "verification": {
            "evidence": {
                "passed": False,
                "score": 0.1,
                "issues": ["Evidence is insufficient."],
            },
            "novelty": {
                "passed": False,
                "score": 0.2,
                "issues": [
                    "Insufficient literature evidence to assess academic novelty."
                ],
                "local_memory_overlap": {
                    "has_overlap": True,
                    "max_similarity": 0.94,
                    "matched_title": "Earlier memory deduplication draft",
                    "effect": "warning_only",
                },
                "literature_novelty": {
                    "status": "insufficient_literature_evidence",
                    "risk": "unknown",
                    "mechanism_difference": (
                        "Insufficient literature for comparison."
                    ),
                },
            },
        },
        "agent_trace": [{
            "step": 1,
            "observation": "A similar idea exists.",
            "decision": "trigger_bounded_revision",
            "action": "perform_bounded_revision",
            "reason": "Novelty verification failed.",
            "result": "One revision was performed.",
        }],
    }


class ChineseMappingTests(unittest.TestCase):
    def test_fixed_value_mappings(self) -> None:
        self.assertEqual(zh_bool(True), "是")
        self.assertEqual(zh_bool(False), "否")
        self.assertEqual(zh_task_type("research_proposal"), "研究方案生成")
        self.assertEqual(zh_task_type("custom"), "custom（未映射）")
        self.assertEqual(zh_status("evidence_insufficient"), "证据不足")
        self.assertEqual(zh_verification_name("novelty"), "新颖性验证")
        self.assertEqual(
            zh_decision("trigger_bounded_revision"),
            "触发一次有限修订",
        )

    def test_summary_uses_actual_failed_verifier(self) -> None:
        summary = render_chinese_summary(sample_result())

        self.assertIn("科研 Agent 长期记忆去重", summary)
        self.assertIn("主要失败项：证据验证、新颖性验证", summary)
        self.assertIn("无法可靠判断学术新颖性", summary)
        self.assertIn(
            "新颖性验证",
            verification_failure_summary(sample_result()),
        )


class ChineseReportTests(unittest.TestCase):
    def test_report_contains_required_chinese_sections(self) -> None:
        report = render_chinese_markdown_report(sample_result())

        for heading in (
            "# AI 科研 Agent 中文报告",
            "## 一、任务概览",
            "## 二、研究计划",
            "## 三、检索到的证据",
            "## 四、候选研究想法",
            "## 五、选中的研究方向",
            "## 六、实验方案",
            "## 七、验证结果",
            "## 八、Agent 执行轨迹",
            "## 九、局限性",
        ):
            self.assertIn(heading, report)
        self.assertIn("外部检索未启用", report)
        self.assertIn("这只是本地去重提醒", report)
        self.assertIn("无法可靠判断学术新颖性", report)
        self.assertIn("本次本地证据不足", report)
        self.assertIn("不替代 result.json", report)

    def test_local_overlap_is_not_reported_as_academic_failure(self) -> None:
        result = sample_result()
        result["verification"]["novelty"]["passed"] = True
        result["verification"]["novelty"]["literature_novelty"] = {
            "status": "potentially_distinct",
            "risk": "medium",
            "mechanism_difference": "A different routing mechanism.",
        }
        report = render_chinese_markdown_report(result)

        self.assertIn("这只是本地去重提醒", report)
        self.assertIn("仍需人工复核最新文献", report)
        self.assertNotIn("本地历史重复导致学术新颖性失败", report)

    def test_zero_relevance_external_results_are_cautioned(self) -> None:
        result = deepcopy(sample_result())
        result["external_search_status"] = {"enabled": True}
        result["external_evidence"] = [{
            "source_type": "arxiv",
            "title": "Unrelated result",
            "relevance_score": 0.0,
        }]
        result["external_evidence_used_for_literature"] = []

        messages = evidence_quality_messages(result)
        report = render_chinese_markdown_report(result)

        self.assertTrue(any("相关性分数为 0" in item for item in messages))
        self.assertIn("由于相关性不足，未用于文献分析", report)

    def test_rejected_external_evidence_is_listed(self) -> None:
        result = deepcopy(sample_result())
        result["external_evidence_rejected_for_literature"] = [{
            "title": "X-to-4D Generation",
            "relevance_score": 0.167,
            "reason": "Excluded domain without topic-core concepts.",
        }]

        report = render_chinese_markdown_report(result)

        self.assertIn("被拒绝的外部证据", report)
        self.assertIn("X-to-4D Generation", report)
        self.assertIn("Excluded domain", report)

    def test_weak_evidence_and_gap_status_are_explained(self) -> None:
        result = deepcopy(sample_result())
        result["evidence_context"][0]["support_level"] = "weak"
        result["literature_analysis"] = {
            "research_gap_status": "insufficient_topic_relevant_evidence"
        }

        report = render_chinese_markdown_report(result)

        self.assertIn("可支持问题背景，但不足以支撑完整研究方案", report)
        self.assertIn("无法从主题相关证据中建立可靠研究空白", report)


if __name__ == "__main__":
    unittest.main()

"""Tests for the deterministic Chinese presentation layer."""

from __future__ import annotations

import unittest

from app.frontend.chinese_renderer import (
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
            "evidence": {"passed": True, "score": 0.8, "issues": []},
            "novelty": {
                "passed": False,
                "score": 0.2,
                "issues": ["Idea overlaps with memory."],
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
        self.assertIn("主要失败项：新颖性验证", summary)
        self.assertIn("已有想法过于相似", summary)
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
        self.assertIn("当前研究想法与历史记忆中的已有想法相似度较高", report)
        self.assertIn("不替代 result.json", report)


if __name__ == "__main__":
    unittest.main()

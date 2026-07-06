"""Deterministic Chinese presentation helpers for scientific-agent results."""

from __future__ import annotations

from typing import Any


TASK_TYPE_LABELS = {
    "research_proposal": "研究方案生成",
    "literature_review": "文献综述",
    "experiment_design": "实验设计",
}

STATUS_LABELS = {
    "sufficient": "证据基本充分",
    "evidence_insufficient": "证据不足",
    "insufficient_evidence": "证据不足",
    "warning": "需要人工复核",
    "failed": "失败",
    "passed": "通过",
    "strong": "强",
    "moderate": "中等",
    "weak": "弱",
    "unknown": "未知",
}

VERIFICATION_LABELS = {
    "evidence": "证据验证",
    "novelty": "新颖性验证",
    "experiment": "实验完整性验证",
    "reproducibility": "可复现性验证",
}

DECISION_LABELS = {
    "mark_insufficient_evidence": "标记为证据不足",
    "downgrade_gap_confidence": "降低研究空白置信度",
    "continue_grounded_planning": "继续基于证据进行规划",
    "use_external_evidence_as_supplement": "将外部证据作为补充材料",
    "skip_revision": "跳过修订",
    "trigger_bounded_revision": "触发一次有限修订",
    "accept_current_plan": "接受当前方案",
    "revise_or_mark_evidence_gap": "修订或明确标记证据缺口",
    "repair_experiment_plan": "修复实验方案",
    "improve_reproducibility_notes": "补充可复现性说明",
    "preserve_verifier_failure": "保留验证失败结果",
    "recommend_as_pre_experiment_plan": "推荐为实验前规划方案",
    "report_as_exploratory_or_insufficient": "以探索性/证据不足方式报告",
}

PLAN_STEP_LABELS = {
    "retrieve_evidence": "检索证据",
    "analyze_literature": "分析文献",
    "generate_and_rank_ideas": "生成并排序研究想法",
    "design_experiment": "设计实验",
    "verify_and_report": "验证并生成报告",
}


def zh_bool(value: bool) -> str:
    return "是" if value else "否"


def zh_task_type(task_type: str) -> str:
    return TASK_TYPE_LABELS.get(task_type, f"{task_type}（未映射）")


def zh_status(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def zh_verification_name(name: str) -> str:
    return VERIFICATION_LABELS.get(name, name)


def zh_decision(decision: str) -> str:
    return DECISION_LABELS.get(decision, decision)


def failed_verification_names(result: dict[str, Any]) -> list[str]:
    verification = result.get("verification") or {}
    return [
        zh_verification_name(name)
        for name, detail in verification.items()
        if isinstance(detail, dict) and not detail.get("passed", False)
    ]


def verification_failure_summary(result: dict[str, Any]) -> str:
    failed_names = failed_verification_names(result)
    if not failed_names:
        return "未记录失败的验证项目。"
    summary = f"主要失败项：{'、'.join(failed_names)}。"
    novelty = (result.get("verification") or {}).get("novelty") or {}
    if not novelty.get("passed", True):
        summary += (
            "新颖性验证未通过，当前想法可能与历史记忆中的已有想法"
            "过于相似。"
        )
    return summary


def render_chinese_summary(result: dict[str, Any]) -> str:
    lines = [
        "### 运行结果总览",
        "",
        f"- **研究主题：** {result.get('topic') or '未提供'}",
        f"- **检索扩展关键词：** {result.get('expanded_query') or '未记录'}",
        f"- **任务类型：** {zh_task_type(str(result.get('task_type', 'unknown')))}",
        f"- **证据状态：** {zh_status(str(result.get('evidence_status', 'unknown')))}",
        f"- **是否通过验证：** {zh_bool(bool(result.get('verification_passed')))}",
    ]
    if not result.get("verification_passed"):
        lines.extend([
            "",
            f"> 本次运行未完全通过验证。{verification_failure_summary(result)}",
        ])
    return "\n".join(lines)


def _append_values(
    lines: list[str],
    label: str,
    values: Any,
    *,
    empty: str = "未提供",
) -> None:
    lines.extend(["", f"**{label}：**"])
    if isinstance(values, list) and values:
        lines.extend(f"- {value}" for value in values)
    elif values not in (None, "", []):
        lines.append(f"- {values}")
    else:
        lines.append(f"- {empty}")


def _verification_issues(detail: dict[str, Any]) -> list[str]:
    issues = detail.get("issues") or []
    return [str(issue) for issue in issues] or ["未记录具体问题。"]


def evidence_quality_messages(result: dict[str, Any]) -> list[str]:
    """Return cautious Chinese diagnostics derived from recorded evidence."""
    messages: list[str] = []
    local_evidence = result.get("evidence_context") or []
    if result.get("evidence_status") == "evidence_insufficient":
        messages.append(
            "本次本地证据不足。虽然系统检索到了若干论文片段，但最高证据"
            "分数低于阈值，且支持等级为 insufficient，因此当前研究方向"
            "只能作为探索性草案。"
        )
    if local_evidence and all(
        item.get("support_level") == "insufficient"
        for item in local_evidence
    ):
        messages.append(
            "所有本地证据均为 insufficient，建议补充更直接相关的论文，"
            "例如 indirect prompt injection、tool-use security、"
            "computer-use agent security、MCP governance 等方向。"
        )
    external_evidence = result.get("external_evidence") or []
    if any(
        float(item.get("relevance_score") or 0.0) == 0.0
        for item in external_evidence
    ):
        messages.append(
            "部分外部检索结果相关性分数为 0，中文报告仅将其作为检索记录，"
            "不应将其视为支持该研究方向的证据。"
        )
    external_status = result.get("external_search_status") or {}
    external_used = result.get("external_evidence_used_for_literature") or []
    if (
        external_status.get("enabled")
        and external_evidence
        and not external_used
    ):
        messages.append(
            "外部检索结果已记录，但由于相关性不足，未用于文献分析和"
            "研究空白生成。"
        )
    return messages


def render_chinese_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# AI 科研 Agent 中文报告",
        "",
        "## 一、任务概览",
        "",
        f"- **研究主题：** {result.get('topic') or '未提供'}",
        f"- **检索扩展关键词：** {result.get('expanded_query') or '未记录'}",
        f"- **任务类型：** {zh_task_type(str(result.get('task_type', 'unknown')))}",
        f"- **证据状态：** {zh_status(str(result.get('evidence_status', 'unknown')))}",
        f"- **是否通过验证：** {zh_bool(bool(result.get('verification_passed')))}",
        f"- **是否执行修订：** {zh_bool(bool(result.get('revision_performed')))}",
    ]
    if not result.get("verification_passed"):
        lines.extend(["", f"> {verification_failure_summary(result)}"])

    lines.extend(["", "## 二、研究计划", ""])
    plan_steps = (result.get("plan") or {}).get("steps") or []
    if plan_steps:
        for index, step in enumerate(plan_steps, start=1):
            name = str(step.get("name", "unknown"))
            lines.append(
                f"{index}. **{PLAN_STEP_LABELS.get(name, name)}**"
                f"：{step.get('description') or '未提供步骤说明。'}"
            )
    else:
        lines.append("- 未生成研究计划。")

    lines.extend([
        "",
        "## 三、检索到的证据",
        "",
        "### 证据质量判断",
        "",
    ])
    quality_messages = evidence_quality_messages(result)
    lines.extend(
        [f"- {message}" for message in quality_messages]
        if quality_messages
        else ["- 当前轻量检查未发现需要额外提示的证据质量问题。"]
    )
    lines.extend(["", "### 本地证据", ""])
    evidence_context = result.get("evidence_context") or []
    if not evidence_context:
        lines.append("- 未检索到相关本地证据。")
    for index, evidence in enumerate(evidence_context, start=1):
        page = evidence.get("page")
        lines.extend([
            f"### 证据 {index}：{evidence.get('title') or '未命名论文'}",
            "",
            f"- **证据编号：** {evidence.get('evidence_id') or '未提供'}",
            f"- **来源文件：** {evidence.get('source') or '未提供'}",
            f"- **页码：** {page if page is not None else '未提供'}",
            f"- **章节：** {evidence.get('section') or '未提供'}",
            (
                "- **匹配关键词：** "
                + ("、".join(evidence.get("matched_keywords") or []) or "无")
            ),
            (
                "- **支持等级：** "
                + zh_status(str(evidence.get("support_level", "unknown")))
            ),
            (
                "- **支持性陈述：** "
                + str(evidence.get("supporting_claim") or "未识别")
            ),
            "",
            "**原文摘录：**",
            "",
            str(evidence.get("excerpt") or evidence.get("text") or "无"),
            "",
        ])

    external_status = result.get("external_search_status") or {}
    lines.extend(["### 外部证据", ""])
    if not external_status.get("enabled"):
        lines.append("- 外部检索未启用，本次运行仅使用本地论文和本地记忆。")
    else:
        external_items = result.get("external_evidence") or []
        if external_items:
            for item in external_items:
                source = (
                    "arXiv"
                    if item.get("source_type") == "arxiv"
                    else "GitHub 仓库"
                )
                lines.append(
                    f"- **{source}：** {item.get('title') or '未命名条目'}"
                    f"（{item.get('url') or '无链接'}）"
                )
        else:
            lines.append("- 已启用外部检索，但未获得结果。")

    lines.extend(["", "## 四、候选研究想法", ""])
    ideas = result.get("candidate_ideas") or []
    if not ideas:
        lines.append("- 未生成候选研究想法。")
    for index, idea in enumerate(ideas, start=1):
        lines.extend([
            f"### 想法 {index}：{idea.get('title') or '未命名想法'}",
            "",
            f"- **假设：** {idea.get('hypothesis') or '未提供'}",
            f"- **动机：** {idea.get('motivation') or '未提供'}",
            f"- **方法：** {idea.get('method') or '未提供'}",
            (
                "- **证据引用：** "
                + ("、".join(idea.get("evidence_refs") or []) or "无")
            ),
            f"- **新颖性分数：** {idea.get('novelty_score', 0.0)}",
            f"- **可行性分数：** {idea.get('feasibility_score', 0.0)}",
            f"- **综合排序分数：** {idea.get('rank_score', 0.0)}",
            "",
        ])

    selected = result.get("selected_direction") or result.get("selected_idea") or {}
    novelty_detail = (result.get("verification") or {}).get("novelty") or {}
    lines.extend([
        "## 五、选中的研究方向",
        "",
        f"- **标题：** {selected.get('title') or '未提供'}",
        (
            "- **选择原因：** 该方向与用于实验规划的首选研究想法一致；"
            "其优先级来自确定性规划规则，不代表已证明新颖或可行。"
        ),
        (
            "- **当前主要问题：** "
            + (
                verification_failure_summary(result)
                if not result.get("verification_passed")
                else "当前轻量验证未记录阻断性问题，仍需人工复核。"
            )
        ),
        (
            "- **是否存在新颖性风险：** "
            + zh_bool(not novelty_detail.get("passed", False))
        ),
        "",
        "## 六、实验方案",
    ])
    experiment = result.get("experiment_plan") or {}
    lines.extend(["", f"- **方法：** {experiment.get('method') or '未提供'}"])
    for field, label in (
        ("datasets", "数据集"),
        ("baselines", "基线方法"),
        ("metrics", "评价指标"),
        ("ablation", "消融实验"),
        ("expected_results", "预期结果"),
        ("risks", "风险"),
        ("implementation_notes", "实现注意事项"),
    ):
        _append_values(lines, label, experiment.get(field))

    lines.extend(["", "## 七、验证结果", ""])
    verification = result.get("verification") or {}
    if not verification:
        lines.append("- 未生成验证结果。")
    for name, detail in verification.items():
        lines.extend([
            f"### {zh_verification_name(name)}",
            "",
            f"- **是否通过：** {zh_bool(bool(detail.get('passed')))}",
            f"- **分数：** {detail.get('score', 0.0)}",
            "- **问题说明：**",
            *[f"  - {issue}" for issue in _verification_issues(detail)],
            "",
        ])
        if name == "novelty" and not detail.get("passed", False):
            lines.extend([
                "> 当前研究想法与历史记忆中的已有想法相似度较高，"
                "因此不应直接作为高新颖性方向使用。",
                "",
            ])

    lines.extend(["## 八、Agent 执行轨迹", ""])
    trace = result.get("agent_trace") or []
    if not trace:
        lines.append("- 未记录执行轨迹。")
    for entry in trace:
        lines.extend([
            f"### 步骤 {entry.get('step', '未知')}",
            "",
            f"- **观察：** {entry.get('observation') or '未提供'}",
            f"- **决策：** {zh_decision(str(entry.get('decision', '未知')))}",
            f"- **动作：** {entry.get('action') or '未提供'}",
            f"- **原因：** {entry.get('reason') or '未提供'}",
            f"- **结果：** {entry.get('result') or '未提供'}",
            "",
        ])

    lines.extend([
        "## 九、局限性",
        "",
        "- 当前检索主要依赖本地论文和轻量关键词匹配。",
        "- 证据充分不代表研究空白一定成立。",
        "- 新颖性验证依赖历史记忆和启发式相似度。",
        "- 实验方案只是规划，不代表真实实验结果。",
        "- 执行实验前仍需人工复核最新文献、数据集、基线和评价指标。",
        "",
        "> 本中文报告是对原始结构化结果的展示，不替代 result.json。",
    ])
    return "\n".join(lines)

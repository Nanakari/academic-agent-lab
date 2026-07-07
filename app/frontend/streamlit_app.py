"""Chinese Streamlit presentation layer for AIScientificAgent."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.ai_scientific_agent import AIScientificAgent
from app.cli.scientific import build_default_llm
from app.frontend.chinese_renderer import (
    render_chinese_markdown_report,
    render_chinese_summary,
    novelty_display_messages,
    verification_failure_summary,
    zh_bool,
    zh_decision,
    zh_status,
    zh_task_type,
    zh_verification_name,
)
from app.frontend.result_artifacts import read_result_json_bytes


PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ai_scientific_agent"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
BROAD_TOPICS = {"agent", "llm", "ai", "rag"}


def _safe_filename(upload_name: str) -> str:
    """Return a flat, portable filename for an uploaded paper."""
    original = Path(upload_name).name
    suffix = Path(original).suffix.casefold()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}")
    stem = Path(original).stem
    safe_stem = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in stem
    ).strip("._")
    return f"{safe_stem or 'uploaded_paper'}{suffix}"


def _save_uploaded_file(upload_name: str, content: bytes) -> Path:
    """Save an upload without overwriting a different existing paper."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    destination = PAPERS_DIR / _safe_filename(upload_name)
    if not destination.exists() or destination.read_bytes() == content:
        destination.write_bytes(content)
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = PAPERS_DIR / f"{stem}_{counter}{suffix}"
        if not candidate.exists() or candidate.read_bytes() == content:
            candidate.write_bytes(content)
            return candidate
        counter += 1


def _render_evidence(result: dict[str, Any]) -> None:
    evidence = result.get("evidence_context") or []
    if not evidence:
        st.info("未检索到相关本地证据。")
    for index, item in enumerate(evidence, start=1):
        with st.expander(
            f"证据 {index}：{item.get('title') or '未命名论文'}",
            expanded=index == 1,
        ):
            st.markdown(
                "\n".join([
                    f"- **证据编号：** {item.get('evidence_id') or '未提供'}",
                    f"- **来源文件：** {item.get('source') or '未提供'}",
                    (
                        "- **页码：** "
                        + str(
                            item.get("page")
                            if item.get("page") is not None
                            else "未提供"
                        )
                    ),
                    f"- **章节：** {item.get('section') or '未提供'}",
                    (
                        "- **匹配关键词：** "
                        + ("、".join(item.get("matched_keywords") or []) or "无")
                    ),
                    (
                        "- **支持等级：** "
                        + zh_status(str(item.get("support_level", "unknown")))
                    ),
                    (
                        "- **支持性陈述：** "
                        + str(item.get("supporting_claim") or "未识别")
                    ),
                    "",
                    "**原文摘录：**",
                    "",
                    str(item.get("excerpt") or item.get("text") or "无"),
                ])
            )
    rejected = result.get("external_evidence_rejected_for_literature") or []
    if rejected:
        with st.expander("被拒绝的外部证据", expanded=False):
            st.caption("这些结果保留用于审计，但未参与文献分析。")
            for item in rejected:
                st.markdown(
                    f"- **{item.get('title') or '未命名条目'}**\n"
                    f"  - 相关性分数：{item.get('relevance_score', 0.0)}\n"
                    f"  - 拒绝原因：{item.get('reason') or '未记录'}"
                )


def _render_ideas(result: dict[str, Any]) -> None:
    ideas = result.get("candidate_ideas") or []
    if not ideas:
        st.info("未生成候选研究想法。")
    for index, idea in enumerate(ideas, start=1):
        with st.expander(
            f"想法 {index}：{idea.get('title') or '未命名想法'}",
            expanded=index == 1,
        ):
            st.markdown(
                "\n".join([
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
                ])
            )


def _render_experiment(result: dict[str, Any]) -> None:
    experiment = result.get("experiment_plan") or {}
    if not experiment:
        st.info("未生成实验方案。")
        return
    st.markdown(f"**方法：** {experiment.get('method') or '未提供'}")
    for field, label in (
        ("datasets", "数据集"),
        ("baselines", "基线方法"),
        ("metrics", "评价指标"),
        ("ablation", "消融实验"),
        ("expected_results", "预期结果"),
        ("risks", "风险"),
        ("implementation_notes", "实现注意事项"),
    ):
        values = experiment.get(field) or []
        st.markdown(
            f"**{label}：**\n"
            + (
                "\n".join(f"- {value}" for value in values)
                if values
                else "- 未提供"
            )
        )


def _render_verification(result: dict[str, Any]) -> None:
    verification = result.get("verification") or {}
    if not verification:
        st.info("未生成验证结果。")
    for name, detail in verification.items():
        with st.expander(zh_verification_name(name), expanded=True):
            st.markdown(
                f"- **是否通过：** {zh_bool(bool(detail.get('passed')))}\n"
                f"- **分数：** {detail.get('score', 0.0)}\n"
                "- **问题说明：**"
            )
            issues = detail.get("issues") or []
            if issues:
                for issue in issues:
                    st.markdown(f"  - {issue}")
            else:
                st.markdown("  - 未记录具体问题。")
            if name == "novelty":
                for severity, message in novelty_display_messages(detail):
                    getattr(st, severity)(message)


def _render_trace(result: dict[str, Any]) -> None:
    trace = result.get("agent_trace") or []
    if not trace:
        st.info("未记录 Agent 执行轨迹。")
    for entry in trace:
        with st.expander(f"步骤 {entry.get('step', '未知')}"):
            st.markdown(
                "\n".join([
                    f"- **观察：** {entry.get('observation') or '未提供'}",
                    (
                        "- **决策：** "
                        + zh_decision(str(entry.get("decision", "未知")))
                    ),
                    f"- **动作：** {entry.get('action') or '未提供'}",
                    f"- **原因：** {entry.get('reason') or '未提供'}",
                    f"- **结果：** {entry.get('result') or '未提供'}",
                ])
            )


def _render_result(
    result: dict[str, Any],
    chinese_report: str,
    result_json: bytes | None,
) -> None:
    st.success("Agent 运行完成。")
    task_column, evidence_column, verification_column = st.columns(3)
    task_column.metric(
        "任务类型",
        zh_task_type(str(result.get("task_type", "unknown"))),
    )
    evidence_column.metric(
        "证据状态",
        zh_status(str(result.get("evidence_status", "unknown"))),
    )
    verification_column.metric(
        "是否通过验证",
        zh_bool(bool(result.get("verification_passed"))),
    )
    novelty = (result.get("verification") or {}).get("novelty") or {}
    literature_novelty = novelty.get("literature_novelty") or {}
    local_overlap = novelty.get("local_memory_overlap") or {}
    novelty_column, overlap_column = st.columns(2)
    novelty_column.metric(
        "文献新颖性状态",
        literature_novelty.get("status", "未记录"),
    )
    overlap_column.metric(
        "本地历史草案重叠",
        zh_bool(bool(local_overlap.get("has_overlap"))),
    )
    if not result.get("verification_passed"):
        st.warning(
            "本次运行未完全通过验证。"
            + verification_failure_summary(result)
        )

    tabs = st.tabs([
        "中文总览",
        "证据",
        "候选想法",
        "实验方案",
        "验证结果",
        "执行轨迹",
        "中文报告",
        "原始 JSON",
    ])
    with tabs[0]:
        st.markdown(render_chinese_summary(result))
        selected = result.get("selected_direction") or result.get("selected_idea")
        if selected:
            st.markdown(
                f"### 选中的研究方向\n\n"
                f"**{selected.get('title') or '未提供'}**"
            )
    with tabs[1]:
        _render_evidence(result)
    with tabs[2]:
        _render_ideas(result)
    with tabs[3]:
        _render_experiment(result)
    with tabs[4]:
        _render_verification(result)
    with tabs[5]:
        _render_trace(result)
    with tabs[6]:
        st.markdown(chinese_report)
        st.download_button(
            "下载中文报告 report_zh.md",
            data=chinese_report.encode("utf-8"),
            file_name="report_zh.md",
            mime="text/markdown",
        )
    with tabs[7]:
        st.json(result)
        if result_json is None:
            st.warning(
                "未找到磁盘上的 result.json 文件，无法提供原始 JSON 下载。"
            )
        else:
            st.download_button(
                "下载原始 JSON result.json",
                data=result_json,
                file_name="result.json",
                mime="application/json",
            )


def main() -> None:
    st.set_page_config(
        page_title="Academic Agent Lab 科研智能体工作台",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Academic Agent Lab 科研智能体工作台")
    st.caption("上传本地论文，并运行现有的 verifier-driven AIScientificAgent。")

    uploaded_files = st.file_uploader(
        "上传论文",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="运行 Agent 时，文件将保存到 data/papers/。",
    )
    topic = st.text_input(
        "研究主题",
        placeholder="例如：基于验证器的 Computer-use Agent 可靠性评估",
    )
    st.caption(
        "不建议只输入 Agent、LLM、RAG 这类过宽主题。建议输入更具体的研究问题，例如："
    )
    st.markdown(
        "- LLM Agent 在间接提示注入攻击下的工具调用安全\n"
        "- 基于验证器的 Computer-use Agent 可靠性评估\n"
        "- 面向科研 Agent 的长期记忆去重与新颖性检查\n"
        "- MCP 工具治理机制在 Agent 系统中的安全边界"
    )
    if topic.strip().casefold() in BROAD_TOPICS:
        st.warning(
            "当前主题过于宽泛，可能导致检索证据分散、研究空白不明确。"
            "建议输入更具体的研究方向。"
        )

    top_k = st.number_input(
        "证据检索数量 top_k",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )
    use_external_search = st.checkbox(
        "是否启用外部检索",
        value=False,
        help="默认关闭，以保持 offline-first 行为。",
    )
    external_sources = st.multiselect(
        "外部来源",
        options=["arxiv", "github"],
        default=["arxiv", "github"],
        disabled=not use_external_search,
    )
    offline_mode = st.checkbox(
        "Offline regression mode (disable LLM tool decision)",
        value=False,
        help=(
            "Default runs call the configured LLM so the agent can choose tools. "
            "Enable only for offline regression checks."
        ),
    )

    if st.button("运行 Agent", type="primary"):
        if not topic.strip():
            st.error("请输入研究主题。")
        elif use_external_search and not external_sources:
            st.error("请至少选择一个外部来源，或关闭外部检索。")
        else:
            for key in ("agent_result", "agent_report_zh", "agent_result_json"):
                st.session_state.pop(key, None)
            try:
                saved_paths = [
                    _save_uploaded_file(upload.name, upload.getvalue())
                    for upload in uploaded_files
                ]
                if saved_paths:
                    st.info(
                        "已保存论文："
                        + "、".join(
                            path.relative_to(PROJECT_ROOT).as_posix()
                            for path in saved_paths
                        )
                    )
                with st.spinner("正在运行 AIScientificAgent，请稍候……"):
                    llm = None if offline_mode else build_default_llm()
                    agent = AIScientificAgent(
                        project_root=PROJECT_ROOT,
                        output_dir=OUTPUT_DIR,
                        papers_dir=PAPERS_DIR,
                        top_k=int(top_k),
                        llm=llm,
                        llm_tool_decision_enabled=not offline_mode,
                        external_search_enabled=use_external_search,
                        external_search_sources=list(external_sources),
                    )
                    result = agent.run(topic.strip())
                    chinese_report = render_chinese_markdown_report(result)
                    report_zh_path = (
                        Path(result["output_paths"]["markdown"])
                        .with_name("report_zh.md")
                    )
                    report_zh_path.write_text(chinese_report, encoding="utf-8")
                    result_path = Path(result["output_paths"]["json"])
                    result_json = read_result_json_bytes(result_path)
                st.session_state["agent_result"] = result
                st.session_state["agent_report_zh"] = chinese_report
                st.session_state["agent_result_json"] = result_json
            except Exception as exc:
                st.error(f"Agent 运行失败：{exc}")
                st.exception(exc)

    if "agent_result" in st.session_state:
        _render_result(
            st.session_state["agent_result"],
            st.session_state.get("agent_report_zh", ""),
            st.session_state.get("agent_result_json"),
        )


if __name__ == "__main__":
    main()

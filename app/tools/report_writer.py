"""JSON and Markdown report serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ReportWriter:
    """Write stable machine-readable and human-readable research reports."""

    def write_json_report(self, result: dict, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._serialize(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_markdown_report(self, result: dict, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = result["selected_idea"]
        experiment = result["experiment_plan"]
        analysis = result["literature_analysis"]
        lines = [
            "# AI Scientific Agent Report",
            "",
            f"**Topic:** {result['topic']}",
            f"**Task type:** {result['task_type']}",
            "",
            "## Research plan",
            "",
        ]
        lines.extend(
            f"{index}. **{step['name']}** — {step['description']}"
            for index, step in enumerate(result["plan"]["steps"], start=1)
        )
        lines.extend([
            "",
            "## Evidence Used",
            "",
            f"Evidence status: **{result['evidence_status']}**",
            "",
        ])
        if result["evidence_context"]:
            for item in result["evidence_context"]:
                lines.extend([
                    f"### {item['evidence_id']} — {item['title']}",
                    "",
                    (
                        f"- Paper: {item['title']}\n"
                        f"- Source: `{item['source']}`\n"
                        f"- Paper ID: `{item['paper_id']}`\n"
                        f"- File Type: {item['file_type']}\n"
                        f"- Section: {item['section'] or 'N/A'}\n"
                        f"- Page: {item['page'] if item['page'] is not None else 'N/A'}\n"
                        f"- Chunk: `{item['chunk_id']}`\n"
                        f"- Score: {item['score']:.3f}\n"
                        f"- Support Level: {item['support_level']}\n"
                        f"- Matched Keywords: "
                        f"{', '.join(item['matched_keywords']) or 'None'}\n"
                        f"- Supporting Claim: "
                        f"{item['supporting_claim'] or 'No supporting sentence identified.'}"
                    ),
                    "",
                    item["text"],
                    "",
                ])
        else:
            lines.extend(["No relevant local evidence was found. Claims are exploratory.", ""])
        supported_claims = result["verification"]["evidence"].get(
            "supported_claims",
            [],
        )
        if supported_claims:
            lines.extend(["### Claim-to-Evidence Citations", ""])
            for citation in supported_claims:
                lines.extend([
                    f"- Claim: {citation['claim']}",
                    (
                        f"  - Citation: {citation['title']} / "
                        f"{citation['section'] or 'N/A'} / "
                        f"page {citation['page'] if citation['page'] is not None else 'N/A'} / "
                        f"chunk {citation['chunk_id']} "
                        f"({citation['support_level']}, {citation['score']:.3f})"
                    ),
                ])
            lines.append("")
        external_status = result.get("external_search_status", {"enabled": False})
        external_evidence = result.get("external_evidence", [])
        lines.extend(["## External Evidence Retrieved", ""])
        if not external_status.get("enabled"):
            lines.append(
                "External search was disabled; this run used only local papers "
                "and local scientific memory."
            )
        else:
            if not external_evidence:
                lines.append("No external evidence was retrieved.")
            else:
                arxiv_items = [
                    item for item in external_evidence
                    if item.get("source_type") == "arxiv"
                ]
                github_items = [
                    item for item in external_evidence
                    if item.get("source_type") == "github_repo"
                ]
                if arxiv_items:
                    lines.extend(["", "### arXiv evidence", ""])
                    for item in arxiv_items:
                        lines.append(
                            f"- [{item['title']}]({item.get('url') or ''}) "
                            f"(retrieved {item.get('retrieved_at') or 'unknown'})"
                        )
                if github_items:
                    lines.extend(["", "### GitHub repository evidence", ""])
                    for item in github_items:
                        lines.append(
                            f"- [{item['title']}]({item.get('url') or ''}) "
                            f"(implementation evidence; retrieved "
                            f"{item.get('retrieved_at') or 'unknown'})"
                        )
            lines.extend([
                "",
                f"- Run at: {external_status.get('run_at') or 'unknown'}",
                (
                    "- Original retrieval times by source: "
                    f"{external_status.get('retrieved_at_by_source') or {}}"
                ),
                (
                    "- Cache loaded at: "
                    f"{external_status.get('cache_loaded_at') or 'not used'}"
                ),
                "",
                "- arXiv evidence is based on metadata / abstract-level retrieval only.",
                (
                    "- GitHub repository evidence indicates implementation "
                    "availability or engineering relevance, not scientific validation."
                ),
                (
                    "- External search results may change over time; retrieved_at "
                    "is recorded for auditability."
                ),
                "",
                "## External Evidence Gaps",
                "",
            ])
            external_gaps = result.get("external_evidence_gaps", [])
            lines.extend(
                [f"- {gap}" for gap in external_gaps]
                if external_gaps
                else ["- None recorded."]
            )
            lines.extend(["", "## External Source Warnings", ""])
            external_warnings = result.get("external_retrieval_warnings", [])
            lines.extend(
                [f"- {warning}" for warning in external_warnings]
                if external_warnings
                else ["- None."]
            )
        lines.append("")
        lines.extend(["## Evidence Gaps", ""])
        if result["evidence_gaps"]:
            lines.extend(f"- {gap}" for gap in result["evidence_gaps"])
        else:
            lines.append("- No material gaps detected by the lightweight verifier.")
        lines.extend(["", "## Unsupported Claims", ""])
        if result["unsupported_claims"]:
            lines.extend(f"- {claim}" for claim in result["unsupported_claims"])
        else:
            lines.append("- None detected by the lightweight verifier.")
        if result["corpus_warnings"]:
            lines.extend(["", "### Corpus Warnings", ""])
            lines.extend(f"- {warning}" for warning in result["corpus_warnings"])
        lines.extend([
            "",
            "## Literature analysis",
            "",
            "### Existing methods",
            "",
            *[f"- {value}" for value in analysis["existing_methods"]],
            "",
            "### Key limitations",
            "",
            *[f"- {value}" for value in analysis["key_limitations"]],
            "",
            f"### Research gap\n\n{analysis['research_gap']}",
            f"\nStatus: `{analysis.get('research_gap_status', 'not_recorded')}`",
            (
                f"\nNote: {analysis['research_gap_note']}"
                if analysis.get("research_gap_note")
                else ""
            ),
            "",
            "## Candidate ideas",
            "",
        ])
        for index, idea in enumerate(result["candidate_ideas"], start=1):
            lines.extend([
                f"### {index}. {idea['title']}",
                "",
                f"- Hypothesis: {idea['hypothesis']}",
                f"- Method: {idea['method']}",
                f"- Rank score: {idea['rank_score']:.3f}",
                "",
            ])
        research_directions = result.get("research_directions", [])
        if research_directions:
            lines.extend(["## Candidate Research Directions", ""])
            for index, direction in enumerate(research_directions, start=1):
                source_index = direction.get("source_idea_index")
                supporting_evidence = (
                    ", ".join(direction.get("supporting_evidence", []))
                    or "None"
                )
                lines.extend([
                    f"### {index}. {direction['title']}",
                    "",
                    (
                        f"- Source Idea: "
                        f"{direction.get('source_idea_title') or 'No source idea'}"
                    ),
                    (
                        f"- Source Idea Index: "
                        f"{source_index if source_index is not None else 'N/A'}"
                    ),
                    f"- Target Gap: {direction['target_gap']}",
                    f"- Core Problem: {direction['core_problem']}",
                    f"- Hypothesis: {direction['hypothesis']}",
                    f"- Method Sketch: {direction['method_sketch']}",
                    (
                        f"- Evidence Support Level: "
                        f"{direction['evidence_support_level']}"
                    ),
                    f"- Novelty Risk: {direction['novelty_risk']}",
                    f"- Feasibility Risk: {direction['feasibility_risk']}",
                    (
                        f"- Recommended Priority: "
                        f"{direction['recommended_priority']}"
                    ),
                    (
                        f"- Assessment Status: "
                        f"{direction.get('assessment_status', 'not_recorded')}"
                    ),
                    f"- Supporting Evidence: {supporting_evidence}",
                    "",
                    "Next Steps:",
                    "",
                    *[
                        f"- {step}"
                        for step in direction.get("next_steps", [])
                    ],
                    "",
                ])
        selected_direction = result.get("selected_direction")
        if selected_direction:
            selected_source_index = selected_direction.get("source_idea_index")
            lines.extend([
                "## Selected Research Direction",
                "",
                f"- Title: {selected_direction['title']}",
                (
                    f"- Source Idea: "
                    f"{selected_direction.get('source_idea_title') or 'No source idea'}"
                ),
                (
                    f"- Source Idea Index: "
                    f"{selected_source_index if selected_source_index is not None else 'N/A'}"
                ),
                (
                    f"- Recommended Priority: "
                    f"{selected_direction['recommended_priority']}"
                ),
                (
                    f"- Assessment Status: "
                    f"{selected_direction.get('assessment_status', 'not_recorded')}"
                ),
                (
                    "- Rationale: Selected because it maps to the idea used for "
                    "experiment planning; its priority is assigned by a "
                    "deterministic planning heuristic."
                ),
                "",
            ])
        feasibility = result.get("feasibility_assessment")
        if feasibility:
            readiness_score = feasibility.get(
                "planning_readiness_score",
                feasibility.get("overall_score", 0.0),
            )
            lines.extend([
                "## Feasibility Assessment",
                "",
                (
                    f"- Planning Readiness Score: "
                    f"{readiness_score:.3f}"
                ),
                f"- Recommendation: {feasibility['recommendation']}",
                f"- Evidence Readiness: {feasibility['evidence_readiness']}",
                (
                    f"- Experiment Readiness: "
                    f"{feasibility['experiment_readiness']}"
                ),
                (
                    f"- Reproducibility Readiness: "
                    f"{feasibility['reproducibility_readiness']}"
                ),
                (
                    f"- Resource Requirement: "
                    f"{feasibility['resource_requirement']}"
                ),
                (
                    f"- Implementation Readiness: "
                    f"{feasibility['implementation_readiness']}"
                ),
                f"- Dataset Clarity: {feasibility['dataset_clarity']}",
                f"- Baseline Clarity: {feasibility['baseline_clarity']}",
                f"- Metric Clarity: {feasibility['metric_clarity']}",
                "",
                "### Main Risks",
                "",
                *[f"- {risk}" for risk in feasibility["main_risks"]],
                "",
                "### Mitigation Strategies",
                "",
                *[
                    f"- {strategy}"
                    for strategy in feasibility["mitigation_strategies"]
                ],
                "",
                "### Minimum Viable Experiment",
                "",
                *[
                    f"- {step}"
                    for step in feasibility["minimum_viable_experiment"]
                ],
                "",
                f"**Assessment Note:** {feasibility['assessment_note']}",
                "",
            ])
        blueprint = result.get("experiment_blueprint")
        if blueprint:
            blockers = blueprint.get("pre_execution_blockers", [])
            lines.extend([
                "## Experiment Blueprint",
                "",
                f"- Objective: {blueprint['objective']}",
                f"- Hypothesis: {blueprint['hypothesis']}",
                (
                    f"- Pilot Planning Ready: "
                    f"{blueprint['pilot_planning_ready']}"
                ),
                (
                    f"- Human Approval Required: "
                    f"{blueprint['human_approval_required']}"
                ),
                f"- Execution Allowed: {blueprint['execution_allowed']}",
                "",
                "### Pre-execution Blockers",
                "",
                *(
                    [f"- {blocker}" for blocker in blockers]
                    if blockers
                    else ["- None"]
                ),
                "",
                "### Minimum Viable Experiment",
                "",
                *[
                    f"- {step}"
                    for step in blueprint.get(
                        "minimum_viable_experiment",
                        [],
                    )
                ],
                "",
                "### Datasets",
                "",
                *[f"- {item}" for item in blueprint.get("datasets", [])],
                "",
                "### Baselines",
                "",
                *[f"- {item}" for item in blueprint.get("baselines", [])],
                "",
                "### Metrics",
                "",
                *[f"- {item}" for item in blueprint.get("metrics", [])],
                "",
                "### Ablations",
                "",
                *[f"- {item}" for item in blueprint.get("ablations", [])],
                "",
                "### Planning Artifacts",
                "",
                *[
                    f"- {item}"
                    for item in blueprint.get("planning_artifacts", [])
                ],
                "",
                "### Experiment Artifacts",
                "",
                *[
                    f"- {item}"
                    for item in blueprint.get("experiment_artifacts", [])
                ],
                "",
                "### Success Criteria",
                "",
                *[
                    f"- {item}"
                    for item in blueprint.get("success_criteria", [])
                ],
                "",
                "### Failure Criteria",
                "",
                *[
                    f"- {item}"
                    for item in blueprint.get("failure_criteria", [])
                ],
                "",
                "### Reproducibility Checklist",
                "",
                *[
                    f"- {item}"
                    for item in blueprint.get(
                        "reproducibility_checklist",
                        [],
                    )
                ],
                "",
                "### Pre-execution Checklist",
                "",
                *[
                    f"- {item}"
                    for item in blueprint.get(
                        "pre_execution_checklist",
                        [],
                    )
                ],
                "",
                f"**Blueprint Note:** {blueprint['blueprint_note']}",
                "",
            ])
        lines.extend([
            "## Selected idea",
            "",
            f"**{selected['title']}**",
            "",
            selected["hypothesis"],
            "",
            "## Experiment design",
            "",
            f"**Method:** {experiment['method']}",
            "",
        ])
        for field_name in (
            "datasets", "baselines", "metrics", "ablation",
            "expected_results", "risks", "implementation_notes",
        ):
            lines.extend([
                f"### {field_name.replace('_', ' ').title()}",
                "",
                *[f"- {value}" for value in experiment[field_name]],
                "",
            ])
        lines.extend(["## Verification", ""])
        for name, verification in result["verification"].items():
            lines.append(
                f"- **{name}**: {'PASS' if verification['passed'] else 'FAIL'} "
                f"(score={verification['score']:.2f})"
            )
            lines.extend(f"  - Issue: {issue}" for issue in verification["issues"])
            lines.extend(
                f"  - Warning: {warning}"
                for warning in verification.get("warnings", [])
            )
        agent_trace = result.get("agent_trace")
        if agent_trace:
            lines.extend(["", "## Agent Decision Trace", ""])
            for entry in agent_trace:
                lines.extend([
                    f"### Step {entry['step']}",
                    "",
                    f"- Observation: {entry['observation']}",
                    f"- Decision: {entry['decision']}",
                    f"- Action: {entry['action']}",
                    f"- Reason: {entry['reason']}",
                    f"- Result: {entry.get('result') or 'Not recorded.'}",
                    "",
                ])
        lines.extend([
            "",
            f"Revision performed: **{result['revision_performed']}**",
            "",
            "## Limitations",
            "",
            (
                "- Scientific verification is limited to local papers and saved "
                "scientific memory; external evidence does not automatically "
                "validate claims."
            ),
            "- Retrieval uses lightweight lexical overlap rather than semantic embeddings.",
            "- Idea generation and ranking use lightweight heuristics in this MVP.",
            "- Benchmark and baseline choices must be checked against the latest literature before execution.",
            "",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls._serialize(value.to_dict())
        if is_dataclass(value):
            return cls._serialize(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: cls._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._serialize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

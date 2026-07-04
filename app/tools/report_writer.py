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
            lines.extend([
                "## Feasibility Assessment",
                "",
                (
                    f"- Planning Readiness Score: "
                    f"{feasibility['overall_score']:.3f}"
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
            "- Evidence is limited to data/papers and saved scientific memory.",
            "- Retrieval uses lightweight lexical overlap rather than semantic embeddings.",
            "- Idea generation and ranking use lightweight heuristics in this MVP.",
            "- Benchmark and baseline choices must be checked against the latest literature before execution.",
            "",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @classmethod
    def _serialize(cls, value: Any) -> Any:
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

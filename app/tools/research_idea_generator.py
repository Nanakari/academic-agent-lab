"""Deterministic and optional LLM-assisted research idea generation."""

from __future__ import annotations

import json

from app.agent.services.llm_json import (
    first_present,
    limited_evidence_payload,
    list_of_strings,
    parse_json_value,
)
from app.agent.services.llm_scientific_analysis_service import LLMStageResult
from app.schemas.research_idea import ResearchIdea


class ResearchIdeaGenerator:
    """Generate a small, diverse idea set that remains testable in an MVP."""

    def __init__(self, llm=None, enabled: bool = False) -> None:
        self.llm = llm
        self.enabled = bool(enabled and llm is not None)
        self.last_stage_result = LLMStageResult(
            stage="idea_generation",
            deterministic_sections=["candidate_ideas"],
        )

    def generate_ideas(self, topic: str, evidence_context: list[dict]) -> list[ResearchIdea]:
        if self.enabled:
            try:
                ideas = self._generate_with_llm(topic, evidence_context)
                self.last_stage_result = LLMStageResult(
                    stage="idea_generation",
                    llm_used=True,
                    generated_sections=["candidate_ideas"],
                )
                return self.rank_ideas(ideas)
            except Exception as exc:
                self.last_stage_result = LLMStageResult(
                    stage="idea_generation",
                    fallback_used=True,
                    deterministic_sections=["candidate_ideas"],
                    warning=str(exc),
                )
        else:
            self.last_stage_result = LLMStageResult(
                stage="idea_generation",
                deterministic_sections=["candidate_ideas"],
            )
        return self._generate_deterministic(topic, evidence_context)

    def _generate_deterministic(
        self,
        topic: str,
        evidence_context: list[dict],
    ) -> list[ResearchIdea]:
        refs = [item.get("evidence_id", "") for item in evidence_context[:3] if item.get("evidence_id")]
        evidence_note = (
            evidence_context[0]["excerpt"][:240]
            if evidence_context
            else "No strong local evidence was retrieved; treat the hypothesis as exploratory."
        )
        ideas = [
            ResearchIdea(
                title=f"Evidence-aware adaptive intervention for {topic}",
                hypothesis=(
                    "Applying an intervention only when an evidence-grounded uncertainty signal "
                    "is high will improve reliability without uniformly increasing inference cost."
                ),
                motivation=f"Local evidence indicates an unresolved reliability gap: {evidence_note}",
                method=(
                    "Build a lightweight uncertainty detector, route high-risk cases through a "
                    "verification/intervention stage, and retain the original path for low-risk cases."
                ),
                evidence_refs=refs,
                required_evidence=refs,
                expected_experiment="Compare adaptive routing against always-on and no-verifier baselines.",
                risks=["The uncertainty detector may be poorly calibrated."],
                novelty_score=0.78,
                feasibility_score=0.84,
            ),
            ResearchIdea(
                title=f"Failure-mode curriculum for {topic}",
                hypothesis=(
                    "Training on automatically clustered failure modes will generalize better than "
                    "using an undifferentiated collection of hard examples."
                ),
                motivation="Existing aggregate evaluation can hide distinct and recurring failure modes.",
                method=(
                    "Cluster model failures by semantic and behavioral features, construct a balanced "
                    "curriculum, and compare it with random hard-example sampling."
                ),
                evidence_refs=refs,
                required_evidence=refs,
                expected_experiment="Evaluate clustered curriculum training against random hard-example sampling.",
                risks=["Failure clusters may reflect annotation artifacts."],
                novelty_score=0.73,
                feasibility_score=0.77,
            ),
            ResearchIdea(
                title=f"Counterfactual consistency benchmark for {topic}",
                hypothesis=(
                    "Paired counterfactual inputs can reveal reliability failures that standard "
                    "single-instance benchmarks underestimate."
                ),
                motivation="Static benchmark scores do not always expose causal sensitivity or shortcuts.",
                method=(
                    "Create controlled input pairs that preserve the correct answer while perturbing "
                    "irrelevant evidence, then measure prediction and rationale consistency."
                ),
                evidence_refs=refs,
                required_evidence=refs,
                expected_experiment="Measure consistency on paired counterfactual examples.",
                risks=["Counterfactual construction may introduce hidden label leakage."],
                novelty_score=0.81,
                feasibility_score=0.70,
            ),
        ]
        return self.rank_ideas(ideas)

    def _generate_with_llm(
        self,
        topic: str,
        evidence_context: list[dict],
    ) -> list[ResearchIdea]:
        payload = {
            "topic": topic,
            "evidence": limited_evidence_payload(evidence_context),
        }
        raw = self.llm.ask([
            {
                "role": "system",
                "content": (
                    "You generate conservative, testable AI research directions. "
                    "Return only strict JSON and use only the provided evidence. "
                    "Do not wrap JSON in Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON schema and no extra text:\n"
                    "{\n"
                    '  "ideas": [\n'
                    "    {\n"
                    '      "title": "short research direction title",\n'
                    '      "hypothesis": "testable hypothesis",\n'
                    '      "motivation": "why evidence suggests this matters",\n'
                    '      "method": "concrete proposed method",\n'
                    '      "required_evidence": ["E1"],\n'
                    '      "expected_experiment": "minimum experiment to test it",\n'
                    '      "risks": ["main risk"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "The ideas array must contain 3 to 5 complete items.\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ])
        parsed = parse_json_value(raw)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = first_present(
                parsed,
                ("ideas", "candidate_ideas", "research_ideas", "research_directions", "directions"),
            )
        else:
            records = None
        if not isinstance(records, list) or not 3 <= len(records) <= 5:
            raise ValueError("LLM idea generation must return 3 to 5 ideas.")
        refs = [
            item.get("evidence_id", "")
            for item in evidence_context[:3]
            if item.get("evidence_id")
        ]
        evidence_summary = (
            evidence_context[0].get("excerpt")
            or evidence_context[0].get("text")
            if evidence_context
            else "No strong local evidence was retrieved."
        )
        ideas = []
        for record in records[:5]:
            if not isinstance(record, dict):
                continue
            title = str(first_present(record, ("title", "name", "direction")) or "").strip()
            method = str(first_present(record, ("method", "approach", "method_sketch")) or "").strip()
            if not all((title, method)):
                continue
            hypothesis = str(
                first_present(record, ("hypothesis", "claim", "expected_effect"))
                or f"{title} should improve the target reliability or safety outcome."
            ).strip()
            motivation = str(
                first_present(record, ("motivation", "rationale", "why"))
                or f"Evidence suggests a scoped gap: {str(evidence_summary)[:220]}"
            ).strip()
            required_evidence = list_of_strings(
                first_present(record, ("required_evidence", "evidence", "evidence_refs")),
                default=refs,
            )
            ideas.append(
                ResearchIdea(
                    title=title,
                    hypothesis=hypothesis,
                    motivation=motivation,
                    method=method,
                    evidence_refs=[
                        item for item in required_evidence if item in set(refs)
                    ]
                    or refs,
                    required_evidence=required_evidence,
                    expected_experiment=str(
                        first_present(record, ("expected_experiment", "experiment", "evaluation"))
                        or "Run a controlled comparison against matched baselines."
                    ).strip(),
                    risks=list_of_strings(
                        first_present(record, ("risks", "limitations", "failure_modes")),
                        default=["Requires stronger evidence before strong claims."],
                    ),
                    novelty_score=0.76,
                    feasibility_score=0.74,
                )
            )
        if len(ideas) < 3:
            raise ValueError("LLM idea generation returned too few complete ideas.")
        return ideas

    def rank_ideas(self, ideas: list[ResearchIdea]) -> list[ResearchIdea]:
        for idea in ideas:
            evidence_bonus = min(len(idea.evidence_refs) * 0.03, 0.09)
            idea.rank_score = round(
                0.55 * idea.novelty_score + 0.45 * idea.feasibility_score + evidence_bonus,
                3,
            )
        return sorted(ideas, key=lambda item: item.rank_score, reverse=True)

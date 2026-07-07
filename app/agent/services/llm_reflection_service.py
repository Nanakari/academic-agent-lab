"""Conservative LLM reflection after verifier results."""

from __future__ import annotations

import json

from app.agent.services.llm_json import list_of_strings, parse_json_object
from app.agent.services.llm_scientific_analysis_service import LLMStageResult


class LLMReflectionService:
    """Ask an LLM for cautionary notes without changing verifier outcomes."""

    def __init__(self, llm=None, enabled: bool = False) -> None:
        self.llm = llm
        self.enabled = bool(enabled and llm is not None)

    def reflect(
        self,
        *,
        verification: dict,
        evidence_status: str,
        selected_idea: dict,
        experiment_plan: dict,
    ) -> tuple[dict, LLMStageResult]:
        if not self.enabled:
            return {}, LLMStageResult(
                stage="reflection",
                deterministic_sections=["reflection"],
            )
        try:
            payload = {
                "verification": verification,
                "evidence_status": evidence_status,
                "selected_idea": selected_idea,
                "experiment_plan": experiment_plan,
                "constraints": [
                    "Do not convert failed verifier checks into success.",
                    "Only lower conclusion strength or suggest missing evidence.",
                    "If evidence_status is memory_only, include conservative_revision_notes.",
                    "If local memory overlap max_similarity is >= 0.95, include conservative_revision_notes.",
                ],
            }
            raw = self.llm.ask([
                {
                    "role": "system",
                    "content": (
                        "You are a conservative scientific reviewer. Return "
                        "only JSON. You may not overrule verifier failures."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with keys: conservative_revision_notes, "
                        "missing_evidence_suggestions, conclusion_strength, "
                        "risk_warnings.\n"
                        + json.dumps(payload, ensure_ascii=False)
                    ),
                },
            ])
            parsed = parse_json_object(raw)
            reflection = {
                "conservative_revision_notes": list_of_strings(
                    parsed.get("conservative_revision_notes")
                ),
                "missing_evidence_suggestions": list_of_strings(
                    parsed.get("missing_evidence_suggestions")
                ),
                "conclusion_strength": str(
                    parsed.get("conclusion_strength") or "bounded_pre_experiment"
                ),
                "risk_warnings": list_of_strings(parsed.get("risk_warnings")),
                "mode": "llm_assisted",
            }
            if self._requires_conservative_notes(
                verification,
                evidence_status,
            ) and not reflection["conservative_revision_notes"]:
                raise ValueError(
                    "Reflection must include conservative_revision_notes for weak evidence."
                )
            return reflection, LLMStageResult(
                stage="reflection",
                llm_used=True,
                generated_sections=["reflection"],
            )
        except Exception as exc:
            return self._deterministic_reflection(
                verification=verification,
                evidence_status=evidence_status,
                warning=str(exc),
            )

    def _deterministic_reflection(
        self,
        *,
        verification: dict,
        evidence_status: str,
        warning: str | None,
    ) -> tuple[dict, LLMStageResult]:
        notes = []
        if evidence_status == "memory_only":
            notes.append(
                "Evidence is memory-only; collect matching local-paper evidence before strong claims."
            )
        novelty = verification.get("novelty", {})
        try:
            similarity = float(
                novelty.get("local_memory_overlap", {}).get("max_similarity", 0.0)
            )
        except (TypeError, ValueError):
            similarity = 0.0
        if similarity >= 0.95:
            notes.append(
                "Local draft overlap is >= 0.95; confirm this is not a duplicate proposal."
            )
        return {
            "conservative_revision_notes": notes,
            "missing_evidence_suggestions": [],
            "conclusion_strength": "bounded_pre_experiment",
            "risk_warnings": [],
            "mode": "deterministic",
        }, LLMStageResult(
            stage="reflection",
            fallback_used=warning is not None,
            deterministic_sections=["reflection"],
            warning=warning,
        )

    @staticmethod
    def _requires_conservative_notes(
        verification: dict,
        evidence_status: str,
    ) -> bool:
        if evidence_status == "memory_only":
            return True
        novelty = verification.get("novelty", {})
        try:
            similarity = float(
                novelty.get("local_memory_overlap", {}).get("max_similarity", 0.0)
            )
        except (TypeError, ValueError):
            similarity = 0.0
        return similarity >= 0.95

"""LLM-assisted, verifier-bounded scientific analysis stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.services.literature_analysis_service import LiteratureAnalysisService
from app.agent.services.llm_json import (
    limited_evidence_payload,
    list_of_strings,
    parse_json_object,
)


@dataclass
class LLMStageResult:
    """Metadata for an optional LLM stage."""

    stage: str
    llm_used: bool = False
    fallback_used: bool = False
    generated_sections: list[str] = field(default_factory=list)
    deterministic_sections: list[str] = field(default_factory=list)
    warning: str | None = None


class LLMScientificAnalysisService:
    """Use an LLM to structure literature analysis, with deterministic fallback."""

    REQUIRED_FIELDS = (
        "existing_methods",
        "attack_models",
        "defense_strategies",
        "evaluation_protocols",
        "limitations",
        "research_gaps",
        "evidence_confidence",
        "missing_evidence",
    )

    def __init__(
        self,
        deterministic_service: LiteratureAnalysisService,
        llm=None,
        enabled: bool = False,
    ) -> None:
        self.deterministic_service = deterministic_service
        self.llm = llm
        self.enabled = bool(enabled and llm is not None)

    def analyze(
        self,
        *,
        evidence_context: list[dict],
        external_literature_evidence: list[dict],
        topic: str,
        fallback_evidence_context: list[dict] | None = None,
    ) -> tuple[dict, LLMStageResult]:
        if not self.enabled:
            return self._fallback(
                fallback_evidence_context or evidence_context,
                warning=None,
            )
        try:
            payload = {
                "topic": topic,
                "local_evidence": limited_evidence_payload(evidence_context),
                "external_literature_evidence": limited_evidence_payload(
                    external_literature_evidence
                ),
            }
            raw = self.llm.ask(self._messages(payload))
            parsed = parse_json_object(raw)
            analysis = self._normalize(parsed)
            return analysis, LLMStageResult(
                stage="literature_analysis",
                llm_used=True,
                generated_sections=list(self.REQUIRED_FIELDS),
            )
        except Exception as exc:
            return self._fallback(
                fallback_evidence_context or evidence_context,
                warning=str(exc),
            )

    def _fallback(
        self,
        evidence_context: list[dict],
        *,
        warning: str | None,
    ) -> tuple[dict, LLMStageResult]:
        analysis = self.deterministic_service.analyze(evidence_context)
        deterministic_sections = [
            "existing_methods",
            "key_limitations",
            "research_gap",
        ]
        return analysis, LLMStageResult(
            stage="literature_analysis",
            fallback_used=warning is not None,
            deterministic_sections=deterministic_sections,
            warning=warning,
        )

    def _normalize(self, parsed: dict[str, Any]) -> dict:
        existing_methods = list_of_strings(parsed.get("existing_methods"))
        limitations = list_of_strings(parsed.get("limitations"))
        research_gaps = list_of_strings(parsed.get("research_gaps"))
        missing_evidence = list_of_strings(parsed.get("missing_evidence"))
        if not existing_methods:
            existing_methods = ["No relevant method was found in the provided evidence."]
        if not limitations:
            limitations = ["Limitations were not explicitly identified by the LLM."]
        if not research_gaps:
            research_gaps = [
                "A defensible research gap cannot be established from the provided evidence."
            ]
        confidence = str(parsed.get("evidence_confidence") or "unknown").strip()
        status = (
            "evidence_supported"
            if confidence.casefold() in {"high", "moderate", "medium"}
            and not missing_evidence
            else "insufficient_evidence"
        )
        return {
            "existing_methods": existing_methods,
            "attack_models": list_of_strings(parsed.get("attack_models")),
            "defense_strategies": list_of_strings(parsed.get("defense_strategies")),
            "evaluation_protocols": list_of_strings(
                parsed.get("evaluation_protocols")
            ),
            "key_limitations": limitations,
            "limitations": limitations,
            "research_gap": research_gaps[0],
            "research_gaps": research_gaps,
            "research_gap_status": status,
            "research_gap_note": (
                "LLM-assisted literature analysis; verifier checks still bound "
                "claim support."
            ),
            "evidence_confidence": confidence,
            "missing_evidence": missing_evidence,
            "analysis_mode": "llm_assisted",
        }

    @classmethod
    def _messages(cls, payload: dict) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a conservative scientific-analysis assistant. "
                    "Return only JSON. Use only the supplied evidence; do not "
                    "invent papers, results, or benchmarks."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Analyze this scientific evidence and return a JSON object "
                    "with exactly these keys: "
                    + ", ".join(cls.REQUIRED_FIELDS)
                    + ". evidence_confidence must be one of high, moderate, "
                    "low, unknown.\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ]

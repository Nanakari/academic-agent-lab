"""Structured literature analysis over retrieved evidence."""

from __future__ import annotations

from app.tools.paper_analyzer import PaperAnalyzer
from app.verifier.claim_filter import is_verifiable_claim


class LiteratureAnalysisService:
    """Extract methods, limitations, and an evidence-aware research gap."""

    def __init__(self, paper_analyzer: PaperAnalyzer) -> None:
        self.paper_analyzer = paper_analyzer

    def analyze(self, evidence_context: list[dict]) -> dict:
        """Return the literature-analysis contract used by the agent."""
        if not evidence_context:
            return {
                "existing_methods": ["No relevant method was found in local evidence."],
                "key_limitations": ["Local evidence coverage is insufficient."],
                "research_gap": (
                    "A defensible gap cannot yet be established from local papers; "
                    "the generated ideas must be treated as exploratory."
                ),
                "research_gap_status": "insufficient_evidence",
                "research_gap_note": (
                    "Retrieved evidence did not explicitly state a concrete limitation."
                ),
            }

        combined = "\n".join(item["excerpt"] for item in evidence_context)
        extracted = self.paper_analyzer.extract_problem_method_experiment_limitation(
            combined
        )
        method_sections = self._evidence_from_sections(
            evidence_context,
            {"method", "methods", "methodology", "approach"},
        )
        limitation_sections = self._evidence_from_sections(
            evidence_context,
            {"limitation", "limitations", "future work"},
        )
        existing_methods = method_sections or extracted["method"]
        limitations = limitation_sections or extracted["limitation"]
        if not limitations or not is_verifiable_claim(limitations[0]):
            return {
                "existing_methods": existing_methods,
                "key_limitations": limitations,
                "research_gap": (
                    "A defensible research gap cannot be established from the "
                    "retrieved evidence."
                ),
                "research_gap_status": "insufficient_evidence",
                "research_gap_note": (
                    "Retrieved evidence did not explicitly state a concrete limitation."
                ),
            }
        return {
            "existing_methods": existing_methods,
            "key_limitations": limitations,
            "research_gap": (
                "The retrieved evidence describes existing methods but leaves "
                f"unresolved: {limitations[0]}"
            ),
            "research_gap_status": "evidence_supported",
        }

    @staticmethod
    def _evidence_from_sections(
        evidence_context: list[dict],
        section_names: set[str],
    ) -> list[str]:
        """Prefer explicit section metadata over cue matching in chunk text."""
        return [
            item["text"]
            for item in evidence_context
            if str(item.get("section") or "").casefold() in section_names
        ][:3]

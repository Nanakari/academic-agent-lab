"""Persistence of scientific-agent run artifacts."""

from __future__ import annotations

from app.memory.scientific_memory import ScientificMemory
from app.tools.paper_analyzer import PaperAnalyzer


class PersistenceService:
    """Persist paper notes, selected ideas, experiments, and verification logs."""

    def __init__(
        self,
        memory: ScientificMemory,
        paper_analyzer: PaperAnalyzer,
    ) -> None:
        self.memory = memory
        self.paper_analyzer = paper_analyzer

    def save(self, result: dict) -> None:
        for evidence in result["evidence_context"]:
            if evidence["kind"] != "local_paper":
                continue
            self.memory.save_paper_note(
                {
                    "topic": result["topic"],
                    "source": evidence["source"],
                    "evidence_id": evidence["evidence_id"],
                    "page": evidence["page"],
                    "section": evidence["section"],
                    "file_type": evidence["file_type"],
                    "chunk_id": evidence["chunk_id"],
                    "summary": self.paper_analyzer.summarize_paper_text(
                        evidence["excerpt"],
                        max_sentences=2,
                    ),
                }
            )
        self.memory.save_idea(
            {"topic": result["topic"], **result["selected_idea"]}
        )
        self.memory.save_experiment(
            {"topic": result["topic"], **result["experiment_plan"]}
        )
        self.memory.save_verification_log(
            {
                "topic": result["topic"],
                "passed": result["verification_passed"],
                "results": result["verification"],
            }
        )

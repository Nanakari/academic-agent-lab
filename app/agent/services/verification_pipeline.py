"""Unified verifier execution for the AI Scientific Agent."""

from __future__ import annotations

from app.verifier.evidence_verifier import EvidenceVerifier
from app.verifier.experiment_verifier import ExperimentVerifier
from app.verifier.novelty_verifier import NoveltyVerifier
from app.verifier.reproducibility_verifier import ReproducibilityVerifier


class VerificationPipeline:
    """Run the four lightweight verifiers and build evidence summaries."""

    def __init__(self) -> None:
        self.evidence_verifier = EvidenceVerifier()
        self.novelty_verifier = NoveltyVerifier()
        self.experiment_verifier = ExperimentVerifier()
        self.reproducibility_verifier = ReproducibilityVerifier()

    def verify(
        self,
        idea,
        experiment_plan,
        evidence_context: list[dict],
        literature_analysis: dict,
        history: list[dict],
        ideas: list,
    ) -> dict:
        results = {
            "evidence": self.evidence_verifier.verify(
                idea,
                evidence_context,
                claims=[
                    literature_analysis["research_gap"],
                    *literature_analysis["existing_methods"],
                ],
                ideas=ideas,
            ),
            "novelty": self.novelty_verifier.verify(idea, history),
            "experiment": self.experiment_verifier.verify(experiment_plan),
            "reproducibility": self.reproducibility_verifier.verify(experiment_plan),
        }
        return {name: result.to_dict() for name, result in results.items()}

    @staticmethod
    def build_evidence_assessment(
        evidence_context: list[dict],
        evidence_verification: dict,
    ) -> dict:
        used = [
            {
                "evidence_id": item["evidence_id"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "source_path": item["source_path"],
                "file_type": item["file_type"],
                "page": item["page"],
                "section": item["section"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "matched_keywords": item["matched_keywords"],
                "supporting_claim": item["supporting_claim"],
                "support_level": item["support_level"],
                "kind": item["kind"],
            }
            for item in evidence_context
        ]
        gaps = []
        if not any(item["kind"] == "local_paper" for item in evidence_context):
            gaps.append("No matching evidence was retrieved from the local paper corpus.")
        gaps.extend(evidence_verification["issues"])
        return {
            "status": (
                "sufficient"
                if evidence_verification["passed"]
                else "evidence_insufficient"
            ),
            "used": used,
            "gaps": list(dict.fromkeys(gaps)),
            "unsupported_claims": evidence_verification["unsupported_claims"],
        }

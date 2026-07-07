"""Unified verifier execution for the AI Scientific Agent."""

from __future__ import annotations

from app.verifier.claim_filter import is_verifiable_claim
from app.verifier.evidence_verifier import EvidenceVerifier
from app.verifier.experiment_verifier import ExperimentVerifier
from app.verifier.novelty_verifier import NoveltyVerifier
from app.verifier.reproducibility_verifier import ReproducibilityVerifier


class VerificationPipeline:
    """Run the four lightweight verifiers and build evidence summaries."""

    def __init__(
        self,
        strict_domain: bool | None = None,
        domain_mode: str = "off",
    ) -> None:
        self.evidence_verifier = EvidenceVerifier(
            domain_mode=domain_mode,
            strict_domain=strict_domain,
        )
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
        topic: str | None = None,
        external_literature_evidence: list[dict] | None = None,
    ) -> dict:
        claims = list(literature_analysis.get("existing_methods", []))
        if literature_analysis.get("research_gap_status") not in {
            "insufficient_evidence",
            "insufficient_topic_relevant_evidence",
            "evidence_insufficient",
        }:
            claims.insert(0, literature_analysis.get("research_gap", ""))
        claims = [claim for claim in claims if is_verifiable_claim(claim)]
        results = {
            "evidence": self.evidence_verifier.verify(
                idea,
                evidence_context,
                claims=claims,
                ideas=ideas,
                topic=topic,
            ),
            "novelty": self.novelty_verifier.verify(
                idea,
                history,
                literature_analysis=literature_analysis,
                evidence_context=evidence_context,
                external_literature_evidence=external_literature_evidence,
            ),
            "experiment": self.experiment_verifier.verify(experiment_plan),
            "reproducibility": self.reproducibility_verifier.verify(experiment_plan),
        }
        return {name: result.to_dict() for name, result in results.items()}

    @staticmethod
    def build_evidence_assessment(
        evidence_context: list[dict],
        evidence_verification: dict,
        literature_analysis: dict | None = None,
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
        has_local_paper = any(
            item["kind"] == "local_paper" for item in evidence_context
        )
        has_scientific_memory = any(
            item["kind"] == "scientific_memory" for item in evidence_context
        )
        if not has_local_paper:
            gaps.append("No matching evidence was retrieved from the local paper corpus.")
        if (
            literature_analysis
            and literature_analysis.get("research_gap_status") in {
                "insufficient_evidence",
                "insufficient_topic_relevant_evidence",
                "evidence_insufficient",
            }
        ):
            gaps.append(
                "Research gap could not be established from retrieved evidence."
        )
        gaps.extend(evidence_verification["issues"])
        if not evidence_verification["passed"]:
            status = "evidence_insufficient"
        elif not has_local_paper and has_scientific_memory:
            status = "memory_only"
        elif not has_local_paper:
            status = "weakly_supported"
        else:
            status = "sufficient"
        return {
            "status": status,
            "used": used,
            "gaps": list(dict.fromkeys(gaps)),
            "unsupported_claims": evidence_verification["unsupported_claims"],
        }

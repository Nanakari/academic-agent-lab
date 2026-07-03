"""Deterministic, replaceable research idea generation."""

from app.schemas.research_idea import ResearchIdea


class ResearchIdeaGenerator:
    """Generate a small, diverse idea set that remains testable in an MVP."""

    def __init__(self, llm=None) -> None:
        # The interface accepts an LLM for future structured generation; the MVP is offline.
        self.llm = llm

    def generate_ideas(self, topic: str, evidence_context: list[dict]) -> list[ResearchIdea]:
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
                novelty_score=0.81,
                feasibility_score=0.70,
            ),
        ]
        return self.rank_ideas(ideas)

    def rank_ideas(self, ideas: list[ResearchIdea]) -> list[ResearchIdea]:
        for idea in ideas:
            evidence_bonus = min(len(idea.evidence_refs) * 0.03, 0.09)
            idea.rank_score = round(
                0.55 * idea.novelty_score + 0.45 * idea.feasibility_score + evidence_bonus,
                3,
            )
        return sorted(ideas, key=lambda item: item.rank_score, reverse=True)

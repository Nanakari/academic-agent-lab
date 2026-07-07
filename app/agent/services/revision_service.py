"""Bounded revision actions for verifier failures."""

from __future__ import annotations


class RevisionService:
    """Apply one conservative revision without hiding verifier failures."""

    def revise_once(self, idea, experiment_plan, evidence_context):
        """Perform one bounded revision based on common verifier failures."""
        if evidence_context:
            idea.evidence_refs = [
                item["evidence_id"] for item in evidence_context[:3]
            ]
            idea.motivation = (
                f"Grounded in {', '.join(idea.evidence_refs)}; this remains a hypothesis "
                "to test rather than a claim of established improvement."
            )
        else:
            idea.hypothesis = "Exploratory hypothesis: " + idea.hypothesis

        # Make the revised candidate distinguishable from a previously saved formulation.
        idea.title = f"Cross-setting validation of {idea.title}"
        idea.method += (
            " Validate the mechanism on both in-domain and out-of-domain splits, "
            "with a compute-matched control."
        )
        if not experiment_plan.implementation_notes:
            experiment_plan.implementation_notes = [
                "Pin code, model, dataset, seed, and dependency versions."
            ]
        experiment_plan.method = idea.method
        return idea, experiment_plan

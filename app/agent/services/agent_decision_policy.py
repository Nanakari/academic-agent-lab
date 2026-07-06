"""Bounded, rule-based decisions for the AI Scientific Agent."""

from __future__ import annotations

from app.schemas.agent_trace import AgentTraceEntry


class AgentDecisionPolicy:
    """Describe verifier-driven decisions without changing workflow state."""

    _REVISION_PRIORITY = (
        "evidence",
        "experiment",
        "reproducibility",
        "novelty",
    )

    def decide_after_external_retrieval(
        self,
        *,
        step: int,
        sources_requested: list[str],
        queries_used: dict[str, str],
        evidence_items: list,
        warnings: list[str],
        cache_used: bool,
    ) -> AgentTraceEntry:
        counts = {
            source: sum(
                1
                for item in evidence_items
                if (
                    getattr(item, "source_type", "") == source
                    or (
                        source == "github"
                        and getattr(item, "source_type", "") == "github_repo"
                    )
                )
            )
            for source in sources_requested
        }
        count_text = ", ".join(
            f"{source}={counts[source]}" for source in sources_requested
        ) or "no sources requested"
        return AgentTraceEntry(
            step=step,
            observation=(
                "External search was enabled for "
                f"{', '.join(sources_requested) or 'no valid sources'}; "
                f"queries={queries_used}."
            ),
            decision="use_external_evidence_as_supplement",
            reason=(
                "arXiv metadata may extend literature discovery, while GitHub "
                "repositories are restricted to implementation evidence."
            ),
            action="retrieve_external_metadata",
            result=(
                f"Retrieved {count_text}; cache_used={cache_used}; "
                f"warnings={len(warnings)}. External evidence does not replace "
                "local scientific verification."
            ),
        )

    def decide_after_evidence(
        self,
        *,
        step: int,
        topic: str,
        evidence_context: list[dict],
        literature_analysis: dict,
    ) -> AgentTraceEntry:
        if not evidence_context:
            return AgentTraceEntry(
                step=step,
                observation="No local evidence was retrieved for the topic.",
                decision="mark_insufficient_evidence",
                reason=(
                    "The local corpus does not provide enough evidence for "
                    "grounded planning."
                ),
                action="generate_exploratory_idea",
                result=(
                    "The following idea and experiment plan must be treated as "
                    "exploratory."
                ),
            )

        if literature_analysis.get("research_gap_status") == "insufficient_evidence":
            return AgentTraceEntry(
                step=step,
                observation=(
                    "Evidence was retrieved, but no explicit research gap was "
                    "established."
                ),
                decision="downgrade_gap_confidence",
                reason=(
                    "Retrieved evidence supports the general topic but does not "
                    "directly establish a defensible research gap."
                ),
                action="continue_with_exploratory_planning",
                result=(
                    "Generated ideas should not be described as strongly "
                    "evidence-supported."
                ),
            )

        return AgentTraceEntry(
            step=step,
            observation=(
                "Retrieved evidence supports a concrete limitation or research gap."
            ),
            decision="continue_grounded_planning",
            reason="Local evidence is sufficient for bounded pre-experiment planning.",
            action="generate_candidate_ideas",
            result="Proceeding to idea generation and experiment planning.",
        )

    def decide_before_revision(
        self,
        *,
        step: int,
        verification: dict,
    ) -> AgentTraceEntry:
        """Record whether the first verifier pass warrants one revision."""
        failures = self._failed_verifiers(verification)
        if not failures:
            return AgentTraceEntry(
                step=step,
                observation="All verifier checks passed before revision.",
                decision="skip_revision",
                reason=(
                    "No verifier failure was detected, so bounded revision is "
                    "not needed."
                ),
                action="continue_to_final_assessment",
                result=(
                    "The agent will continue to final evidence assessment "
                    "without revision."
                ),
            )

        primary = self._primary_failure(failures)
        failed_names = ", ".join(failures)
        observation = (
            f"Verifier checks failed before revision: {failed_names}. "
            f"Primary trigger: {primary}. "
            f"Failure details: {self._failure_details(failures, verification)}"
        )
        return AgentTraceEntry(
            step=step,
            observation=observation,
            decision="trigger_bounded_revision",
            reason=self._revision_reason(primary, verification[primary]),
            action="perform_bounded_revision",
            result=(
                "The agent will perform at most one bounded revision. "
                f"The primary verifier concern is {self._revision_target(primary)}; "
                "any remaining failure will be preserved in the final report."
            ),
        )

    def decide_after_verification(
        self,
        *,
        step: int,
        verification: dict,
        revision_performed: bool,
    ) -> AgentTraceEntry:
        evidence = verification.get("evidence", {})
        experiment = verification.get("experiment", {})
        reproducibility = verification.get("reproducibility", {})

        if not evidence.get("passed", False):
            return AgentTraceEntry(
                step=step,
                observation=self._failed_check_observation("evidence", evidence),
                decision="revise_or_mark_evidence_gap",
                reason=(
                    "The evidence verifier failed, so its evidence gap must remain "
                    "visible."
                ),
                action=(
                    "keep_failure_and_report_gap"
                    if revision_performed
                    else "bounded_revision"
                ),
                result=self._revision_result(revision_performed, failure=True),
            )

        if not experiment.get("passed", False):
            missing = self._missing_experiment_details(experiment)
            return AgentTraceEntry(
                step=step,
                observation=self._failed_check_observation("experiment", experiment),
                decision="repair_experiment_plan",
                reason=f"The experiment plan is missing {missing}.",
                action="add_missing_experiment_fields",
                result=self._revision_result(revision_performed, failure=True),
            )

        if not reproducibility.get("passed", False):
            return AgentTraceEntry(
                step=step,
                observation=self._failed_check_observation(
                    "reproducibility",
                    reproducibility,
                ),
                decision="improve_reproducibility_notes",
                reason=(
                    "The reproducibility verifier found missing seed, version, "
                    "environment, or execution details."
                ),
                action="add_seed_version_or_environment_notes",
                result=self._revision_result(revision_performed, failure=True),
            )

        failed_other = [
            name
            for name, result in verification.items()
            if not result.get("passed", False)
        ]
        if failed_other:
            checks = ", ".join(failed_other)
            return AgentTraceEntry(
                step=step,
                observation=f"Verifier checks still failing: {checks}.",
                decision="preserve_verifier_failure",
                reason=(
                    "A failed verifier result must be reported rather than silently "
                    "changed to passed."
                ),
                action="keep_failure_and_report_gap",
                result=self._revision_result(revision_performed, failure=True),
            )

        return AgentTraceEntry(
            step=step,
            observation="All evidence, novelty, experiment, and reproducibility checks passed.",
            decision="accept_current_plan",
            reason="All bounded verifier checks passed for the current plan.",
            action="prepare_report",
            result=self._revision_result(revision_performed, failure=False),
        )

    def decide_before_report(
        self,
        *,
        step: int,
        evidence_status: str,
        verification_passed: bool,
        selected_idea: dict,
    ) -> AgentTraceEntry:
        title = selected_idea.get("title", "the selected idea")
        if verification_passed and evidence_status == "sufficient":
            return AgentTraceEntry(
                step=step,
                observation=(
                    f"Evidence is sufficient and all verifier checks passed for "
                    f"'{title}'."
                ),
                decision="recommend_as_pre_experiment_plan",
                reason=(
                    "Evidence and verifier checks are sufficient for a "
                    "planning-stage recommendation."
                ),
                action="write_grounded_report",
                result=(
                    "The report presents the selected idea as a pre-experiment "
                    "plan, not a proven result."
                ),
            )

        return AgentTraceEntry(
            step=step,
            observation=(
                f"Evidence status is '{evidence_status}' and verification_passed "
                f"is {verification_passed} for '{title}'."
            ),
            decision="report_as_exploratory_or_insufficient",
            reason=(
                "The agent should not overstate conclusions when evidence or "
                "verifier checks are insufficient."
            ),
            action="write_cautious_report",
            result=(
                "The final report preserves evidence gaps and failed verifier "
                "results."
            ),
        )

    @staticmethod
    def _failed_check_observation(name: str, result: dict) -> str:
        issues = result.get("issues", [])
        detail = "; ".join(issues) if issues else "No issue detail was provided."
        return f"The {name} verifier failed: {detail}"

    @staticmethod
    def _failed_verifiers(verification: dict) -> list[str]:
        return [
            name
            for name, result in verification.items()
            if not result.get("passed", False)
        ]

    @classmethod
    def _primary_failure(cls, failures: list[str]) -> str:
        for name in cls._REVISION_PRIORITY:
            if name in failures:
                return name
        return failures[0]

    @staticmethod
    def _failure_details(failures: list[str], verification: dict) -> str:
        details = []
        for name in failures:
            issues = verification[name].get("issues", [])
            issue_text = "; ".join(issues) if issues else "No issue detail provided."
            details.append(f"{name}: {issue_text}")
        return " | ".join(details)

    def _revision_reason(self, primary: str, result: dict) -> str:
        if primary == "evidence":
            return (
                "The evidence verifier failed, so the bounded revision should "
                "improve evidence grounding without claiming unsupported results."
            )
        if primary == "experiment":
            missing = self._missing_experiment_details(result)
            return (
                f"The experiment plan has missing or incomplete {missing}, so "
                "it is the primary bounded revision target."
            )
        if primary == "reproducibility":
            return (
                "The reproducibility verifier failed, so seed, dependency "
                "version, environment, logging, or execution notes need attention."
            )
        if primary == "novelty":
            return (
                "The novelty verifier failed because the current idea overlaps "
                "with saved or candidate ideas, so the revision should reduce "
                "duplication."
            )
        return (
            f"An unclassified verifier failure ({primary}) was detected and "
            "must remain visible through bounded revision and reporting."
        )

    @staticmethod
    def _revision_target(primary: str) -> str:
        targets = {
            "evidence": "evidence grounding",
            "experiment": "experiment-plan completeness",
            "reproducibility": "reproducibility notes",
            "novelty": "idea differentiation",
        }
        return targets.get(primary, f"the {primary} verifier failure")

    @staticmethod
    def _missing_experiment_details(result: dict) -> str:
        issue_text = " ".join(result.get("issues", [])).casefold()
        labels = {
            "baseline": ("baseline", "baselines"),
            "metric": ("metric", "metrics"),
            "ablation": ("ablation",),
            "implementation detail": (
                "implementation",
                "execution detail",
                "output format",
            ),
            "dataset": ("dataset", "datasets"),
            "risk analysis": ("risk", "risks"),
        }
        missing = [
            label
            for label, markers in labels.items()
            if any(marker in issue_text for marker in markers)
        ]
        return ", ".join(missing) if missing else "required experiment details"

    @staticmethod
    def _revision_result(revision_performed: bool, *, failure: bool) -> str:
        messages = []
        if revision_performed:
            messages.append("A bounded revision was performed once.")
        if failure:
            messages.append("The failure is preserved in the final report.")
        elif not messages:
            messages.append("The current plan was accepted without revision.")
        return " ".join(messages)

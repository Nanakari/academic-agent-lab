"""Offline heuristic planning for evidence-aware research directions."""

from __future__ import annotations

from app.schemas.evidence import SUPPORT_LEVELS
from app.schemas.research_direction import ResearchDirection


class ResearchDirectionService:
    """Transform candidate ideas into conservative planning-stage directions."""

    def plan(
        self,
        *,
        topic: str,
        literature_analysis: dict,
        candidate_ideas: list,
        selected_idea,
        evidence_context: list[dict],
        verification: dict | None = None,
    ) -> tuple[list[ResearchDirection], ResearchDirection]:
        selected_title = self._value(selected_idea, "title")
        directions = self.generate_directions(
            topic=topic,
            literature_analysis=literature_analysis,
            candidate_ideas=candidate_ideas,
            selected_idea_title=selected_title,
            evidence_context=evidence_context,
            verification=verification,
        )
        return directions, self.select_direction(
            directions,
            selected_idea_title=selected_title,
        )

    def generate_directions(
        self,
        *,
        topic: str,
        literature_analysis: dict,
        candidate_ideas: list,
        selected_idea_title: str | None,
        evidence_context: list[dict],
        verification: dict | None = None,
    ) -> list[ResearchDirection]:
        if not candidate_ideas:
            return [
                self._fallback_direction(
                    topic,
                    literature_analysis,
                    evidence_context,
                )
            ]

        return [
            self._direction_from_idea(
                topic=topic,
                literature_analysis=literature_analysis,
                idea=idea,
                evidence_context=evidence_context,
                verification=(
                    verification
                    if self._value(idea, "title") == selected_idea_title
                    else None
                ),
            )
            for idea in candidate_ideas
        ]

    @staticmethod
    def select_direction(
        directions: list[ResearchDirection],
        *,
        selected_idea_title: str | None,
    ) -> ResearchDirection:
        """Keep direction selection aligned with the idea used by the experiment."""
        if not directions:
            raise ValueError("directions must not be empty")
        for direction in directions:
            if direction.source_idea_title == selected_idea_title:
                return direction
        if selected_idea_title is not None:
            raise ValueError(
                "No research direction maps to the selected idea: "
                f"{selected_idea_title}"
            )
        return directions[0]

    def _direction_from_idea(
        self,
        *,
        topic: str,
        literature_analysis: dict,
        idea,
        evidence_context: list[dict],
        verification: dict | None,
    ) -> ResearchDirection:
        title = self._value(idea, "title") or f"Exploratory direction for {topic}"
        evidence_refs = list(self._value(idea, "evidence_refs") or [])
        supporting_evidence = evidence_refs or self._evidence_ids(evidence_context)
        is_verified_selected = verification is not None
        evidence_level = self._evidence_support_level(
            evidence_context,
            verification,
        )
        novelty_risk = self._novelty_risk(verification)
        feasibility_risk = self._feasibility_risk(verification)
        evidence_passed = self._check_passed(verification, "evidence")
        priority = self._priority(
            evidence_level=evidence_level,
            novelty_risk=novelty_risk,
            feasibility_risk=feasibility_risk,
            evidence_passed=evidence_passed,
            verified=is_verified_selected,
        )
        return ResearchDirection(
            title=title,
            source_idea_title=title,
            target_gap=self._target_gap(literature_analysis),
            core_problem=(
                self._value(idea, "motivation")
                or f"How to address {topic} under the limitations found in local evidence."
            ),
            hypothesis=(
                self._value(idea, "hypothesis")
                or (
                    "A planning-stage hypothesis should be validated before "
                    "being treated as a result."
                )
            ),
            method_sketch=(
                self._value(idea, "method")
                or (
                    "Design a lightweight method based on the retrieved evidence "
                    "and verify it against baselines."
                )
            ),
            supporting_evidence=supporting_evidence,
            evidence_support_level=evidence_level,
            novelty_risk=novelty_risk,
            feasibility_risk=feasibility_risk,
            recommended_priority=priority,
            assessment_status=(
                "verifier_assessed_selected"
                if is_verified_selected
                else "heuristic_unverified"
            ),
            next_steps=self._next_steps(evidence_level, novelty_risk),
        )

    def _fallback_direction(
        self,
        topic: str,
        literature_analysis: dict,
        evidence_context: list[dict],
    ) -> ResearchDirection:
        return ResearchDirection(
            title=f"Exploratory direction for {topic}",
            source_idea_title=None,
            target_gap=self._target_gap(literature_analysis),
            core_problem=(
                f"How to scope {topic} when no candidate idea is available."
            ),
            hypothesis=(
                "A planning-stage hypothesis should be validated before being "
                "treated as a result."
            ),
            method_sketch=(
                "Design a lightweight method based on the retrieved evidence "
                "and verify it against baselines."
            ),
            supporting_evidence=self._evidence_ids(evidence_context),
            evidence_support_level=(
                "weak" if evidence_context else "insufficient"
            ),
            novelty_risk="unknown",
            feasibility_risk="unknown",
            recommended_priority=(
                "low" if evidence_context else "exploratory"
            ),
            assessment_status="heuristic_unverified",
            next_steps=self._next_steps(
                "weak" if evidence_context else "insufficient",
                "unknown",
            ),
        )

    @staticmethod
    def _target_gap(literature_analysis: dict) -> str:
        if (
            literature_analysis.get("research_gap_status")
            == "insufficient_evidence"
        ):
            return (
                "The local corpus does not explicitly establish a defensible "
                "research gap."
            )
        return literature_analysis.get("research_gap") or (
            "Not explicitly established by the local evidence."
        )

    @staticmethod
    def _evidence_ids(evidence_context: list[dict]) -> list[str]:
        return [
            item["evidence_id"]
            for item in evidence_context[:2]
            if item.get("evidence_id")
        ]

    @staticmethod
    def _evidence_support_level(
        evidence_context: list[dict],
        verification: dict | None,
    ) -> str:
        if verification is not None:
            evidence = verification.get("evidence", {})
            if not evidence.get("passed", False):
                return "insufficient"
            level = evidence.get("support_level")
            if level in SUPPORT_LEVELS:
                return level
        if not evidence_context:
            return "insufficient"
        if len(evidence_context) == 1:
            return "weak"
        return "moderate"

    @staticmethod
    def _novelty_risk(verification: dict | None) -> str:
        if verification is None or "novelty" not in verification:
            return "unknown"
        return "low" if verification["novelty"].get("passed", False) else "high"

    @staticmethod
    def _feasibility_risk(verification: dict | None) -> str:
        if verification is None:
            return "unknown"
        checks = [
            verification.get(name, {}).get("passed")
            for name in ("experiment", "reproducibility")
        ]
        if any(value is None for value in checks):
            return "unknown"
        failures = checks.count(False)
        if failures == 0:
            return "low"
        return "high" if failures == 2 else "medium"

    @staticmethod
    def _check_passed(
        verification: dict | None,
        name: str,
    ) -> bool | None:
        if verification is None or name not in verification:
            return None
        return bool(verification[name].get("passed", False))

    @staticmethod
    def _priority(
        *,
        evidence_level: str,
        novelty_risk: str,
        feasibility_risk: str,
        evidence_passed: bool | None,
        verified: bool,
    ) -> str:
        if evidence_passed is False or evidence_level == "insufficient":
            return "exploratory"
        if not verified:
            return "low"
        if novelty_risk == "high" or feasibility_risk == "high":
            return "low"
        if (
            evidence_level == "strong"
            and novelty_risk == "low"
            and feasibility_risk == "low"
        ):
            return "high"
        if evidence_level in {"moderate", "strong"}:
            return "medium"
        return "low"

    @staticmethod
    def _next_steps(
        evidence_level: str,
        novelty_risk: str,
    ) -> list[str]:
        steps = [
            "Review the cited local evidence and confirm the research gap.",
            "Define the minimum viable experiment.",
            "Choose datasets, baselines, and metrics.",
        ]
        if evidence_level == "insufficient":
            steps.append(
                "Add more relevant papers before treating this as a grounded direction."
            )
        if novelty_risk == "high":
            steps.append(
                "Compare against saved or candidate ideas to reduce duplication."
            )
        if len(steps) < 4:
            steps.append(
                "Run a small human-approved pilot before scaling the experiment."
            )
        return steps[:5]

    @staticmethod
    def _value(item, name: str):
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

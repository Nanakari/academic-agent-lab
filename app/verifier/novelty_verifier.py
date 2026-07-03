"""Simple historical similarity check for generated ideas."""

import re

from app.schemas.research_idea import ResearchIdea
from app.schemas.verification_result import VerificationResult


class NoveltyVerifier:
    """Flag ideas with high token overlap against local idea history."""

    def __init__(self, similarity_threshold: float = 0.72) -> None:
        self.similarity_threshold = similarity_threshold

    def verify(self, idea: ResearchIdea, historical_ideas: list[dict]) -> VerificationResult:
        current = self._tokens(idea.title + " " + idea.method)
        maximum = 0.0
        nearest = ""
        for record in historical_ideas:
            previous = self._tokens(
                str(record.get("title", "")) + " " + str(record.get("method", ""))
            )
            similarity = len(current & previous) / max(1, len(current | previous))
            if similarity > maximum:
                maximum = similarity
                nearest = str(record.get("title", "untitled historical idea"))
        passed = maximum < self.similarity_threshold
        issues = [] if passed else [f"Idea is too similar to history: {nearest} ({maximum:.2f})."]
        suggestions = [] if passed else ["Change the hypothesis, mechanism, or evaluation setting."]
        return VerificationResult(passed, round(1.0 - maximum, 3), issues, suggestions)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[\w-]+", text.casefold()) if len(token) > 2}

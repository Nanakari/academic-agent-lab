"""Structured record of one bounded agent decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class AgentTraceEntry:
    """Capture what the agent observed, decided, and did at a decision point."""

    step: int
    observation: str
    decision: str
    reason: str
    action: str
    result: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

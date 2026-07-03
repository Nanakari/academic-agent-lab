"""Structured data models for the AI Scientific Agent."""

from app.schemas.evidence import (
    EvidenceChunk,
    PaperChunk,
    PaperDocument,
    support_level_for_score,
)
from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.research_idea import ResearchIdea
from app.schemas.scientific_task import ScientificTaskType
from app.schemas.verification_result import VerificationResult

__all__ = [
    "ExperimentPlan",
    "EvidenceChunk",
    "PaperChunk",
    "PaperDocument",
    "ResearchIdea",
    "ScientificTaskType",
    "VerificationResult",
    "support_level_for_score",
]

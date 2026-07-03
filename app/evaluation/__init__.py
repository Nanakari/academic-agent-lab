"""Fixture-based evaluation for AIScientificAgent."""

from app.evaluation.eval_cases import load_eval_cases
from app.evaluation.scientific_eval import ScientificEvaluator

__all__ = ["ScientificEvaluator", "load_eval_cases"]

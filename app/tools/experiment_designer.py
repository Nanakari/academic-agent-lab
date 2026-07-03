"""Experiment-plan construction for candidate AI research ideas."""

from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.research_idea import ResearchIdea


class ExperimentDesigner:
    """Select domain-aware evaluation ingredients and expose assumptions."""

    def design_experiment(self, idea: ResearchIdea, topic: str) -> ExperimentPlan:
        datasets, baselines, metrics = self._domain_defaults(topic)
        return ExperimentPlan(
            idea_title=idea.title,
            method=idea.method,
            datasets=datasets,
            baselines=baselines,
            metrics=metrics,
            ablation=[
                "Remove the proposed intervention/verification component.",
                "Replace adaptive routing with always-on routing at matched compute.",
                "Vary detector threshold and evidence-context size.",
            ],
            expected_results=[
                "Primary reliability metric improves over the strongest matched baseline.",
                "Adaptive routing retains most gains with lower average inference cost.",
                "Report mean, standard deviation, and per-failure-category results.",
            ],
            risks=[
                "The uncertainty signal may be poorly calibrated under domain shift.",
                "Benchmark contamination or annotation artifacts may inflate apparent gains.",
                "Additional inference stages may increase latency and compute cost.",
            ],
            implementation_notes=[
                "Pin model, dataset, and dependency versions.",
                "Run at least three random seeds and publish all evaluation prompts/configuration.",
                "Record hardware, decoding parameters, and preprocessing steps.",
            ],
        )

    @staticmethod
    def _domain_defaults(topic: str) -> tuple[list[str], list[str], list[str]]:
        lowered = topic.casefold()
        if any(token in lowered for token in ("lvlm", "multimodal", "多模态", "幻觉")):
            return (
                ["POPE", "MMHal-Bench", "MME"],
                ["Unmodified base LVLM", "Prompt-only mitigation", "Self-consistency verification"],
                ["Accuracy/F1", "Hallucination rate", "MMHal score", "Latency and token cost"],
            )
        if any(token in lowered for token in ("agent memory", "memory", "记忆")):
            return (
                ["LongMemEval", "LoCoMo", "A task-specific held-out interaction set"],
                ["No-memory agent", "Sliding-window memory", "Retrieval-only memory"],
                ["Task success", "Retrieval recall@k", "Answer F1", "Latency and token cost"],
            )
        return (
            ["One established public benchmark", "One out-of-domain benchmark", "A curated failure set"],
            ["Unmodified base model", "Strong published baseline", "Compute-matched simple baseline"],
            ["Primary task score", "Calibration/error rate", "Robustness under shift", "Compute cost"],
        )

"""Planner for the AI Scientific Agent workflow."""

from app.planner.plan_schema import ResearchPlan, ResearchPlanStep
from app.planner.task_classifier import TaskClassifier
from app.schemas.scientific_task import ScientificTaskType


class ResearchPlanner:
    """Classify a request and turn it into an explicit research plan."""

    def __init__(self, classifier: TaskClassifier | None = None) -> None:
        self.classifier = classifier or TaskClassifier()

    def classify_task(self, user_query: str) -> ScientificTaskType:
        return self.classifier.classify(user_query)

    def create_plan(
        self,
        user_query: str,
        task_type: ScientificTaskType,
    ) -> ResearchPlan:
        steps = [
            ResearchPlanStep(
                "retrieve_evidence",
                "Search local papers and scientific memory for relevant evidence.",
                "local_evidence_search",
                "Ranked evidence excerpts with source identifiers",
            ),
            ResearchPlanStep(
                "analyze_literature",
                "Extract existing methods, limitations, and a research gap.",
                "paper_analyzer",
                "Evidence-grounded literature analysis",
            ),
            ResearchPlanStep(
                "generate_and_rank_ideas",
                "Generate 2-3 testable ideas and select the strongest candidate.",
                "research_idea_generator",
                "Ranked candidate research ideas",
            ),
            ResearchPlanStep(
                "design_experiment",
                "Specify datasets, baselines, metrics, ablations, outcomes, and risks.",
                "experiment_designer",
                "Reproducible experiment plan",
            ),
            ResearchPlanStep(
                "verify_and_report",
                "Verify evidence, novelty, experiment completeness, and reproducibility.",
                "scientific_verifiers",
                "Verification logs plus JSON and Markdown reports",
            ),
        ]
        return ResearchPlan(
            task_type=task_type,
            goal=f"Produce a grounded and testable {task_type.value} for: {user_query.strip()}",
            steps=steps,
            required_tools=[
                "local_evidence_search",
                "paper_analyzer",
                "research_idea_generator",
                "experiment_designer",
                "report_writer",
            ],
            expected_outputs=[
                "literature_analysis",
                "candidate_ideas",
                "selected_idea",
                "experiment_plan",
                "verification_results",
                "result.json",
                "report.md",
            ],
        )

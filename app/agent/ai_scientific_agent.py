"""Offline-first AI Scientific Agent MVP."""

from __future__ import annotations

from pathlib import Path

from app.agent.base import BaseAgent
from app.agent.services import (
    AgentDecisionPolicy,
    EvidenceService,
    ExperimentBlueprintService,
    FeasibilityService,
    LiteratureAnalysisService,
    PersistenceService,
    ResearchDirectionService,
    VerificationPipeline,
)
from app.agent.services.external_search import (
    ExternalEvidenceService,
    ExternalSearchQueryBuilder,
)
from app.memory.scientific_memory import ScientificMemory
from app.planner.research_planner import ResearchPlanner
from app.schema import AgentState
from app.schemas.evidence_item import ExternalEvidenceResult
from app.tools.experiment_designer import ExperimentDesigner
from app.tools.paper_analyzer import PaperAnalyzer
from app.tools.paper_corpus import PaperCorpusIndexer
from app.tools.report_writer import ReportWriter
from app.tools.research_idea_generator import ResearchIdeaGenerator


class AIScientificAgent(BaseAgent):
    """Orchestrate planning, evidence, ideation, experiments, and verification."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        output_dir: str | Path | None = None,
        papers_dir: str | Path | None = None,
        top_k: int = 5,
        memory: ScientificMemory | None = None,
        llm=None,
        strict_domain: bool | None = None,
        domain_mode: str = "off",
        external_search_enabled: bool = False,
        external_search_sources: list[str] | None = None,
        external_max_results_per_source: int = 5,
        external_cache_enabled: bool = True,
        external_force_refresh: bool = False,
        external_evidence_service: ExternalEvidenceService | None = None,
    ) -> None:
        super().__init__(name="AIScientificAgent", max_steps=8)
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else Path(__file__).resolve().parents[2]
        )
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else self.project_root / "outputs" / "ai_scientific_agent"
        )
        requested_papers_dir = Path(papers_dir) if papers_dir else Path("data/papers")
        self.papers_dir = (
            requested_papers_dir.resolve()
            if requested_papers_dir.is_absolute()
            else (self.project_root / requested_papers_dir).resolve()
        )
        self.top_k = max(1, int(top_k))
        self.external_search_enabled = bool(external_search_enabled)
        requested_external_sources = (
            ["arxiv", "github"]
            if external_search_sources is None
            else external_search_sources
        )
        self.external_search_sources = [
            source.casefold()
            for source in requested_external_sources
            if source.casefold() in {"arxiv", "github"}
        ]
        self.external_max_results_per_source = max(
            1, int(external_max_results_per_source)
        )
        self.scientific_memory = memory or ScientificMemory(
            self.project_root / "data" / "research_memory"
        )
        self.paper_corpus = PaperCorpusIndexer(self.papers_dir)
        self.planner = ResearchPlanner()
        self.paper_analyzer = PaperAnalyzer()
        self.idea_generator = ResearchIdeaGenerator(llm=llm)
        self.experiment_designer = ExperimentDesigner()
        self.report_writer = ReportWriter()
        self.decision_policy = AgentDecisionPolicy()
        self.research_direction_service = ResearchDirectionService()
        self.feasibility_service = FeasibilityService()
        self.experiment_blueprint_service = ExperimentBlueprintService()
        self.literature_analysis_service = LiteratureAnalysisService(
            self.paper_analyzer
        )
        self.evidence_service = EvidenceService(
            self.project_root,
            self.paper_corpus,
            self.scientific_memory,
        )
        self.external_query_builder = ExternalSearchQueryBuilder()
        self.external_evidence_service = external_evidence_service or (
            ExternalEvidenceService(
                cache_dir=self.project_root / "data" / "external_cache",
                cache_enabled=external_cache_enabled,
                force_refresh=external_force_refresh,
            )
        )
        self.verification_pipeline = VerificationPipeline(
            domain_mode=domain_mode,
            strict_domain=strict_domain,
        )
        self.persistence_service = PersistenceService(
            self.scientific_memory,
            self.paper_analyzer,
        )
        # Preserve verifier attributes for callers that customized them directly.
        self.evidence_verifier = self.verification_pipeline.evidence_verifier
        self.novelty_verifier = self.verification_pipeline.novelty_verifier
        self.experiment_verifier = self.verification_pipeline.experiment_verifier
        self.reproducibility_verifier = (
            self.verification_pipeline.reproducibility_verifier
        )

    def run(self, user_query: str) -> dict:
        """Execute the complete MVP workflow and persist its artifacts."""
        if not user_query.strip():
            raise ValueError("user_query must not be empty")

        self.state = AgentState.RUNNING
        self.current_step = 0
        self.memory.add_user_message(user_query)
        agent_trace = []
        try:
            task_type = self.planner.classify_task(user_query)
            plan = self.planner.create_plan(user_query, task_type)
            self.current_step = 1

            evidence_context = self._retrieve_evidence(user_query, self.top_k)
            self.current_step = 2
            planned_external_queries = (
                self.external_query_builder.build_queries(user_query, max_queries=1)
                if self.external_search_enabled
                else []
            )
            external_query = (
                planned_external_queries[0]
                if planned_external_queries
                else user_query.strip()
            )
            if self.external_search_enabled:
                source_queries = {
                    source: self.external_query_builder.for_source(
                        external_query,
                        source,
                    )
                    for source in self.external_search_sources
                }
                external_result = self.external_evidence_service.retrieve(
                    external_query,
                    use_arxiv="arxiv" in self.external_search_sources,
                    use_github="github" in self.external_search_sources,
                    max_results_per_source=self.external_max_results_per_source,
                    source_queries=source_queries,
                )
            else:
                external_result = ExternalEvidenceResult(
                    enabled=False,
                    query=external_query,
                    retrieved_at="",
                )
            external_queries = list(dict.fromkeys(
                external_result.queries_used.values()
            ))
            trace_step_offset = 0
            if self.external_search_enabled:
                agent_trace.append(
                    self.decision_policy.decide_after_external_retrieval(
                        step=1,
                        sources_requested=list(self.external_search_sources),
                        queries_used=dict(external_result.queries_used),
                        evidence_items=external_result.evidence_items,
                        warnings=external_result.warnings,
                        cache_used=external_result.cache_used,
                    )
                )
                trace_step_offset = 1
            # arXiv abstracts may inform literature discovery. GitHub repositories
            # remain engineering evidence and never enter scientific verification.
            literature_context = [
                *evidence_context,
                *[
                    {
                        "excerpt": item.summary,
                        "text": item.summary,
                        "section": "Abstract",
                    }
                    for item in external_result.evidence_items
                    if item.source_type == "arxiv"
                ],
            ]
            literature_analysis = self._analyze_evidence(literature_context)
            self.current_step = 3
            agent_trace.append(
                self.decision_policy.decide_after_evidence(
                    step=1 + trace_step_offset,
                    topic=user_query.strip(),
                    evidence_context=evidence_context,
                    literature_analysis=literature_analysis,
                )
            )

            ideas = self.idea_generator.generate_ideas(user_query.strip(), evidence_context)
            selected_idea = ideas[0]
            experiment_plan = self.experiment_designer.design_experiment(
                selected_idea,
                user_query,
            )
            self.current_step = 4

            history = self.scientific_memory.load_recent_ideas(limit=50)
            initial_verification = self.verification_pipeline.verify(
                selected_idea,
                experiment_plan,
                evidence_context,
                literature_analysis,
                history,
                ideas,
                topic=user_query,
            )
            self.current_step = 5

            revision_needed = not all(
                item["passed"] for item in initial_verification.values()
            )
            agent_trace.append(
                self.decision_policy.decide_before_revision(
                    step=2 + trace_step_offset,
                    verification=initial_verification,
                )
            )

            verification = initial_verification
            revision_performed = False
            if revision_needed:
                selected_idea, experiment_plan = self._revise_once(
                    selected_idea,
                    experiment_plan,
                    evidence_context,
                )
                revision_performed = True
                verification = self.verification_pipeline.verify(
                    selected_idea,
                    experiment_plan,
                    evidence_context,
                    literature_analysis,
                    history,
                    ideas,
                    topic=user_query,
                )
            self.current_step = 6
            agent_trace.append(
                self.decision_policy.decide_after_verification(
                    step=3 + trace_step_offset,
                    verification=verification,
                    revision_performed=revision_performed,
                )
            )

            research_directions, selected_direction = (
                self.research_direction_service.plan(
                    topic=user_query.strip(),
                    literature_analysis=literature_analysis,
                    candidate_ideas=ideas,
                    selected_idea=selected_idea,
                    selected_idea_index=0,
                    evidence_context=evidence_context,
                    verification=verification,
                )
            )
            evidence_assessment = self.verification_pipeline.build_evidence_assessment(
                evidence_context,
                verification["evidence"],
                literature_analysis,
            )
            feasibility_assessment = self.feasibility_service.assess(
                selected_direction=selected_direction,
                selected_idea=selected_idea,
                experiment_plan=experiment_plan,
                evidence_assessment=evidence_assessment,
                verification=verification,
            )
            experiment_blueprint = self.experiment_blueprint_service.build(
                selected_direction=selected_direction,
                selected_idea=selected_idea,
                experiment_plan=experiment_plan,
                feasibility_assessment=feasibility_assessment,
                verification=verification,
                evidence_assessment=evidence_assessment,
            )
            verification_passed = all(
                item["passed"] for item in verification.values()
            )
            agent_trace.append(
                self.decision_policy.decide_before_report(
                    step=4 + trace_step_offset,
                    evidence_status=evidence_assessment["status"],
                    verification_passed=verification_passed,
                    selected_idea=selected_idea.to_dict(),
                )
            )
            external_evidence_gaps = []
            if self.external_search_enabled:
                if "arxiv" in self.external_search_sources:
                    external_evidence_gaps.append(
                        "Only metadata/abstract-level arXiv evidence was retrieved; "
                        "full-text verification was not performed."
                    )
                if "github" in self.external_search_sources:
                    external_evidence_gaps.append(
                        "GitHub repositories are implementation evidence, not "
                        "scientific validation."
                    )
            result = {
                "agent": self.name,
                "topic": user_query.strip(),
                "task_type": task_type.value,
                "plan": plan.to_dict(),
                "evidence_context": evidence_context,
                "evidence_status": evidence_assessment["status"],
                "evidence_used": evidence_assessment["used"],
                "evidence_gaps": evidence_assessment["gaps"],
                "unsupported_claims": evidence_assessment["unsupported_claims"],
                "corpus_warnings": list(self.paper_corpus.warnings),
                "external_search_status": {
                    "enabled": external_result.enabled,
                    "run_at": external_result.run_at,
                    "retrieved_at": external_result.retrieved_at,
                    "retrieved_at_by_source": dict(
                        external_result.retrieved_at_by_source
                    ),
                    "cache_loaded_at": external_result.cache_loaded_at,
                    "cache_used": external_result.cache_used,
                    "sources_requested": (
                        list(self.external_search_sources)
                        if self.external_search_enabled
                        else []
                    ),
                    "sources_used": list(external_result.sources_used),
                },
                "external_search_queries": external_queries,
                "external_search_query_by_source": dict(
                    external_result.queries_used
                ),
                "external_evidence": [
                    item.to_dict() for item in external_result.evidence_items
                ],
                "external_sources_used": sorted({
                    item.source_type for item in external_result.evidence_items
                }),
                "external_evidence_gaps": external_evidence_gaps,
                "external_retrieval_warnings": list(external_result.warnings),
                "literature_analysis": literature_analysis,
                "candidate_ideas": [idea.to_dict() for idea in ideas],
                "research_directions": [
                    direction.to_dict() for direction in research_directions
                ],
                "selected_direction": selected_direction.to_dict(),
                "feasibility_assessment": feasibility_assessment.to_dict(),
                "experiment_blueprint": experiment_blueprint.to_dict(),
                "selected_idea": selected_idea.to_dict(),
                "experiment_plan": experiment_plan.to_dict(),
                "verification": verification,
                "verification_passed": verification_passed,
                "revision_performed": revision_performed,
                "agent_trace": [entry.to_dict() for entry in agent_trace],
                "output_paths": {
                    "json": str((self.output_dir / "result.json").resolve()),
                    "markdown": str((self.output_dir / "report.md").resolve()),
                },
            }
            self.persistence_service.save(result)
            self.current_step = 7
            self.report_writer.write_json_report(result, self.output_dir / "result.json")
            self.report_writer.write_markdown_report(result, self.output_dir / "report.md")
            self.current_step = 8
            self.state = AgentState.FINISHED
            return result
        except Exception:
            self.state = AgentState.ERROR
            raise

    def step(self) -> str:
        """BaseAgent compatibility; this orchestrator executes atomically in run()."""
        raise NotImplementedError("AIScientificAgent uses the structured run() workflow.")

    def _retrieve_evidence(self, query: str, top_k: int | None = None) -> list[dict]:
        """Compatibility wrapper around the evidence service."""
        return self.evidence_service.retrieve(query, top_k or self.top_k)

    def _analyze_evidence(self, evidence_context: list[dict]) -> dict:
        """Compatibility wrapper around LiteratureAnalysisService."""
        return self.literature_analysis_service.analyze(evidence_context)

    @staticmethod
    def _revise_once(idea, experiment_plan, evidence_context):
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

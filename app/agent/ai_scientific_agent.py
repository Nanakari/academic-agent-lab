"""LLM-driven, verifier-bounded AI Scientific Agent MVP."""

from __future__ import annotations

from pathlib import Path

from app.agent.base import BaseAgent
from app.agent.services import (
    AgentDecisionPolicy,
    EvidenceService,
    ExperimentBlueprintService,
    FeasibilityService,
    LiteratureAnalysisService,
    LLMReflectionService,
    LLMScientificAnalysisService,
    LLMStageResult,
    LLMToolDecisionService,
    PersistenceService,
    ResearchDirectionService,
    RevisionService,
    VerificationPipeline,
)
from app.agent.services.external_search import (
    ExternalEvidenceService,
    ExternalSearchQueryBuilder,
)
from app.memory.scientific_memory import ScientificMemory
from app.planner.research_planner import ResearchPlanner
from app.schema import AgentState
from app.schemas.evidence_item import EvidenceItem, ExternalEvidenceResult
from app.tools.experiment_designer import ExperimentDesigner
from app.tools.external_relevance import (
    has_excluded_topic_drift,
    is_external_evidence_relevant_to_topic,
)
from app.tools.paper_analyzer import PaperAnalyzer
from app.tools.paper_corpus import PaperCorpusIndexer
from app.tools.query_normalizer import normalize_research_query
from app.tools.report_writer import ReportWriter
from app.tools.research_idea_generator import ResearchIdeaGenerator


class AIScientificAgent(BaseAgent):
    """Orchestrate planning, evidence, ideation, experiments, and verification."""

    SUPPORTED_EXTERNAL_SOURCES = {"arxiv", "github"}
    LLM_STAGE_NAMES = {
        "tool_decision",
        "literature_analysis",
        "idea_generation",
        "experiment_design",
        "reflection",
    }

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
        llm_tool_decision_enabled: bool | None = None,
        llm_stages: list[str] | str | None = None,
    ) -> None:
        super().__init__(name="AIScientificAgent", max_steps=8)
        self.llm = llm
        self.llm_stages = self._normalize_llm_stages(llm_stages, llm=llm)
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
        if isinstance(requested_external_sources, str):
            requested_external_sources = requested_external_sources.split(",")
        self.external_search_sources = [
            str(source).strip().casefold()
            for source in requested_external_sources
            if str(source).strip()
        ]
        invalid_external_sources = sorted(
            set(self.external_search_sources) - self.SUPPORTED_EXTERNAL_SOURCES
        )
        if invalid_external_sources:
            raise ValueError(
                "Unsupported external_search_sources value(s): "
                + ", ".join(invalid_external_sources)
            )
        if self.external_search_enabled and not self.external_search_sources:
            raise ValueError(
                "external_search_sources must include at least one supported source "
                "when external_search_enabled is true."
            )
        self.external_max_results_per_source = max(
            1, int(external_max_results_per_source)
        )
        self.scientific_memory = memory or ScientificMemory(
            self.project_root / "data" / "research_memory"
        )
        self.paper_corpus = PaperCorpusIndexer(self.papers_dir)
        self.planner = ResearchPlanner()
        self.paper_analyzer = PaperAnalyzer()
        self.idea_generator = ResearchIdeaGenerator(
            llm=llm,
            enabled=self._llm_stage_enabled("idea_generation"),
        )
        self.experiment_designer = ExperimentDesigner(
            llm=llm,
            enabled=self._llm_stage_enabled("experiment_design"),
        )
        self.report_writer = ReportWriter()
        self.decision_policy = AgentDecisionPolicy()
        self.tool_decision_service = LLMToolDecisionService(
            llm=llm,
            enabled=(
                llm is not None
                if llm_tool_decision_enabled is None
                else llm_tool_decision_enabled
            )
            and self._llm_stage_enabled("tool_decision"),
        )
        self.research_direction_service = ResearchDirectionService()
        self.feasibility_service = FeasibilityService()
        self.experiment_blueprint_service = ExperimentBlueprintService()
        self.revision_service = RevisionService()
        self.literature_analysis_service = LiteratureAnalysisService(
            self.paper_analyzer
        )
        self.llm_scientific_analysis_service = LLMScientificAnalysisService(
            deterministic_service=self.literature_analysis_service,
            llm=llm,
            enabled=self._llm_stage_enabled("literature_analysis"),
        )
        self.llm_reflection_service = LLMReflectionService(
            llm=llm,
            enabled=self._llm_stage_enabled("reflection"),
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

            expanded_query = normalize_research_query(user_query)
            tool_decision = self.tool_decision_service.decide(
                topic=user_query.strip(),
                default_top_k=self.top_k,
                external_search_enabled=self.external_search_enabled,
                external_sources=self.external_search_sources,
            )
            llm_stage_results: list[LLMStageResult] = []
            evidence_context = self._retrieve_evidence(
                expanded_query,
                tool_decision.top_k,
                use_local_papers=tool_decision.use_local_evidence_search,
                use_scientific_memory=tool_decision.use_scientific_memory,
            )
            self.current_step = 2
            run_external_search = tool_decision.use_external_search
            run_external_sources = (
                tool_decision.external_sources
                if tool_decision.external_sources
                else list(self.external_search_sources)
            )
            planned_external_queries = (
                self.external_query_builder.build_queries(
                    expanded_query,
                    max_queries=1,
                )
                if run_external_search
                else []
            )
            external_query = (
                planned_external_queries[0]
                if planned_external_queries
                else user_query.strip()
            )
            if run_external_search:
                source_queries = {
                    source: self.external_query_builder.for_source(
                        external_query,
                        source,
                    )
                    for source in run_external_sources
                }
                external_result = self.external_evidence_service.retrieve(
                    external_query,
                    use_arxiv="arxiv" in run_external_sources,
                    use_github="github" in run_external_sources,
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
            if run_external_search:
                agent_trace.append(
                    self.decision_policy.decide_after_external_retrieval(
                        step=1,
                        sources_requested=list(run_external_sources),
                        queries_used=dict(external_result.queries_used),
                        evidence_items=external_result.evidence_items,
                        warnings=external_result.warnings,
                        cache_used=external_result.cache_used,
                    )
                )
                trace_step_offset = 1
            (
                literature_context,
                external_literature_evidence,
                rejected_external_literature_evidence,
            ) = (
                self._build_literature_context(
                    evidence_context,
                    external_result.evidence_items,
                    expanded_query,
                    user_query.strip(),
                )
            )
            external_literature_evidence_dicts = [
                item.to_dict() for item in external_literature_evidence
            ]
            literature_analysis, analysis_stage = (
                self.llm_scientific_analysis_service.analyze(
                    evidence_context=evidence_context,
                    external_literature_evidence=external_literature_evidence_dicts,
                    topic=user_query.strip(),
                    fallback_evidence_context=literature_context,
                )
            )
            llm_stage_results.append(analysis_stage)
            if has_excluded_topic_drift(
                str(literature_analysis.get("research_gap", "")),
                expanded_query,
                user_query,
            ):
                literature_analysis = {
                    **literature_analysis,
                    "research_gap": (
                        "A defensible research gap cannot be established from "
                        "the retrieved topic-relevant evidence."
                    ),
                    "research_gap_status": (
                        "insufficient_topic_relevant_evidence"
                    ),
                    "research_gap_note": (
                        "The generated gap failed the topic-drift safeguard."
                    ),
                }
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
            llm_stage_results.append(self.idea_generator.last_stage_result)
            selected_idea = ideas[0]
            experiment_plan = self.experiment_designer.design_experiment(
                selected_idea,
                user_query,
                evidence_context,
            )
            llm_stage_results.append(self.experiment_designer.last_stage_result)
            self.current_step = 4

            history = self.scientific_memory.load_recent_ideas(limit=50)
            initial_verification = self.verification_pipeline.verify(
                selected_idea,
                experiment_plan,
                evidence_context,
                literature_analysis,
                history,
                ideas,
                topic=expanded_query,
                external_literature_evidence=external_literature_evidence_dicts,
            )
            self.current_step = 5

            revision_needed = any(
                not item["passed"]
                for name, item in initial_verification.items()
                if name != "novelty"
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
                selected_idea, experiment_plan = self.revision_service.revise_once(
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
                    topic=expanded_query,
                    external_literature_evidence=external_literature_evidence_dicts,
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
            reflection, reflection_stage = self.llm_reflection_service.reflect(
                verification=verification,
                evidence_status=evidence_assessment["status"],
                selected_idea=selected_idea.to_dict(),
                experiment_plan=experiment_plan.to_dict(),
            )
            llm_stage_results.append(reflection_stage)
            llm_metadata = self._llm_stage_metadata(
                tool_decision,
                llm_stage_results,
            )
            evidence_source_breakdown = self._evidence_source_breakdown(
                evidence_context
            )
            local_paper_evidence_count = evidence_source_breakdown.get(
                "local_paper",
                0,
            )
            memory_evidence_count = evidence_source_breakdown.get(
                "scientific_memory",
                0,
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
            scientific_readiness = self._scientific_readiness(
                evidence_status=evidence_assessment["status"],
                local_paper_evidence_count=local_paper_evidence_count,
                literature_analysis=literature_analysis,
                verification_passed=verification_passed,
            )
            final_recommendation = self._final_recommendation(
                scientific_readiness=scientific_readiness,
                local_paper_evidence_count=local_paper_evidence_count,
            )
            novelty_revision_strategy = self._novelty_revision_strategy(
                initial_verification.get("novelty", {})
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
            if run_external_search:
                if "arxiv" in run_external_sources:
                    external_evidence_gaps.append(
                        "Only metadata/abstract-level arXiv evidence was retrieved; "
                        "full-text verification was not performed."
                    )
                    if (
                        any(
                            item.source_type == "arxiv"
                            for item in external_result.evidence_items
                        )
                        and not external_literature_evidence
                    ):
                        external_evidence_gaps.append(
                            "External search results were recorded but excluded "
                            "from literature analysis because they did not meet "
                            "the relevance threshold."
                        )
                if "github" in run_external_sources:
                    external_evidence_gaps.append(
                        "GitHub repositories are implementation evidence, not "
                        "scientific validation."
                    )
            result = {
                "agent": self.name,
                "topic": user_query.strip(),
                "expanded_query": expanded_query,
                "task_type": task_type.value,
                "plan": plan.to_dict(),
                "tool_decision": tool_decision.to_dict(),
                **llm_metadata,
                "evidence_context": evidence_context,
                "evidence_status": evidence_assessment["status"],
                "evidence_used": evidence_assessment["used"],
                "evidence_gaps": evidence_assessment["gaps"],
                "unsupported_claims": evidence_assessment["unsupported_claims"],
                "evidence_source_breakdown": evidence_source_breakdown,
                "local_paper_evidence_count": local_paper_evidence_count,
                "memory_evidence_count": memory_evidence_count,
                "corpus_warnings": list(self.paper_corpus.warnings),
                "deduplicated_evidence_count": (
                    self.evidence_service.last_deduplicated_count
                ),
                "external_cache_used": external_result.cache_used,
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
                        list(run_external_sources)
                        if run_external_search
                        else []
                    ),
                    "sources_used": list(external_result.sources_used),
                    "literature_external_evidence_count": len(
                        external_literature_evidence
                    ),
                },
                "external_search_queries": external_queries,
                "external_search_query_by_source": dict(
                    external_result.queries_used
                ),
                "external_evidence": [
                    item.to_dict() for item in external_result.evidence_items
                ],
                "external_evidence_used_for_literature": [
                    item.to_dict() for item in external_literature_evidence
                ],
                "external_evidence_rejected_for_literature": (
                    rejected_external_literature_evidence
                ),
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
                "scientific_readiness": scientific_readiness,
                "final_recommendation": final_recommendation,
                "llm_reflection": reflection,
                "revision_performed": revision_performed,
                "novelty_revision_strategy": novelty_revision_strategy,
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

    def _retrieve_evidence(
        self,
        query: str,
        top_k: int | None = None,
        *,
        use_local_papers: bool = True,
        use_scientific_memory: bool = True,
    ) -> list[dict]:
        """Compatibility wrapper around the evidence service."""
        return self.evidence_service.retrieve(
            query,
            top_k or self.top_k,
            use_local_papers=use_local_papers,
            use_scientific_memory=use_scientific_memory,
        )

    def _analyze_evidence(self, evidence_context: list[dict]) -> dict:
        """Compatibility wrapper around LiteratureAnalysisService."""
        return self.literature_analysis_service.analyze(evidence_context)

    def _llm_stage_enabled(self, stage: str) -> bool:
        return stage in self.llm_stages and self.llm is not None

    @classmethod
    def _normalize_llm_stages(
        cls,
        stages: list[str] | str | None,
        *,
        llm,
    ) -> set[str]:
        if llm is None:
            return set()
        if stages is None or stages == "all":
            return set(cls.LLM_STAGE_NAMES)
        if isinstance(stages, str):
            raw_stages = [
                item.strip()
                for item in stages.split(",")
                if item.strip()
            ]
        else:
            raw_stages = [str(item).strip() for item in stages if str(item).strip()]
        selected = {stage for stage in raw_stages if stage != "none"}
        invalid = sorted(selected - cls.LLM_STAGE_NAMES)
        if invalid:
            raise ValueError("Unsupported llm_stages value(s): " + ", ".join(invalid))
        return selected

    @staticmethod
    def _evidence_source_breakdown(evidence_context: list[dict]) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for item in evidence_context:
            kind = str(item.get("kind") or "unknown")
            breakdown[kind] = breakdown.get(kind, 0) + 1
        return dict(sorted(breakdown.items()))

    @staticmethod
    def _llm_stage_metadata(
        tool_decision,
        stage_results: list[LLMStageResult],
    ) -> dict:
        stages = list(tool_decision.llm_call_stages)
        fallback_stages = []
        fallback_reasons = []
        generated_sections = []
        deterministic_sections = []
        for item in stage_results:
            if item.llm_used or item.fallback_used:
                stages.append(item.stage)
            if item.fallback_used:
                fallback_stages.append(item.stage)
                fallback_reasons.append({
                    "stage": item.stage,
                    "reason": item.warning or "LLM stage fell back to deterministic logic.",
                })
            generated_sections.extend(item.generated_sections)
            deterministic_sections.extend(item.deterministic_sections)
        return {
            "llm_call_count": len(stages),
            "llm_call_stages": list(dict.fromkeys(stages)),
            "llm_fallback_stages": list(dict.fromkeys(fallback_stages)),
            "llm_fallback_reasons": fallback_reasons,
            "llm_generated_sections": list(dict.fromkeys(generated_sections)),
            "deterministic_sections": list(dict.fromkeys(deterministic_sections)),
        }

    @staticmethod
    def _scientific_readiness(
        *,
        evidence_status: str,
        local_paper_evidence_count: int,
        literature_analysis: dict,
        verification_passed: bool,
    ) -> str:
        gap_status = str(
            literature_analysis.get("research_gap_status") or ""
        )
        if local_paper_evidence_count == 0 and gap_status in {
            "insufficient_evidence",
            "insufficient_topic_relevant_evidence",
            "evidence_insufficient",
        }:
            return "exploratory"
        if evidence_status == "memory_only":
            return "memory_only"
        if evidence_status in {"evidence_insufficient", "insufficient_evidence"}:
            return "insufficient_evidence"
        if evidence_status == "sufficient" and verification_passed:
            return "strong"
        return "weak"

    @staticmethod
    def _final_recommendation(
        *,
        scientific_readiness: str,
        local_paper_evidence_count: int,
    ) -> str:
        if local_paper_evidence_count == 0:
            return "needs_more_local_paper_evidence"
        if scientific_readiness in {"exploratory", "insufficient_evidence"}:
            return "needs_stronger_topic_evidence"
        if scientific_readiness in {"memory_only", "weak"}:
            return "needs_human_review_before_claiming_support"
        return "ready_for_human_reviewed_pilot"

    @staticmethod
    def _novelty_revision_strategy(novelty: dict) -> str:
        literature_status = (
            novelty.get("literature_novelty", {}).get("status")
        )
        local_overlap = novelty.get("local_memory_overlap", {})
        if literature_status == "insufficient_literature_evidence":
            return "no_sufficient_literature_novelty_evidence"
        if literature_status == "overlapping":
            return "literature_overlap_requires_human_review"
        try:
            max_similarity = float(local_overlap.get("max_similarity", 0.0))
        except (TypeError, ValueError):
            max_similarity = 0.0
        if max_similarity >= 0.95:
            return "local_memory_duplicate_requires_human_review"
        if local_overlap.get("has_overlap"):
            return "warning_only_local_overlap"
        return "not_required"

    @staticmethod
    def _build_literature_context(
        local_evidence: list[dict],
        external_evidence: list[EvidenceItem],
        expanded_query: str,
        topic: str,
    ) -> tuple[list[dict], list[EvidenceItem], list[dict]]:
        """Admit only relevant arXiv abstracts to literature discovery."""
        selected_external = []
        rejected_external = []
        for item in external_evidence:
            accepted, reason = is_external_evidence_relevant_to_topic(
                item,
                expanded_query,
                topic,
            )
            if accepted:
                selected_external.append(item)
            else:
                rejected_external.append({
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "url": item.url,
                    "relevance_score": item.relevance_score,
                    "reason": reason,
                })
        literature_context = [
            *local_evidence,
            *[
                {
                    "title": item.title,
                    "excerpt": item.summary,
                    "text": item.summary,
                    "section": "Abstract",
                    "source_type": "arxiv",
                    "source_id": item.source_id,
                }
                for item in selected_external
            ],
        ]
        return literature_context, selected_external, rejected_external

    def _revise_once(self, idea, experiment_plan, evidence_context):
        """Compatibility wrapper for callers that used the old helper directly."""
        return self.revision_service.revise_once(
            idea,
            experiment_plan,
            evidence_context,
        )

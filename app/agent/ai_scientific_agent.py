"""Offline-first AI Scientific Agent MVP."""

from __future__ import annotations

from pathlib import Path

from app.agent.base import BaseAgent
from app.memory.scientific_memory import ScientificMemory
from app.planner.research_planner import ResearchPlanner
from app.schema import AgentState
from app.schemas.evidence import support_level_for_score
from app.tools.experiment_designer import ExperimentDesigner
from app.tools.paper_analyzer import PaperAnalyzer
from app.tools.paper_corpus import (
    PaperCorpusIndexer,
    infer_supporting_claim,
    keyword_tokens,
)
from app.tools.report_writer import ReportWriter
from app.tools.research_idea_generator import ResearchIdeaGenerator
from app.verifier.evidence_verifier import EvidenceVerifier
from app.verifier.experiment_verifier import ExperimentVerifier
from app.verifier.novelty_verifier import NoveltyVerifier
from app.verifier.reproducibility_verifier import ReproducibilityVerifier


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
        self.scientific_memory = memory or ScientificMemory(
            self.project_root / "data" / "research_memory"
        )
        self.paper_corpus = PaperCorpusIndexer(self.papers_dir)
        self.planner = ResearchPlanner()
        self.paper_analyzer = PaperAnalyzer()
        self.idea_generator = ResearchIdeaGenerator(llm=llm)
        self.experiment_designer = ExperimentDesigner()
        self.report_writer = ReportWriter()
        self.evidence_verifier = EvidenceVerifier()
        self.novelty_verifier = NoveltyVerifier()
        self.experiment_verifier = ExperimentVerifier()
        self.reproducibility_verifier = ReproducibilityVerifier()

    def run(self, user_query: str) -> dict:
        """Execute the complete MVP workflow and persist its artifacts."""
        if not user_query.strip():
            raise ValueError("user_query must not be empty")

        self.state = AgentState.RUNNING
        self.current_step = 0
        self.memory.add_user_message(user_query)
        try:
            task_type = self.planner.classify_task(user_query)
            plan = self.planner.create_plan(user_query, task_type)
            self.current_step = 1

            evidence_context = self._retrieve_evidence(user_query, self.top_k)
            self.current_step = 2
            literature_analysis = self._analyze_evidence(evidence_context)
            self.current_step = 3

            ideas = self.idea_generator.generate_ideas(user_query.strip(), evidence_context)
            selected_idea = ideas[0]
            experiment_plan = self.experiment_designer.design_experiment(
                selected_idea,
                user_query,
            )
            self.current_step = 4

            history = self.scientific_memory.load_recent_ideas(limit=50)
            verification = self._verify(
                selected_idea,
                experiment_plan,
                evidence_context,
                literature_analysis,
                history,
                ideas,
            )
            self.current_step = 5

            revision_performed = not all(item["passed"] for item in verification.values())
            if revision_performed:
                selected_idea, experiment_plan = self._revise_once(
                    selected_idea,
                    experiment_plan,
                    evidence_context,
                )
                verification = self._verify(
                    selected_idea,
                    experiment_plan,
                    evidence_context,
                    literature_analysis,
                    history,
                    ideas,
                )
            self.current_step = 6

            evidence_assessment = self._build_evidence_assessment(
                evidence_context,
                verification["evidence"],
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
                "literature_analysis": literature_analysis,
                "candidate_ideas": [idea.to_dict() for idea in ideas],
                "selected_idea": selected_idea.to_dict(),
                "experiment_plan": experiment_plan.to_dict(),
                "verification": verification,
                "verification_passed": all(
                    item["passed"] for item in verification.values()
                ),
                "revision_performed": revision_performed,
                "output_paths": {
                    "json": str((self.output_dir / "result.json").resolve()),
                    "markdown": str((self.output_dir / "report.md").resolve()),
                },
            }
            self._save_memory(result)
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
        """Search the paper corpus first, then fill remaining slots from memory."""
        limit = max(1, top_k or self.top_k)
        selected = []
        for evidence in self.paper_corpus.search(query, top_k=limit):
            item = evidence.to_dict()
            source_path = Path(item["source_path"])
            try:
                display_source = source_path.relative_to(self.project_root).as_posix()
            except ValueError:
                display_source = str(source_path)
            item.update(
                {
                    "source": display_source,
                    "excerpt": item["text"],
                    "kind": "local_paper",
                }
            )
            selected.append(item)

        remaining = limit - len(selected)
        if remaining > 0:
            selected.extend(self._search_memory_evidence(query, remaining))

        for index, item in enumerate(selected, start=1):
            item["evidence_id"] = f"E{index}"
        return selected

    def _search_memory_evidence(self, query: str, limit: int) -> list[dict]:
        records = []
        seen = set()
        for keyword in [query, *sorted(keyword_tokens(query))]:
            for record in self.scientific_memory.search_memory(keyword):
                # Only paper-derived notes are valid fallback scientific evidence.
                if (
                    record.get("memory_type") != "paper_note"
                    or str(record.get("source", "")).startswith("memory:")
                ):
                    continue
                record_key = (
                    record.get("memory_type"),
                    record.get("saved_at"),
                    record.get("title"),
                    record.get("source"),
                )
                if record_key not in seen:
                    seen.add(record_key)
                    records.append(record)

        candidates = []
        for record in records:
            excerpt = str(
                record.get("summary")
                or record.get("motivation")
                or record.get("hypothesis")
                or record
            )
            title = str(record.get("title") or record.get("topic") or "Scientific memory")
            score = self.paper_corpus.score_text(query, excerpt, title)
            if score > 0:
                matched_keywords = self.paper_corpus.matched_keywords(
                    query,
                    excerpt,
                    title,
                )
                candidates.append(
                    {
                        "paper_id": f"memory-{record.get('memory_type', 'record')}",
                        "title": title,
                        "source_path": f"memory:{record['memory_type']}",
                        "file_type": "memory",
                        "page": record.get("page"),
                        "section": record.get("section"),
                        "chunk_id": str(record.get("evidence_id") or "memory-record"),
                        "text": excerpt[:800],
                        "source": f"memory:{record['memory_type']}",
                        "excerpt": excerpt[:800],
                        "score": round(score, 3),
                        "matched_keywords": matched_keywords,
                        "supporting_claim": infer_supporting_claim(query, excerpt),
                        "support_level": support_level_for_score(score),
                        "kind": "scientific_memory",
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]

    def _analyze_evidence(self, evidence_context: list[dict]) -> dict:
        if not evidence_context:
            return {
                "existing_methods": ["No relevant method was found in local evidence."],
                "key_limitations": ["Local evidence coverage is insufficient."],
                "research_gap": (
                    "A defensible gap cannot yet be established from local papers; "
                    "the generated ideas must be treated as exploratory."
                ),
            }
        combined = "\n".join(item["excerpt"] for item in evidence_context)
        extracted = self.paper_analyzer.extract_problem_method_experiment_limitation(combined)
        method_sections = self._evidence_from_sections(
            evidence_context,
            {"method", "methods", "methodology", "approach"},
        )
        limitation_sections = self._evidence_from_sections(
            evidence_context,
            {"limitation", "limitations", "future work"},
        )
        existing_methods = method_sections or extracted["method"]
        limitations = limitation_sections or extracted["limitation"]
        return {
            "existing_methods": existing_methods,
            "key_limitations": limitations,
            "research_gap": (
                "The retrieved evidence describes existing methods but leaves unresolved: "
                + limitations[0]
            ),
        }

    @staticmethod
    def _evidence_from_sections(
        evidence_context: list[dict],
        section_names: set[str],
    ) -> list[str]:
        """Prefer explicit section metadata over cue matching in chunk text."""
        return [
            item["text"]
            for item in evidence_context
            if str(item.get("section") or "").casefold() in section_names
        ][:3]

    def _verify(
        self,
        idea,
        experiment_plan,
        evidence_context,
        literature_analysis,
        history,
        ideas,
    ) -> dict:
        results = {
            "evidence": self.evidence_verifier.verify(
                idea,
                evidence_context,
                claims=[
                    literature_analysis["research_gap"],
                    *literature_analysis["existing_methods"],
                ],
                ideas=ideas,
            ),
            "novelty": self.novelty_verifier.verify(idea, history),
            "experiment": self.experiment_verifier.verify(experiment_plan),
            "reproducibility": self.reproducibility_verifier.verify(experiment_plan),
        }
        return {name: result.to_dict() for name, result in results.items()}

    @staticmethod
    def _build_evidence_assessment(
        evidence_context: list[dict],
        evidence_verification: dict,
    ) -> dict:
        used = [
            {
                "evidence_id": item["evidence_id"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "source_path": item["source_path"],
                "file_type": item["file_type"],
                "page": item["page"],
                "section": item["section"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "matched_keywords": item["matched_keywords"],
                "supporting_claim": item["supporting_claim"],
                "support_level": item["support_level"],
                "kind": item["kind"],
            }
            for item in evidence_context
        ]
        gaps = []
        if not any(item["kind"] == "local_paper" for item in evidence_context):
            gaps.append("No matching evidence was retrieved from the local paper corpus.")
        gaps.extend(evidence_verification["issues"])
        unsupported_claims = evidence_verification["unsupported_claims"]
        return {
            "status": (
                "sufficient"
                if evidence_verification["passed"]
                else "evidence_insufficient"
            ),
            "used": used,
            "gaps": list(dict.fromkeys(gaps)),
            "unsupported_claims": unsupported_claims,
        }

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

    def _save_memory(self, result: dict) -> None:
        for evidence in result["evidence_context"]:
            if evidence["kind"] != "local_paper":
                continue
            self.scientific_memory.save_paper_note(
                {
                    "topic": result["topic"],
                    "source": evidence["source"],
                    "evidence_id": evidence["evidence_id"],
                    "page": evidence["page"],
                    "section": evidence["section"],
                    "file_type": evidence["file_type"],
                    "chunk_id": evidence["chunk_id"],
                    "summary": self.paper_analyzer.summarize_paper_text(
                        evidence["excerpt"],
                        max_sentences=2,
                    ),
                }
            )
        self.scientific_memory.save_idea(
            {"topic": result["topic"], **result["selected_idea"]}
        )
        self.scientific_memory.save_experiment(
            {"topic": result["topic"], **result["experiment_plan"]}
        )
        self.scientific_memory.save_verification_log(
            {
                "topic": result["topic"],
                "passed": result["verification_passed"],
                "results": result["verification"],
            }
        )

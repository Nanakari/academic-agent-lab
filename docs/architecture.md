# Architecture

## Module Structure

The repository keeps the original academic tool-calling path and the newer
scientific workflow side by side.

```text
app/
  agent/
    base.py
    toolcall.py
    academic_agent.py
    ai_scientific_agent.py
    services/
      evidence_service.py
      literature_analysis_service.py
      verification_pipeline.py
      persistence_service.py
  planner/
    task_classifier.py
    research_planner.py
    plan_schema.py
  memory/
    scientific_memory.py
    paper_memory.py
    idea_memory.py
    experiment_memory.py
  tools/
    paper_corpus.py
    paper_analyzer.py
    research_idea_generator.py
    experiment_designer.py
    report_writer.py
  verifier/
    evidence_verifier.py
    novelty_verifier.py
    experiment_verifier.py
    reproducibility_verifier.py
  evaluation/
    eval_cases.py
    eval_metrics.py
    scientific_eval.py
    eval_report.py
    validation_report.py
  schemas/
    evidence.py
    evaluation.py
    experiment_plan.py
    research_idea.py
    verification_result.py
```

The singular `app/tool/` package contains the original generic tool-calling
tools. The plural `app/tools/` package contains deterministic scientific
workflow components.

## AIScientificAgent Run Flow

`AIScientificAgent.run()` executes a bounded orchestration flow:

1. `ResearchPlanner` classifies the request and creates a structured plan.
2. `EvidenceService` uses `PaperCorpusIndexer` to scan and chunk local papers;
   paper-derived memory is a secondary evidence source. The index is cached
   within the current process and refreshed when file metadata or chunk settings
   change.
3. `LiteratureAnalysisService` uses `PaperAnalyzer` to extract methods,
   limitations, and an evidence-aware research-gap status, preferring detected
   section metadata.
4. `ResearchIdeaGenerator` creates and ranks three candidate ideas.
5. `ExperimentDesigner` creates datasets, baselines, metrics, ablations,
   expected results, risks, and reproducibility notes.
6. `VerificationPipeline` runs four verifiers for evidence, novelty, experiment
   completeness, and reproducibility.
7. One bounded revision is allowed when verification fails.
8. `PersistenceService` writes deduplicated paper notes and ideas plus experiment
   and verification records to ScientificMemory; ReportWriter emits JSON and
   Markdown.

This flow is intentionally explicit rather than hidden inside an LLM prompt.
Each stage can be tested or replaced independently.

The services layer is deliberately small: it extracts retrieval, verifier
coordination, and persistence from the orchestrator without introducing
framework abstractions. ScientificMemory remains a local JSONL store, not a
production database.

## Runtime Data

- `data/papers/` contains user-supplied papers and is ignored by Git.
- `data/research_memory/` contains runtime JSONL records and is ignored by Git.
- `outputs/` contains agent, evaluation, and validation reports and is ignored
  by Git.
- Test fixtures remain tracked under `tests/fixtures/`.

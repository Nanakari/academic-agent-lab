# academic-agent-lab

A small, modular academic-agent project for local paper reading, structured
summarization, retrieval-augmented question answering, and AI research workflow
experiments.

The original `AcademicAgent` remains available through `main.py`. The new mode
below is an incremental, offline-first extension and does not replace the
existing tool-calling implementation.

## AI Scientific Agent Mode

`AIScientificAgent` upgrades the project from paper understanding to a minimal
AI scientific workflow. Given an AI research direction, it classifies the task,
plans the workflow, searches local evidence, identifies limitations and a
research gap, generates and ranks three ideas, designs an experiment, verifies
the result, and saves a report.

### Architecture

- **Planner** — classifies `paper_reading`, `literature_analysis`,
  `idea_generation`, `experiment_design`, or `research_proposal`, then emits a
  structured `ResearchPlan`.
- **Memory** — stores paper notes, ideas, experiments, and verification logs as
  JSONL under `data/research_memory/`.
- **Action Space** — includes local document analysis, evidence search, idea
  generation, experiment design, and JSON/Markdown report writing.
- **Verifier** — checks evidence grounding, similarity to saved ideas,
  experiment completeness, and reproducibility information. A failed check
  triggers one bounded revision.

The MVP reads `.txt`, `.md`, and text-based `.pdf` files already present under
`data/`. It does not require a remote LLM or API key. The idea generator accepts
an optional LLM dependency so structured model generation can be added later
without changing the agent workflow.

Python 3.11 or newer is required, matching the existing project's use of
`tomllib` and modern type annotations.

### Run the demo

From the repository root:

```bash
python app/ai_scientific_demo.py --topic "LVLM hallucination mitigation"
```

Another topic that matches the included demo paper is:

```bash
python app/ai_scientific_demo.py --topic "LLM Agent Memory"
```

Optional output directory:

```bash
python app/ai_scientific_demo.py \
  --topic "multimodal model reliability evaluation" \
  --output-dir outputs/my_scientific_run
```

The default outputs are:

```text
outputs/ai_scientific_agent/
  result.json
  report.md
```

`result.json` contains the task type, structured plan, evidence excerpts,
literature analysis, candidate ideas, selected idea, experiment plan, four
verification results, revision flag, and output paths. `report.md` presents the
same information for human review.

### Current limitations

- Evidence retrieval is lexical and limited to local project documents and
  saved memory; it is not an online literature search.
- Generated ideas and their ranking are heuristic templates in this MVP.
- Dataset and baseline suggestions are starting points and must be checked
  against current papers before running an experiment.
- PDF support is limited to files with extractable text; OCR is not included.
- Novelty checking only measures token overlap against local `ideas.jsonl`.

### Run tests

```bash
python -m unittest discover -s tests -v
```

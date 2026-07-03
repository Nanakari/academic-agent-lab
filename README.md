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
- **Action Space** — includes local paper corpus indexing, evidence search,
  document analysis, idea generation, experiment design, and JSON/Markdown
  report writing.
- **Verifier** — checks evidence grounding, similarity to saved ideas,
  experiment completeness, and reproducibility information. A failed check
  triggers one bounded revision.

The MVP reads `.txt`, `.md`, and text-based `.pdf` papers under `data/papers/`.
It does not require a remote LLM or API key. The idea generator accepts an
optional LLM dependency so structured model generation can be added later
without changing the agent workflow.

Python 3.11 or newer is required, matching the existing project's use of
`tomllib` and modern type annotations.

### Run the demo

From the repository root:

```bash
python app/ai_scientific_demo.py \
  --topic "LVLM hallucination mitigation" \
  --papers-dir data/papers \
  --top-k 5
```

Place local papers in the corpus directory before running:

```text
data/papers/
  paper_one.txt
  paper_two.md
  paper_three.pdf
```

Optional output directory:

```bash
python app/ai_scientific_demo.py \
  --topic "multimodal model reliability evaluation" \
  --papers-dir data/papers \
  --top-k 5 \
  --output-dir outputs/my_scientific_run
```

The default outputs are:

```text
outputs/ai_scientific_agent/
  result.json
  report.md
```

`result.json` contains the task type, structured plan, evidence excerpts,
evidence status, evidence gaps, unsupported claims, literature analysis,
candidate ideas, selected idea, experiment plan, four verification results,
revision flag, and output paths. `report.md` presents the same information for
human review.

The Agent searches `data/papers/` first and fills remaining evidence slots from
ScientificMemory. If neither source contains sufficiently relevant evidence,
the result uses `evidence_status: "evidence_insufficient"` and
EvidenceVerifier fails intentionally. This prevents an exploratory idea from
being reported as paper-supported.

### Structured Evidence Citation

Place local `.txt`, `.md`, or text-based `.pdf` papers in `data/papers/`. The
corpus indexer splits them into `EvidenceChunk` records that retain the detected
section, PDF page when available, source path, chunk ID, matched keywords,
supporting sentence, retrieval score, and support level.

The Markdown report connects generated claims to local chunks under **Evidence
Used**, and separately lists **Evidence Gaps** and **Unsupported Claims**. These
citations are lightweight local provenance records, not formal BibTeX
references. Future versions can add embedding retrieval, BibTeX management, and
venue-specific citation formatting without changing the current report
contract.

### Current limitations

- Evidence retrieval uses lightweight keyword overlap over local papers and
  saved memory; it is not an online literature search.
- The corpus index is rebuilt in memory for each run. A later version can
  replace lexical scoring with cached embeddings or the existing RAG stack.
- Generated ideas and their ranking are heuristic templates in this MVP.
- Dataset and baseline suggestions are starting points and must be checked
  against current papers before running an experiment.
- PDF support is limited to files with extractable text; OCR is not included.
- Novelty checking only measures token overlap against local `ideas.jsonl`.

### Run tests

```bash
pytest -q
```

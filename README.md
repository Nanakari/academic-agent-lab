# academic-agent-lab

academic-agent-lab is a lightweight scientific assistant agent MVP. Its default
CLI path uses an LLM tool-decision step, while the executed tools remain
offline-first and verifier-driven with optional external evidence retrieval.

## Project Overview

This project began as an `AcademicAgent` for local paper reading, summarization,
and RAG question answering. The current mainline is an LLM-driven,
offline-first, verifier-driven `AIScientificAgent` that follows a broader AI
research workflow:

1. classify and plan a research task;
2. parse and retrieve evidence from local TXT, Markdown, and text-based PDF
   papers;
3. identify methods, limitations, and research gaps;
4. generate and rank research ideas;
5. design datasets, baselines, metrics, ablations, and risk analysis;
6. verify evidence support, novelty, experiment completeness, and
   reproducibility;
7. write structured JSON/Markdown reports and run fixture-based evaluation.

The original tool-calling `AcademicAgent` remains available as an explicit
legacy demo through `python main.py legacy-academic`. The scientific mode is
the default project entry point.

Python 3.11 or newer is required.

The project is a bounded AI Scientific Agent and offline-first
pre-experiment planning assistant. Its verifier-driven decision policy records
observations, decisions, actions, reasons, and outcomes in `agent_trace`. This
is a bounded rule-based trace, not open-ended autonomous discovery: the system
does not run real experiments or prove scientific hypotheses. Failed verifier
results are preserved and reported instead of being silently corrected.
The trace also records whether the agent skipped or triggered a bounded
revision after the first verifier pass.

The scientific mode also separates lightweight research ideas from structured
research directions. A direction records its source idea, target gap,
hypothesis, method sketch, evidence support, risks, heuristic priority, and
next steps. Only the direction tied to the selected idea inherits final
verifier results; other candidate directions remain explicitly unverified.
This planning-stage heuristic does not prove novelty or feasibility.

The selected direction also receives a planning-readiness assessment based on
local evidence, verifier outputs, and the selected experiment plan. It
summarizes readiness, explicit resource signals, risks, mitigations, and a
minimum viable experiment for human review. Its score is checklist-oriented,
not a probability of feasibility, and it does not trigger experiment execution.

The agent also builds a human-reviewable pilot planning protocol for the
selected direction. The blueprint separates planning and future experiment
artifacts and records objectives, criteria, reproducibility checks, and
pre-execution blockers. It does not execute experiments, download datasets, or
train models; human approval is always required and execution remains disabled.

For result compatibility, `overall_score` is a deprecated alias of
`planning_readiness_score` and may be removed in a future version.

## Architecture

```text
User Query
  ↓
ResearchPlanner
  ↓
PaperCorpusIndexer / ScientificMemory
  ↓
ResearchIdeaGenerator / ExperimentDesigner
  ↓
EvidenceVerifier / NoveltyVerifier
ExperimentVerifier / ReproducibilityVerifier
  ↓
ReportWriter / Evaluation
```

The implementation follows the Scientific Agent pattern:

- **Planner** creates a structured `ResearchPlan`.
- **Memory** persists paper notes, ideas, experiments, and verification logs in
  local JSONL files.
- **Action Space** provides paper parsing, retrieval, analysis, idea generation,
  experiment design, and report writing.
- **Verifier** checks evidence grounding, novelty overlap, experiment
  completeness, and reproducibility before results are saved.

More detail is available in [architecture.md](docs/architecture.md) and
[scientific_agent_design.md](docs/scientific_agent_design.md).

## Features

- **Paper parsing:** TXT, Markdown, and text-based PDF support
- **Local evidence retrieval:** explainable keyword-based paper search
- **Optional external evidence:** controlled arXiv metadata/abstract and GitHub
  repository retrieval with a local audit cache
- **Structured evidence citation:** paper, section/page, chunk, keywords, and
  support level
- **Research idea generation:** three ranked, testable candidate ideas
- **Experiment plan generation:** datasets, baselines, metrics, ablations, and
  risks
- **Verifier-based reliability check:** evidence, novelty, experiment, and
  reproducibility verification
- **Scientific evaluation:** deterministic positive and negative fixture cases
- **Real paper validation:** one command for local corpus validation and summary

## Project Structure

```text
app/
  agent/        # Agent lifecycle and AIScientificAgent orchestration
  planner/      # Task classification and structured research plans
  memory/       # JSONL paper, idea, experiment, and verification memory
  tools/        # Paper parsing, retrieval, idea, experiment, and report tools
  verifier/     # Evidence, novelty, experiment, and reproducibility checks
  evaluation/   # Eval cases, metrics, runners, and validation summaries
  schemas/      # Structured task, evidence, experiment, and evaluation models
docs/           # Architecture, scientific-agent, and evaluation design notes
examples/       # Short report excerpts and resume-ready project descriptions
tests/          # Regression tests and deterministic paper fixtures
```

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For development and tests, install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Create local configuration when you need to edit settings or run the default
LLM-driven scientific agent:

```bash
cp config.example.toml config.toml
```

Windows PowerShell:

```powershell
Copy-Item config.example.toml config.toml
```

If `config.toml` is absent, application config loading falls back to the
tracked `config.example.toml`. `config.toml` is ignored by Git and should stay
local.

Set a Gemini API key for the default agent mode:

```bash
export GEMINI_API_KEY=...
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="..."
```

Run all tests:

```bash
pytest -q
```

Run the default top-level entry point:

```bash
python main.py
```

By default, this calls the LLM API first so the model can choose bounded tools
such as local evidence search, scientific memory, optional external metadata
retrieval, and verification. For CI or offline regression checks, disable the
LLM decision step explicitly:

```bash
python main.py --offline
```

Run the fixture-backed AI Scientific Agent demo:

```bash
python app/ai_scientific_demo.py --topic "LVLM hallucination mitigation" --papers-dir tests/fixtures/papers --top-k 5
```

Offline regression form:

```bash
python app/ai_scientific_demo.py --topic "LVLM hallucination mitigation" --papers-dir tests/fixtures/papers --top-k 5 --offline
```

Run the Streamlit Web UI from the project root:

```bash
streamlit run app/frontend/streamlit_app.py
```

The page accepts PDF, TXT, and Markdown uploads, saves them under
`data/papers/`, and exposes local retrieval and optional arXiv/GitHub search
controls without changing the core `AIScientificAgent` workflow. User-facing
labels, summaries, verification explanations, traces, and the generated
`outputs/ai_scientific_agent/report_zh.md` are rendered in Chinese. The
original `result.json` and English `report.md` remain unchanged and available
for auditing and downstream use.

The demo uses temporary isolated ScientificMemory, so running the same Quick
Start command repeatedly does not create novelty failures from earlier demo
runs. Application code can still use persistent `data/research_memory/`.

Run the legacy `AcademicAgent` demo only when you specifically need the
original LLM tool-calling example:

```bash
python main.py legacy-academic --paper-path data/demo_paper.pdf
```

This legacy path requires `GEMINI_API_KEY` or an `api_key` value in local
`config.toml`. The default `AIScientificAgent` CLI also requires an API key
because it asks the LLM to decide which bounded tools to use. Use `--offline`
only when you intentionally want the deterministic local regression path.

Run Evaluation Mode:

```bash
python app/scientific_eval_demo.py
```

Run Real Paper Validation:

```bash
python app/real_paper_validation_demo.py --topic "LVLM hallucination mitigation" --papers-dir data/papers --top-k 8
```

Local outputs are written below `outputs/` and are ignored by Git.
Place private/local papers under `data/papers/`; that directory is also ignored
except for `.gitkeep`. `config.toml` is local configuration and must not be
committed. Use the tracked `config.example.toml` as a safe template; if
`config.toml` is missing, `load_config()` falls back to the example template.

## AI Scientific Agent Mode

Place local papers in:

```text
data/papers/
  paper_one.txt
  paper_two.md
  paper_three.pdf
```

The Agent searches `data/papers/` first and then paper-derived notes in
ScientificMemory. If neither source provides adequate evidence, the result uses
`evidence_status: "evidence_insufficient"` and EvidenceVerifier fails
intentionally.

Retrieval scores are driven primarily by keywords found in each chunk body.
Paper-title matches contribute only a small bonus (at most 0.1) and cannot turn
an unrelated body chunk into strong evidence. This remains lightweight lexical
retrieval, not semantic relevance or entailment checking.

Within one `PaperCorpusIndexer` instance, the parsed index is cached in memory
and refreshed when supported files or chunk settings change. ScientificMemory
also applies exact-key deduplication to ideas and paper notes and warns on
malformed JSONL lines. These safeguards improve repeated local runs, but they
are not a production database, distributed cache, or vector index.

Domain consistency is a lightweight lexical/concept coverage check with
`off`, `warning`, and `strict` modes. It is not semantic entailment,
embedding-based retrieval, or proof that a scientific conclusion is correct.
Real Paper Validation uses strict mode to expose topic mismatch; this can be
conservative when relevant papers use substantially different terminology.
Warning mode is diagnostic only and does not mean that evidence is sufficient.

Default scientific-agent outputs:

```text
outputs/ai_scientific_agent/
  result.json
  report.md
```

### Structured Evidence Citation

Each `EvidenceChunk` retains the paper ID, title, source path, file type,
detected section, PDF page where available, chunk ID, matched keywords,
supporting sentence, score, and support level. Reports map ideas and key claims
back to these chunks under:

- **Evidence Used**
- **Claim-to-Evidence Citations**
- **Evidence Gaps**
- **Unsupported Claims**

These citations are lightweight local provenance records, not formal BibTeX
references.

### External Evidence Retrieval

The agent can optionally retrieve external evidence from arXiv and GitHub.
External search is disabled by default, so the normal pipeline remains
offline-first:

```bash
python app/ai_scientific_demo.py \
  --topic "scientific agent verifier" \
  --use-external-search
```

Available controls are:

```text
--use-external-search
--no-external-search
--external-sources arxiv,github
--external-max-results 5
--external-force-refresh
```

GitHub repository search works without authentication but is more
rate-limited. To authenticate:

```bash
export GITHUB_TOKEN=...
```

Windows PowerShell:

```powershell
$env:GITHUB_TOKEN="..."
```

External results are cached as JSON below `data/external_cache/`. Cached
records include the query, source, retrieval time, normalized items, and
warnings so a run can be audited and repeated. Cache keys include the source,
actual source-specific query, result limit, and cache schema version. Successful
zero-result searches may be cached, but failed network/API retrievals are not.
Network or cache failures are reported in `result.json` and `report.md` and do
not stop the local pipeline.

The result distinguishes the current `run_at` time, the original
`retrieved_at_by_source` times, and `cache_loaded_at`. It also records the
actual arXiv and GitHub queries separately; GitHub queries receive an
implementation-oriented suffix.

Limitations:

- arXiv retrieval is metadata / abstract-level only; PDFs are not downloaded.
- GitHub repositories indicate implementation availability or engineering
  relevance and are never treated as scientific proof.
- External results can change over time; `retrieved_at` is recorded.
- No repository is cloned and no external code or experiment is executed.

## Evaluation Mode

Evaluation uses deterministic fixture cases to test retrieval, verifier
behavior, experiment completeness, and citation traceability:

```bash
python app/scientific_eval_demo.py \
  --cases tests/fixtures/eval_cases.json \
  --output outputs/evaluation/evaluation_report.md
```

Metrics include `evidence_count`, `keyword_hit_rate`, `section_hit_rate`,
`verifier_pass_match`, `experiment_completeness`, `citation_completeness`, and
a weighted `overall_score`. See
[evaluation_design.md](docs/evaluation_design.md).

Fixture evaluation is a regression check. It does not establish real scientific
reasoning ability.

## Real Paper Validation

Real Paper Validation runs the same Agent and evaluation metrics against papers
that you place locally in `data/papers/`:

```bash
python app/real_paper_validation_demo.py \
  --topic "LVLM hallucination mitigation" \
  --papers-dir data/papers \
  --top-k 8
```

Outputs:

```text
outputs/real_paper_validation/
  result.json
  report.md
  evaluation_result.json
  validation_summary.md
```

Real papers are intentionally ignored by Git. If the directory is empty, the
command exits cleanly and asks you to add TXT/MD/PDF papers.

## Example Output

```text
Selected idea:
  Evidence-aware adaptive intervention for LVLM hallucination mitigation

Evidence used:
  Grounded LVLM Hallucination Mitigation / Method / chunk C2 / strong

Unsupported claims:
  None detected by the lightweight verifier

Verifier result:
  evidence=PASS, experiment=PASS, reproducibility=PASS
```

Actual results depend on the local paper collection and may include explicit
evidence gaps or unsupported claims.

Short tracked examples:

- [Fixture report excerpt](examples/fixture_report_excerpt.md)
- [Evaluation report excerpt](examples/evaluation_report_excerpt.md)
- [Resume and interview descriptions](examples/resume_description.md)

## Reproducibility

- Tests and evaluation cases are stored under `tests/fixtures/`.
- Evaluation cases declare expected keywords, sections, evidence count,
  verifier behavior, and required experiment fields.
- Agent outputs record datasets, baselines, metrics, ablations, risks, and
  implementation notes.
- Evidence citations retain local source, section/page, chunk, score, and
  support level.
- Runtime outputs, research memory, and local papers are excluded from commits.

## Limitations

- This is not a fully autonomous research system.
- Generated ideas are hypotheses; the system cannot prove that an idea is
  genuinely novel.
- Evidence retrieval depends entirely on the quality and coverage of local
  papers.
- Retrieval is lightweight lexical matching rather than a complete literature
  search.
- Local evidence citations are not formal BibTeX references.
- PDF support requires extractable text and does not include OCR.
- Fixture evaluation does not represent real-world scientific capability.
- Suggested datasets and baselines must be checked before running experiments.

## Future Work

- embedding-based and hybrid retrieval
- BibTeX citation management and formal reference formatting
- larger, multi-topic scientific-agent benchmarks
- human expert evaluation of ideas and experiment plans
- multi-agent debate, critique, and evidence adjudication
- sandboxed code generation and experiment execution

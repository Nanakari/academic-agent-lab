# Scientific Agent Design

## Beyond Paper RAG

RAG answers a question by retrieving relevant passages and generating a
response. This project uses retrieval, but its goal is a broader scientific
workflow: identify a research gap, propose testable hypotheses, design an
experiment, challenge claims, record failures, and produce a traceable report.

The Agent is still an MVP. “Scientific” describes the workflow and verification
contract, not a claim that the system can autonomously perform reliable
research.

## Four-Module Mapping

### Planner

`ResearchPlanner` classifies paper reading, literature analysis, idea
generation, experiment design, and research proposal requests. It emits
structured steps, required tools, and expected outputs before execution.

### Memory

`ScientificMemory` stores paper notes, ideas, experiments, and verification
logs as inspectable JSONL. Only paper-derived notes can be reused as scientific
evidence, preventing generated ideas from recursively becoming “paper facts.”

### Action Space

The action space includes document parsing, corpus indexing, lexical retrieval,
paper analysis, idea generation, experiment design, and report writing. The
components use replaceable interfaces and do not require a remote LLM.

Lexical retrieval is body-first: matched keywords come from chunk text, while a
matching paper title contributes only a small ranking bonus. This reduces title
leakage but does not provide semantic relevance guarantees.

### Verifier

EvidenceVerifier maps each idea and key claim to its strongest local chunk and
labels support as strong, moderate, weak, or insufficient. The other verifiers
check historical idea overlap, experiment fields, and reproducibility details.
Failure triggers at most one revision.

### Domain Consistency Design Notes

Domain consistency is lightweight lexical/concept coverage, not
embedding-based semantic search or LLM entailment. Broad n-gram overlap is
reported as diagnostic coverage, while strict acceptance still requires a
topic-critical phrase or multi-term group in one evidence item.

The ordinary demo leaves this check off. Warning mode records terminology
mismatches without failing verification, while Real Paper Validation uses
strict mode to make negative controls visible. Strict mode may be conservative
when relevant papers use substantially different terminology.

EvidenceVerifier is a reliability check over local evidence provenance. It
does not prove that a generated idea is novel or that a scientific conclusion
is true. The current system is intended for learning, demonstrations, offline
prototyping, and explainable validation rather than production literature
review.

## Why Structured Evidence Matters

A score alone is not an adequate scientific citation. `EvidenceChunk` retains:

- paper and source identity;
- file type;
- section and PDF page where available;
- stable chunk ID;
- matched keywords and score;
- the supporting sentence;
- support level.

This makes claims auditable in a local report and exposes evidence gaps instead
of hiding them behind fluent generation. It is a provenance mechanism, not a
replacement for BibTeX or formal citation review.

# Evaluation Design

## Evaluation Cases

An `EvalCase` defines a topic, paper directory, expected keywords and sections,
minimum evidence count, required experiment fields, and the expected
EvidenceVerifier decision.

The bundled cases cover:

- a positive LVLM hallucination case with local paper fixtures;
- an empty-corpus case where verifier failure is the correct behavior.

Each evaluation case runs with isolated temporary ScientificMemory so previous
ideas or paper notes cannot change the expected outcome.

## Metrics

- `evidence_count`: number of retrieved chunks.
- `keyword_hit_rate`: proportion of expected terms found in retrieved evidence.
- `section_hit_rate`: proportion of expected sections represented.
- `verifier_pass_match`: whether the evidence verifier decision matches the
  case expectation.
- `experiment_completeness`: proportion of required experiment fields present.
- `citation_completeness`: whether evidence, unsupported-claim, and claim-level
  citation records are populated appropriately.
- `overall_score`: equal-weight average of keyword hit, verifier match,
  experiment completeness, and citation completeness.

The Markdown report includes aggregate results, per-case metrics, main issues,
and failure analysis.

## Limitations

Fixture evaluation is deterministic regression testing, not a scientific
benchmark. The fixtures are small, keyword overlap is easy to satisfy, and
there is no human judgment of novelty, usefulness, or experimental validity.

A stronger evaluation would use a larger real-paper corpus, diverse AI topics,
adversarial negative cases, formal citation checks, expert ratings, and
comparisons against retrieval and proposal-generation baselines.

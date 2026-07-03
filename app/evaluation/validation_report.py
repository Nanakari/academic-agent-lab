"""Human-readable summary for real local-paper validation runs."""

from __future__ import annotations

from pathlib import Path


class RealPaperValidationReportWriter:
    """Write a compact showcase summary from agent and evaluation results."""

    def write(
        self,
        *,
        topic: str,
        papers_dir: str | Path,
        top_k: int,
        scanned_papers: int,
        agent_result: dict,
        metrics: dict,
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence = agent_result.get("evidence_context", [])
        strongest = max(
            evidence,
            key=lambda item: float(item.get("score", 0.0)),
            default=None,
        )
        selected_idea = agent_result.get("selected_idea", {})
        experiment = agent_result.get("experiment_plan", {})
        unsupported = agent_result.get("unsupported_claims", [])
        gaps = agent_result.get("evidence_gaps", [])

        lines = [
            "# Real Paper Validation Summary",
            "",
            "## Input",
            "",
            f"- Topic: {topic}",
            f"- Papers directory: `{Path(papers_dir)}`",
            f"- Top-k: {top_k}",
            f"- Number of scanned papers: {scanned_papers}",
            f"- Number of evidence chunks: {len(evidence)}",
            "",
            "## Agent Output Summary",
            "",
            f"- Selected idea: {selected_idea.get('title', 'N/A')}",
            f"- Experiment datasets: {self._join(experiment.get('datasets'))}",
            f"- Baselines: {self._join(experiment.get('baselines'))}",
            f"- Metrics: {self._join(experiment.get('metrics'))}",
            f"- Verifier passed: {agent_result.get('verification_passed', False)}",
            "",
            "## Evidence Summary",
            "",
            f"- Evidence count: {len(evidence)}",
            "- Strongest evidence: " + self._strongest_evidence(strongest),
            f"- Weak or unsupported claims: {self._join(unsupported)}",
            f"- Evidence gaps: {self._join(gaps)}",
            "",
            "## Evaluation Summary",
            "",
            f"- Keyword hit rate: {float(metrics.get('keyword_hit_rate', 0.0)):.3f}",
            f"- Verifier pass match: {metrics.get('verifier_pass_match', False)}",
            (
                "- Experiment completeness: "
                f"{float(metrics.get('experiment_completeness', 0.0)):.3f}"
            ),
            (
                "- Citation completeness: "
                f"{float(metrics.get('citation_completeness', 0.0)):.3f}"
            ),
            f"- Overall score: {float(metrics.get('overall_score', 0.0)):.3f}",
            "",
            "## Notes",
            "",
            "- Results depend on the quality and topical coverage of local data/papers.",
            "- Evidence gaps are expected when the local paper collection is small.",
            (
                "- Retrieval currently uses lightweight keyword overlap; "
                "it is not a complete literature review."
            ),
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _join(values) -> str:
        if not values:
            return "None"
        if isinstance(values, str):
            return values
        return "; ".join(str(value) for value in values)

    @staticmethod
    def _strongest_evidence(evidence: dict | None) -> str:
        if not evidence:
            return "None"
        return (
            f"{evidence.get('title', 'Unknown paper')} / "
            f"{evidence.get('section') or 'N/A'} / "
            f"chunk {evidence.get('chunk_id', 'N/A')} / "
            f"score {float(evidence.get('score', 0.0)):.3f}"
        )

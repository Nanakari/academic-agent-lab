"""Human-readable summary for real local-paper validation runs."""

from __future__ import annotations

from pathlib import Path

from app.tools.paper_corpus import keyword_tokens


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
        support_distribution = {
            level: sum(
                item.get("support_level") == level for item in evidence
            )
            for level in ("strong", "moderate", "weak", "insufficient")
        }
        expected_keywords = keyword_tokens(topic)
        evidence_keywords = keyword_tokens(
            " ".join(
                f"{' '.join(item.get('matched_keywords', []))} "
                f"{item.get('text', '')}"
                for item in evidence
            )
        )
        covered_keywords = sorted(expected_keywords & evidence_keywords)
        missing_keywords = sorted(expected_keywords - evidence_keywords)
        evidence_verification = (
            agent_result.get("verification", {}).get("evidence", {})
        )
        domain_consistency = evidence_verification.get("domain_consistency", {})
        strongest_score = float((strongest or {}).get("score", 0.0))
        coverage_warning = (
            len(evidence) < 5
            or (
                not evidence_verification.get("passed", False)
                and strongest_score >= 0.6
            )
        )
        weak_only_warning = bool(evidence) and all(
            item.get("support_level") == "weak" for item in evidence
        )

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
            "### Evidence support distribution",
            "",
            f"- strong: {support_distribution['strong']}",
            f"- moderate: {support_distribution['moderate']}",
            f"- weak: {support_distribution['weak']}",
            f"- insufficient: {support_distribution['insufficient']}",
            "",
            "### Topic keyword coverage",
            "",
            f"- Expected keywords: {self._join(sorted(expected_keywords))}",
            f"- Covered keywords: {self._join(covered_keywords)}",
            f"- Missing keywords: {self._join(missing_keywords)}",
            "",
            "### Strongest evidence detail",
            "",
            self._strongest_evidence_detail(strongest),
            "",
            "### Domain consistency",
            "",
            f"- Passed: {domain_consistency.get('passed', 'Not enabled')}",
            f"- Mode: {domain_consistency.get('mode', 'off')}",
            (
                "- Matched topic concepts: "
                f"{self._join(domain_consistency.get('matched_topic_concepts'))}"
            ),
            (
                "- Missing topic concepts: "
                f"{self._join(domain_consistency.get('missing_topic_concepts'))}"
            ),
            f"- Issues: {self._join(domain_consistency.get('issues'))}",
            f"- Warnings: {self._join(domain_consistency.get('warnings'))}",
            f"- Reason: {domain_consistency.get('reason', 'Not evaluated')}",
            (
                "- Verifier warnings: "
                f"{self._join(evidence_verification.get('warnings'))}"
            ),
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
        ]
        if coverage_warning:
            lines.append(
                "- Evidence coverage may be incomplete; consider increasing top_k "
                "or adding more papers."
            )
        if weak_only_warning:
            lines.append(
                "- All retrieved evidence is weak; treat the result as exploratory."
            )
        lines.append("")
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

    @classmethod
    def _strongest_evidence_detail(cls, evidence: dict | None) -> str:
        if not evidence:
            return "- None"
        return "\n".join([
            f"- Title: {evidence.get('title', 'Unknown paper')}",
            f"- Section: {evidence.get('section') or 'N/A'}",
            f"- Chunk: {evidence.get('chunk_id', 'N/A')}",
            f"- Score: {float(evidence.get('score', 0.0)):.3f}",
            f"- Support level: {evidence.get('support_level', 'unknown')}",
            (
                "- Matched keywords: "
                f"{cls._join(evidence.get('matched_keywords'))}"
            ),
        ])

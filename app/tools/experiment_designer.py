"""Experiment-plan construction for candidate AI research ideas."""

from __future__ import annotations

import json

from app.agent.services.llm_json import (
    limited_evidence_payload,
    list_of_strings,
    parse_json_object,
)
from app.agent.services.llm_scientific_analysis_service import LLMStageResult
from app.schemas.experiment_plan import ExperimentPlan
from app.schemas.research_idea import ResearchIdea


class ExperimentDesigner:
    """Select domain-aware evaluation ingredients and expose assumptions."""

    def __init__(self, llm=None, enabled: bool = False) -> None:
        self.llm = llm
        self.enabled = bool(enabled and llm is not None)
        self.last_stage_result = LLMStageResult(
            stage="experiment_design",
            deterministic_sections=["experiment_plan"],
        )

    def design_experiment(
        self,
        idea: ResearchIdea,
        topic: str,
        evidence_context: list[dict] | None = None,
    ) -> ExperimentPlan:
        if self.enabled:
            try:
                plan = self._design_with_llm(idea, topic, evidence_context or [])
                self.last_stage_result = LLMStageResult(
                    stage="experiment_design",
                    llm_used=True,
                    generated_sections=["experiment_plan"],
                )
                return plan
            except Exception as exc:
                self.last_stage_result = LLMStageResult(
                    stage="experiment_design",
                    fallback_used=True,
                    deterministic_sections=["experiment_plan"],
                    warning=str(exc),
                )
        else:
            self.last_stage_result = LLMStageResult(
                stage="experiment_design",
                deterministic_sections=["experiment_plan"],
            )
        return self._design_deterministic(idea, topic)

    def _design_deterministic(self, idea: ResearchIdea, topic: str) -> ExperimentPlan:
        datasets, baselines, metrics = self._domain_defaults(topic)
        attack_scenarios: list[str] = []
        tool_schemas: list[str] = []
        failure_taxonomy: list[str] = []
        risks = [
            "The uncertainty signal may be poorly calibrated under domain shift.",
            "Benchmark contamination or annotation artifacts may inflate apparent gains.",
            "Additional inference stages may increase latency and compute cost.",
        ]
        implementation_notes = [
            "Pin model, dataset, and dependency versions.",
            "Run at least three random seeds and publish all evaluation prompts/configuration.",
            "Record hardware, decoding parameters, and preprocessing steps.",
        ]
        if self._is_tool_call_safety_topic(topic):
            risks = [
                "Overly strict defenses may increase false refusal on benign tool-use tasks.",
                "Attack templates may overfit to a narrow prompt-injection style.",
                "Verifier or routing calls may add measurable latency and API cost.",
            ]
            attack_scenarios = [
                "Indirect prompt injection in retrieved web pages",
                "Malicious tool output that attempts to change later instructions",
                "Tool-argument injection against file, browser, or shell tools",
            ]
            tool_schemas = [
                "search_web(query: string)",
                "read_url(url: string)",
                "execute_tool(name: string, arguments: object)",
            ]
            failure_taxonomy = [
                "unsafe tool invocation",
                "unsafe argument construction",
                "benign task refusal",
                "missed attack detection",
            ]
            implementation_notes = [
                "Attack scenarios: indirect prompt injection in retrieved web pages, malicious tool output, tool-argument injection, and cross-step instruction smuggling.",
                "Defense mechanisms: tool-call intent classification, argument allowlists, evidence-grounded verification before execution, and human approval for high-risk tools.",
                "Benchmark types: synthetic adversarial tool-use suites, realistic benign tool workflows, and mixed attack/benign regression sets.",
                "Baseline types: unprotected tool-calling agent, prompt-only safety instruction, rule-based allowlist/denylist guard, and verifier-gated tool router.",
                "Pin model, tool schema, prompts, seeds, and cost accounting configuration.",
            ]

        return ExperimentPlan(
            idea_title=idea.title,
            method=idea.method,
            datasets=datasets,
            baselines=baselines,
            metrics=metrics,
            attack_scenarios=attack_scenarios,
            tool_schemas=tool_schemas,
            ablation=[
                "Remove the proposed intervention/verification component.",
                "Replace adaptive routing with always-on routing at matched compute.",
                "Vary detector threshold and evidence-context size.",
            ],
            failure_taxonomy=failure_taxonomy,
            expected_results=[
                "Primary reliability metric improves over the strongest matched baseline.",
                "Adaptive routing retains most gains with lower average inference cost.",
                "Report mean, standard deviation, and per-failure-category results.",
            ],
            risks=risks,
            implementation_notes=implementation_notes,
            reproducibility_notes=[
                "Pin model, prompt, tool schema, dataset, and dependency versions.",
                "Store raw predictions, tool-call logs, verifier decisions, and cost traces.",
            ],
        )

    def _design_with_llm(
        self,
        idea: ResearchIdea,
        topic: str,
        evidence_context: list[dict],
    ) -> ExperimentPlan:
        payload = {
            "topic": topic,
            "selected_idea": idea.to_dict(),
            "evidence": limited_evidence_payload(evidence_context),
        }
        raw = self.llm.ask([
            {
                "role": "system",
                "content": (
                    "You design bounded, reproducible AI safety experiments. "
                    "Return only JSON. Do not claim results; propose an "
                    "experiment plan for human review."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return a JSON object with required keys: attack_scenarios, "
                    "tool_schemas, datasets, baselines, metrics, ablations, "
                    "failure_taxonomy, reproducibility_notes, risks, "
                    "expected_results.\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ])
        parsed = parse_json_object(raw)
        datasets = list_of_strings(parsed.get("datasets"))
        baselines = list_of_strings(parsed.get("baselines"))
        metrics = list_of_strings(parsed.get("metrics"))
        ablations = list_of_strings(parsed.get("ablations"))
        risks = list_of_strings(parsed.get("risks"))
        if not all((datasets, baselines, metrics, ablations, risks)):
            raise ValueError("LLM experiment plan missed required verifier fields.")
        reproducibility_notes = list_of_strings(parsed.get("reproducibility_notes"))
        return ExperimentPlan(
            idea_title=idea.title,
            method=idea.method,
            datasets=datasets,
            baselines=baselines,
            metrics=metrics,
            attack_scenarios=list_of_strings(parsed.get("attack_scenarios")),
            tool_schemas=list_of_strings(parsed.get("tool_schemas")),
            ablation=ablations,
            failure_taxonomy=list_of_strings(parsed.get("failure_taxonomy")),
            expected_results=list_of_strings(parsed.get("expected_results")),
            risks=risks,
            implementation_notes=reproducibility_notes
            or ["Pin model, prompt, tool schema, dataset, and dependency versions."],
            reproducibility_notes=reproducibility_notes,
        )

    @staticmethod
    def _domain_defaults(topic: str) -> tuple[list[str], list[str], list[str]]:
        lowered = topic.casefold()
        if ExperimentDesigner._is_tool_call_safety_topic(topic):
            return (
                [
                    "Indirect prompt-injection tool-call benchmark",
                    "Benign multi-step tool-use task suite",
                    "Mixed adversarial/benign regression set",
                ],
                [
                    "Unprotected tool-calling agent",
                    "Prompt-only safety instruction baseline",
                    "Rule-based allowlist/denylist guard",
                    "Verifier-gated tool router",
                ],
                [
                    "attack success rate",
                    "safe tool-call rate",
                    "benign task success",
                    "false refusal",
                    "latency/cost",
                ],
            )
        if any(token in lowered for token in ("lvlm", "multimodal", "多模态", "幻觉")):
            return (
                ["POPE", "MMHal-Bench", "MME"],
                [
                    "Unmodified base LVLM",
                    "Prompt-only mitigation",
                    "Self-consistency verification",
                ],
                [
                    "Accuracy/F1",
                    "Hallucination rate",
                    "MMHal score",
                    "Latency and token cost",
                ],
            )
        if any(token in lowered for token in ("agent memory", "memory", "记忆")):
            return (
                ["LongMemEval", "LoCoMo", "A task-specific held-out interaction set"],
                ["No-memory agent", "Sliding-window memory", "Retrieval-only memory"],
                ["Task success", "Retrieval recall@k", "Answer F1", "Latency and token cost"],
            )
        return (
            [
                "Topic-matched public benchmark identified during literature review",
                "Out-of-domain robustness benchmark matched to the target failure mode",
                "Curated failure-case set with documented inclusion criteria",
            ],
            [
                "Unmodified base model",
                "Strong published baseline matched to the target task",
                "Compute-matched simple baseline",
            ],
            [
                "Task-specific success score",
                "Calibration/error rate",
                "Robustness under shift",
                "Latency and compute cost",
            ],
        )

    @staticmethod
    def _is_tool_call_safety_topic(topic: str) -> bool:
        lowered = topic.casefold()
        has_agent = any(token in lowered for token in ("agent", "代理", "智能体"))
        has_tool = any(token in lowered for token in ("tool", "function call", "工具"))
        has_safety = any(
            token in lowered
            for token in (
                "safety",
                "safe",
                "security",
                "attack",
                "prompt injection",
                "indirect prompt injection",
                "jailbreak",
                "安全",
                "攻击",
                "提示注入",
                "间接提示注入",
            )
        )
        has_llm = any(token in lowered for token in ("llm", "large language model", "大模型"))
        return (has_agent and (has_tool or has_safety)) or (
            has_llm and has_tool and has_safety
        )

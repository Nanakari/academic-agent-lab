"""LLM-driven tool planning for the scientific agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


SUPPORTED_EXTERNAL_SOURCES = {"arxiv", "github"}


@dataclass
class ScientificToolDecision:
    """A bounded tool plan produced by an LLM or deterministic fallback."""

    llm_used: bool
    mode: str
    use_local_evidence_search: bool = True
    use_scientific_memory: bool = True
    use_external_search: bool = False
    external_sources: list[str] = field(default_factory=list)
    top_k: int = 5
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_tool_call: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class LLMToolDecisionService:
    """Ask the LLM which bounded scientific tools should be used."""

    TOOL_NAME = "choose_scientific_agent_tools"

    def __init__(self, llm=None, enabled: bool = False) -> None:
        self.llm = llm
        self.enabled = bool(enabled and llm is not None)

    def decide(
        self,
        *,
        topic: str,
        default_top_k: int,
        external_search_enabled: bool,
        external_sources: list[str],
    ) -> ScientificToolDecision:
        fallback = self._fallback(
            default_top_k=default_top_k,
            external_search_enabled=external_search_enabled,
            external_sources=external_sources,
        )
        if not self.enabled:
            return fallback

        try:
            _, tool_calls = self.llm.ask_tool(
                messages=self._messages(
                    topic=topic,
                    default_top_k=default_top_k,
                    external_search_enabled=external_search_enabled,
                    external_sources=external_sources,
                ),
                tools=[self._tool_schema()],
            )
        except Exception as exc:
            fallback.warnings.append(f"LLM tool decision failed: {exc}")
            return fallback

        selected_call = next(
            (call for call in tool_calls if call.name == self.TOOL_NAME),
            None,
        )
        if selected_call is None:
            fallback.warnings.append(
                "LLM did not call choose_scientific_agent_tools; deterministic fallback used."
            )
            return fallback

        return self._from_arguments(
            selected_call.arguments,
            default_top_k=default_top_k,
            external_search_enabled=external_search_enabled,
            external_sources=external_sources,
            raw_tool_call=selected_call.to_dict(),
        )

    @staticmethod
    def _fallback(
        *,
        default_top_k: int,
        external_search_enabled: bool,
        external_sources: list[str],
    ) -> ScientificToolDecision:
        return ScientificToolDecision(
            llm_used=False,
            mode="deterministic_fallback",
            use_external_search=external_search_enabled,
            external_sources=list(external_sources if external_search_enabled else []),
            top_k=max(1, int(default_top_k)),
            reason=(
                "No LLM tool decision was used; the agent followed its bounded "
                "offline-first scientific workflow."
            ),
        )

    @classmethod
    def _from_arguments(
        cls,
        arguments: dict,
        *,
        default_top_k: int,
        external_search_enabled: bool,
        external_sources: list[str],
        raw_tool_call: dict,
    ) -> ScientificToolDecision:
        warnings = []
        requested_sources = [
            str(source).strip().casefold()
            for source in arguments.get("external_sources", [])
            if str(source).strip()
        ]
        invalid_sources = sorted(set(requested_sources) - SUPPORTED_EXTERNAL_SOURCES)
        if invalid_sources:
            warnings.append(
                "LLM requested unsupported external source(s): "
                + ", ".join(invalid_sources)
            )
        requested_sources = [
            source for source in requested_sources if source in SUPPORTED_EXTERNAL_SOURCES
        ]
        allowed_sources = set(external_sources)
        selected_sources = [
            source for source in requested_sources if source in allowed_sources
        ]
        if requested_sources and not selected_sources:
            warnings.append(
                "LLM requested external sources that are not enabled by agent configuration."
            )

        use_external = bool(arguments.get("use_external_search", False))
        if use_external and not external_sources:
            warnings.append(
                "LLM requested external search, but no external sources are allowed."
            )
            use_external = False
        if external_search_enabled:
            use_external = True
            if not selected_sources:
                selected_sources = list(external_sources)
        elif use_external and not selected_sources:
            selected_sources = list(external_sources)

        top_k = arguments.get("top_k", default_top_k)
        try:
            top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            warnings.append("LLM returned invalid top_k; default was used.")
            top_k = max(1, int(default_top_k))

        return ScientificToolDecision(
            llm_used=True,
            mode="llm_tool_decision",
            use_local_evidence_search=bool(
                arguments.get("use_local_evidence_search", True)
            ),
            use_scientific_memory=bool(arguments.get("use_scientific_memory", True)),
            use_external_search=use_external,
            external_sources=selected_sources if use_external else [],
            top_k=top_k,
            reason=str(arguments.get("reason") or "LLM selected scientific tools."),
            warnings=warnings,
            raw_tool_call=raw_tool_call,
        )

    @staticmethod
    def _messages(
        *,
        topic: str,
        default_top_k: int,
        external_search_enabled: bool,
        external_sources: list[str],
    ) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the tool-planning policy for a scientific assistant agent. "
                    "Choose only bounded tools. Local evidence and verifiers are preferred; "
                    "external search is supplemental and must not replace verification."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research topic: {topic}\n"
                    f"Default top_k: {default_top_k}\n"
                    f"External search currently enabled: {external_search_enabled}\n"
                    f"Allowed external sources: {external_sources}\n"
                    "Decide whether to use local evidence search, scientific memory, "
                    "and optional external metadata retrieval."
                ),
            },
        ]

    @classmethod
    def _tool_schema(cls) -> dict:
        return {
            "function": {
                "name": cls.TOOL_NAME,
                "description": (
                    "Choose the bounded scientific-agent tools to use before "
                    "running retrieval, planning, and verification."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "use_local_evidence_search": {
                            "type": "boolean",
                            "description": "Whether to search the local paper corpus.",
                        },
                        "use_scientific_memory": {
                            "type": "boolean",
                            "description": "Whether to allow paper-note memory fallback.",
                        },
                        "use_external_search": {
                            "type": "boolean",
                            "description": "Whether to retrieve supplemental arXiv/GitHub metadata.",
                        },
                        "external_sources": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["arxiv", "github"],
                            },
                            "description": "External sources to use if external search is selected.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of local evidence chunks to retrieve.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief rationale for the tool choices.",
                        },
                    },
                    "required": [
                        "use_local_evidence_search",
                        "use_scientific_memory",
                        "use_external_search",
                        "external_sources",
                        "top_k",
                        "reason",
                    ],
                },
            }
        }

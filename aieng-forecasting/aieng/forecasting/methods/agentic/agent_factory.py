"""Factory functions for building Google ADK agents for forecasting.

This module exposes :class:`AgentConfig` plus its nested
:class:`CodeExecutionConfig` and :class:`ContextRetrievalConfig` configs,
and the :func:`build_adk_agent` factory that turns a config into a fully
configured :class:`google.adk.agents.LlmAgent` (with optional E2B-backed
code execution and a proxy-grounded web-search tool for context retrieval).

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError` with installation guidance.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from aieng.forecasting.models import LITE_MODEL
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Suppress LiteLLM startup and OTEL noise
# ---------------------------------------------------------------------------
# LiteLLM logs Bedrock/SageMaker "no botocore" warnings and an OTEL proxy-
# server notice on every import — all harmless when using the Vector proxy.
# OTEL span-lifecycle warnings ("Tried calling set_status on an ended span")
# fire when LiteLLM callbacks run after spans close; also benign.
# These filters run at module-import time so they are active before the first
# litellm import (which happens lazily inside search_web / build_adk_agent).


class _LiteLLMNoiseFilter(logging.Filter):
    _NOISE = ("botocore", "Proxy Server is not installed")

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(n in record.getMessage() for n in self._NOISE)


logging.getLogger("LiteLLM").addFilter(_LiteLLMNoiseFilter())
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Tried calling set_status on an ended span")
warnings.filterwarnings("ignore", message="Setting attribute on ended span")


try:
    from aieng.agents.tools.code_interpreter import CodeInterpreter
    from google.adk.agents import LlmAgent
    from google.adk.skills import load_skill_from_dir
    from google.adk.skills.models import Skill
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.tools.skill_toolset import SkillToolset
    from google.adk.tools.tool_context import ToolContext
    from google.genai.types import (
        AutomaticFunctionCallingConfig,
        GenerateContentConfig,
        ThinkingConfig,
        ThinkingLevel,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


# Session-state key used by our proxy-compatible set_model_response shim.
# When a LiteLlm agent has both output_schema and tools, we register a flat
# set_model_response(json_response: str) tool that stores the JSON here.
# AdkTextRunner reads this key after each run and returns it in place of the
# final text, giving the predictor the structured JSON it expects.
SMR_STATE_KEY = "__smr_output__"


def _build_set_model_response_tool() -> FunctionTool:
    """Return a proxy-compatible ``set_model_response`` shim.

    Gemini thinking models call ``set_model_response`` when they produce
    structured output alongside other tools — regardless of whether ADK
    registered the tool.  The real ``SetModelResponseTool`` uses a nested
    Pydantic schema for its function declaration, which Gemini rejects via the
    OpenAI-compatible proxy (``$defs``/``$ref`` not supported).

    This shim accepts the JSON as a plain string and stores it in session
    state under :data:`SMR_STATE_KEY`.  :class:`AdkTextRunner` reads that key
    after the run and returns it as the final output, bypassing the model's
    subsequent "Done." text response.
    """

    async def set_model_response(json_response: str, tool_context: ToolContext) -> str:
        """Submit your final structured JSON response as a string.

        Call this tool once, passing the complete JSON object that satisfies
        the required output schema. Do not produce any further text after
        calling this tool.
        """
        tool_context.state[SMR_STATE_KEY] = json_response
        return "Response submitted. Task complete."

    return FunctionTool(set_model_response)


class ContextRetrievalConfig(BaseModel):
    """Configuration for the web-search context-retrieval tool.

    When enabled, :func:`build_adk_agent` attaches a ``search_web``
    :class:`~google.adk.tools.FunctionTool` to the agent backed by the
    Tavily search API.

    Temporal cutoff enforcement is soft (LLM-judgment-based): when
    ``enforce_cutoff`` is ``True`` and the calling agent passes a
    ``cutoff_date`` to the tool, the query is augmented with a
    ``before:<date>`` constraint.  Backtest leakage is a pedagogically
    useful discussion point, not a hard guarantee.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable context retrieval. Disabled by default.
    tavily_api_key : str | None, default=TAVILY_API_KEY env var
        API key for the Tavily search API. Falls back to the
        ``TAVILY_API_KEY`` environment variable when not set explicitly.
    instruction : str
        Unused by the Tavily tool directly, kept for config compatibility.
    enforce_cutoff : bool, default=True
        When ``True``, the ``search_web`` tool appends a ``before:<date>``
        constraint to the query whenever ``cutoff_date`` is supplied by the
        calling agent.  Set to ``False`` for live (non-backtest) agents
        where no temporal fence is needed.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    tavily_api_key: str | None = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    instruction: str = (
        "You are a specialized web search assistant.\n\n"
        "Search for information relevant to the query and return a concise, "
        "grounded summary with source URLs."
    )
    enforce_cutoff: bool = True


class CodeExecutionConfig(BaseModel):
    """Configuration for the E2B code execution tool.

    Code runs in an E2B-backed sandbox managed by the
    :class:`~aieng.agents.tools.code_interpreter.CodeInterpreter` tool.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable code execution. Disabled by default.
    template_name : str | None, default="agentic-forecasting-bootcamp"
        E2B template name.
    sandbox_timeout_seconds : int, default=3600
        E2B sandbox lifetime in seconds.
    code_execution_timeout_seconds : float | None, default=3300
        Per-execution timeout in seconds.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    template_name: str | None = "agentic-forecasting-bootcamp"
    sandbox_timeout_seconds: int = Field(default=3600, ge=1, le=3600)
    code_execution_timeout_seconds: float | None = Field(default=3300, gt=0)

    @model_validator(mode="after")
    def _timeouts_consistent(self) -> "CodeExecutionConfig":
        """Ensure code execution cannot outlive the sandbox itself."""
        if (
            self.code_execution_timeout_seconds is not None
            and self.code_execution_timeout_seconds > self.sandbox_timeout_seconds
        ):
            raise ValueError("code_execution_timeout_seconds cannot exceed sandbox_timeout_seconds")
        return self


def _build_automatic_function_calling_config(
    config: AgentConfig,
    *,
    tools: list[Any],
    output_schema: type[AgentForecastOutput] | None,
) -> AutomaticFunctionCallingConfig | None:
    """Disable genai AFC when ADK orchestrates tools or schemas."""
    disable = config.disable_automatic_function_calling
    if disable is None:
        disable = bool(tools or output_schema is not None)
    if not disable:
        return None
    return AutomaticFunctionCallingConfig(disable=True)


def _build_search_tool(config: ContextRetrievalConfig) -> Callable[..., Any]:
    """Return an async ``search_web`` FunctionTool backed by Tavily.

    The returned coroutine function is registered as an ADK tool.  It calls
    the Tavily search API and returns a formatted list of results with titles,
    URLs, and content snippets.
    """

    async def search_web(query: str, cutoff_date: str | None = None) -> str:
        """Search the web and return relevant results with source URLs.

        Args:
            query: What to search for.
            cutoff_date: ISO date (YYYY-MM-DD). When provided, appends a
                         ``before:<date>`` constraint to the query.

        Returns
        -------
            Formatted search results with titles, URLs, and content snippets.
        """
        from tavily import AsyncTavilyClient  # noqa: PLC0415

        search_query = (
            f"{query} before:{cutoff_date}"
            if cutoff_date and config.enforce_cutoff
            else query
        )
        client = AsyncTavilyClient(api_key=config.tavily_api_key)
        response = await client.search(search_query, max_results=5)
        results = response.get("results", [])
        if not results:
            return "No results found."
        lines = [
            f"- [{r['title']}]({r['url']}): {r['content']}"
            for r in results
        ]
        return "\n".join(lines)

    return search_web


class AgentConfig(BaseModel):
    """Configuration for building an ADK agent for forecasting tasks.

    Attributes
    ----------
    name : str, default="adk_forecasting_agent"
        Name of the agent.
    model : str | BaseLlm, default=LITE_MODEL (``"gemini-3.1-flash-lite-preview"``)
        Model name (bare, no provider prefix) or a custom
        :class:`~google.adk.models.base_llm.BaseLlm` instance.  When
        ``proxy_base_url`` is set and ``model`` is a plain string,
        :func:`build_adk_agent` wraps it in a
        :class:`~google.adk.models.lite_llm.LiteLlm` instance pointing to
        the proxy.  Pass a ``BaseLlm`` directly to skip automatic wrapping.
    proxy_base_url : str | None, default=PROXY_BASE_URL env var
        Base URL for the OpenAI-compatible LLM proxy.  Defaults to the
        ``PROXY_BASE_URL`` environment variable.  When set, the agent (and
        the ``search_web`` tool) route all calls through the proxy.
    proxy_api_key : str | None, default=PROXY_API_KEY env var
        API key for the proxy.  Defaults to the ``PROXY_API_KEY``
        environment variable.
    description : str, default=""
        Description of the agent. Useful when the agent is used as a sub-agent.
    instruction : str, default=""
        Instruction for the agent.
    skills_dirs : Sequence[Path], default=()
        Sequence of paths to skill directories.
    function_tools : Sequence[Any], default=()
        Conventional ADK tools (e.g. :class:`~google.adk.tools.FunctionTool`
        instances or plain callables) appended directly to the agent's tool
        list. Use this to give the agent a rigid, pre-specified capability such
        as the
        :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
        (in contrast to open-ended code execution). Stored as-is; not validated.
    seed : int or None, default=None
        Generation seed forwarded to the model for reproducibility.
    temperature : float or None, default=None
        Sampling temperature; ``None`` uses the model default.
    max_output_tokens : int or None, default=None
        Maximum tokens per model response; ``None`` uses the model default.
    thinking_budget : int or None, default=None
        Token budget for extended thinking (Gemini thinking models only).
        **Proxy-path caveat:** when routing through the Vector proxy (or any
        OpenAI-compatible proxy), ``thinking_budget`` is passed via ADK's
        ``ThinkingConfig`` → ``GenerateContentConfig``. Whether LiteLLM's
        ``drop_params`` strips it on the proxy path is untested — if you set
        this and see no change in thinking behaviour, treat it as silently
        dropped (same root cause as the ``reasoning_effort`` stripping issue
        documented in ``planning-docs/vector-llm-proxy.md``).
    thinking_level : ThinkingLevel or None, default=None
        Thinking-level preset; overrides ``thinking_budget`` when both are set.
        Subject to the same proxy-path caveat as ``thinking_budget``.
    code_execution : CodeExecutionConfig
        Configuration for E2B code execution. Disabled by default.
    context_retrieval : ContextRetrievalConfig
        Configuration for web-search context retrieval. Disabled by default.
    disable_automatic_function_calling : bool or None, default=None
        When ``True``, sets ``automatic_function_calling.disable`` on the
        Gemini request config.  ADK agents execute tools via the ADK runtime,
        not the genai SDK's Automatic Function Calling (AFC) helper.
        ``None`` (default) auto-disables AFC whenever tools or an
        ``output_schema`` are configured.
    extra_tools : Sequence[Callable[..., Any]], default=()
        Additional callable tools to register with the agent beyond the
        standard code-execution and context-retrieval tools.  Use this to
        inject implementation-specific tools (e.g. adaptive skill mutation
        tools) without coupling the shared factory to implementation code.
        Each callable is appended to the tool list after skills are loaded
        and will be wrapped by ADK as a ``FunctionTool``.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    name: str = "adk_forecasting_agent"
    model: str | BaseLlm = LITE_MODEL
    proxy_base_url: str | None = Field(
        default_factory=lambda: os.getenv("PROXY_BASE_URL"),
        description=(
            "Base URL for the OpenAI-compatible LLM proxy. Defaults to the PROXY_BASE_URL environment variable."
        ),
    )
    proxy_api_key: str | None = Field(
        default_factory=lambda: os.getenv("PROXY_API_KEY"),
        description="API key for the proxy. Defaults to the PROXY_API_KEY environment variable.",
    )
    description: str = ""
    instruction: str = ""
    skills_dirs: Sequence[Path] = ()
    function_tools: Sequence[Any] = ()
    # Optional generation overrides (None = model/provider defaults).
    seed: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    thinking_level: ThinkingLevel | None = None

    # Capabilities
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
    context_retrieval: ContextRetrievalConfig = Field(default_factory=ContextRetrievalConfig)
    disable_automatic_function_calling: bool | None = None
    extra_tools: Sequence[Callable[..., Any]] = ()

    @field_validator("skills_dirs")
    @classmethod
    def _skill_dirs_exist(cls, dirs: Sequence[Path]) -> Sequence[Path]:
        """Reject skill directories that do not resolve to a real directory."""
        missing = [p for p in dirs if not p.is_dir()]
        if missing:
            raise ValueError(f"Skill directories do not exist: {missing}")
        return dirs

    @model_validator(mode="after")
    def _enabled_requires_instruction(self) -> "AgentConfig":
        """Require non-empty instructions for the root and context-retrieval agents."""
        if self.context_retrieval.enabled and not self.context_retrieval.instruction.strip():
            raise ValueError(
                "Expected non-empty instruction for context retrieval agent. "
                "Please provide an instruction in the agent configuration."
            )
        if not self.instruction.strip():
            raise ValueError(
                "Expected non-empty instruction for root agent. "
                "Please provide an instruction in the agent configuration."
            )
        return self


def build_adk_agent(
    config: AgentConfig,
    *,
    output_schema: type[AgentForecastOutput] | None = None,
) -> LlmAgent:
    """Build an ADK agent for forecasting tasks with the given configuration.

    Code execution (E2B) and the web-search context-retrieval tool are wired
    only when the corresponding capability blocks in ``config`` are enabled.

    When ``config.proxy_base_url`` is set and ``config.model`` is a plain
    string, the model is automatically wrapped in a
    :class:`~google.adk.models.lite_llm.LiteLlm` instance that routes all
    calls through the proxy.  Pass a ``BaseLlm`` instance directly to bypass
    automatic wrapping.

    Parameters
    ----------
    config : AgentConfig
        Configuration for the agent.  ``config.instruction`` must be
        non-empty; if ``config.context_retrieval.enabled`` is ``True``,
        ``config.context_retrieval.instruction`` must also be non-empty
        (enforced by :class:`AgentConfig`).
    output_schema : type[AgentForecastOutput] or None, default=None
        When provided, configures the agent to return JSON constrained to
        this schema.  Typically supplied by :class:`AgentPredictor`.

        Note: avoid ``str | None`` optional fields on schemas that also
        contain ``list[BaseModel]`` fields; use string defaults (e.g.
        ``rationale=""``) to stay compatible with ADK's
        ``set_model_response`` tool.

    Returns
    -------
    LlmAgent
        Configured ADK agent with tools and skills attached.

    Examples
    --------
    Interactive analyst — free-form output, no schema constraint:

    >>> from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
    >>> agent = build_adk_agent(AgentConfig(instruction="You are a helpful analyst."))

    Predictor role — structured JSON output constrained to a schema:

    >>> from aieng.forecasting.methods.agentic import (
    ...     AgentConfig,
    ...     ContinuousAgentForecastOutput,
    ...     build_adk_agent,
    ... )
    >>> agent = build_adk_agent(
    ...     AgentConfig(instruction="Forecast the supplied series."),
    ...     output_schema=ContinuousAgentForecastOutput,
    ... )
    """
    # Resolve model: wrap non-native strings in LiteLlm so ADK can route them.
    # ADK's LlmAgent only understands native Gemini model strings or BaseLlm
    # instances — any other string must be wrapped.
    model: str | BaseLlm = config.model
    if isinstance(model, str) and config.proxy_base_url:
        from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415
        from aieng.forecasting.methods.llm_processes._client import _parse_custom_headers  # noqa: PLC0415

        if model.startswith("anthropic/"):
            # Anthropic provider path via a custom gateway (e.g. build-cli).
            # Pass Authorization: Bearer alongside x-api-key so the Roche
            # gateway accepts the request regardless of which auth it checks.
            extra_hdrs: dict[str, str] = {}
            if config.proxy_api_key:
                extra_hdrs["Authorization"] = f"Bearer {config.proxy_api_key}"
            extra_hdrs.update(_parse_custom_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS")))
            model = LiteLlm(
                model=model,
                api_base=config.proxy_base_url,
                api_key=config.proxy_api_key,
                extra_headers=extra_hdrs or None,
            )
        else:
            # OpenAI-compatible proxy path (e.g. Vector proxy).
            # Prefix with "openai/" so LiteLLM routes via the OpenAI-compatible
            # path; LiteLLM strips the prefix before sending to the proxy.
            litellm_model = model if model.startswith("openai/") else f"openai/{model}"
            extra_hdrs = _parse_custom_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS"))
            model = LiteLlm(
                model=litellm_model,
                api_base=config.proxy_base_url,
                api_key=config.proxy_api_key,
                extra_headers=extra_hdrs or None,
            )

    # Configure tools
    tools: list[Any] = []

    if config.code_execution.enabled:
        tools.append(
            CodeInterpreter(
                template_name=config.code_execution.template_name,
                sandbox_timeout_seconds=config.code_execution.sandbox_timeout_seconds,
                code_execution_timeout_seconds=config.code_execution.code_execution_timeout_seconds,
            ).run_code
        )

    if config.context_retrieval.enabled:
        tools.append(_build_search_tool(config.context_retrieval))

    # Load skills
    skills: list[Skill] = []
    for skills_dir in config.skills_dirs:
        skills.append(load_skill_from_dir(skills_dir))

    if skills:
        tools.append(SkillToolset(skills=skills))

    # Append any extra implementation-specific tools (e.g. adaptive skill
    # mutation tools).  These run in the host process, not in E2B.
    for extra in config.extra_tools:
        tools.append(extra)

    # For LiteLlm agents with both output_schema and tools, ADK's
    # can_use_output_schema_with_tools() returns True and skips set_model_response
    # injection, using response_format instead.  However, Gemini thinking models
    # (e.g. gemini-3.5-flash) are trained to call set_model_response when
    # producing structured output alongside other tools — and they do so even when
    # output_schema=None on the Python side.
    #
    # The real SetModelResponseTool fails here because its function declaration
    # uses JSON Schema $defs/$ref (from the Pydantic output schema), which Gemini
    # rejects via the OpenAI-compatible proxy.
    #
    # Fix: register our flat-schema shim (_build_set_model_response_tool) that
    # accepts the JSON as a plain string and parks it in session state.  Clear
    # output_schema so ADK does not also try to enforce it via response_format.
    # AdkTextRunner reads the state key after the run and returns the captured
    # JSON as the final output.
    #
    # This applies to *every* proxy-routed (LiteLlm) agent with an output_schema,
    # not only tool-bearing ones: a schema-only agent with no other tools (e.g. a
    # bare AgentPredictor) would otherwise send the Pydantic schema as Gemini's
    # response_schema and 400 on $defs/$ref/additionalProperties through the proxy.
    # When the shim is the only tool, the model emits the JSON via set_model_response
    # (or as plain text, which AdkTextRunner returns as a fallback) — both paths are
    # handled downstream. Direct-Gemini (non-LiteLlm) agents keep the native schema.
    effective_output_schema = output_schema
    try:
        from google.adk.models.lite_llm import LiteLlm as _LiteLlm  # noqa: PLC0415

        if output_schema is not None and isinstance(model, _LiteLlm):
            tools.append(_build_set_model_response_tool())
            effective_output_schema = None
    except ImportError:
        pass

    # Conventional function tools (e.g. ForecastTool) attach directly.
    tools.extend(config.function_tools)

    thinking_config = (
        ThinkingConfig(
            include_thoughts=True,
            thinking_budget=config.thinking_budget,
            thinking_level=config.thinking_level,
        )
        if config.thinking_budget is not None or config.thinking_level is not None
        else None
    )

    automatic_function_calling = _build_automatic_function_calling_config(
        config,
        tools=tools,
        output_schema=output_schema,
    )

    return LlmAgent(
        name=config.name,
        description=config.description,
        model=model,
        instruction=config.instruction,
        tools=tools,
        output_schema=effective_output_schema,
        generate_content_config=GenerateContentConfig(
            seed=config.seed,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            thinking_config=thinking_config,
            automatic_function_calling=automatic_function_calling,
        ),
    )

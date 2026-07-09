"""Tests for generic ADK agent configuration helpers."""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
    _build_search_tool,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.outputs import ContinuousAgentForecastOutput
from google.adk.models.lite_llm import LiteLlm
from pydantic import ValidationError


class TestCodeExecutionConfig:
    """Cross-field checks tying sandbox lifetime to code execution timeout."""

    def test_raises_when_execution_timeout_exceeds_sandbox(self) -> None:
        """Reject when execution timeout exceeds sandbox lifetime."""
        with pytest.raises(ValidationError, match="code_execution_timeout_seconds"):
            CodeExecutionConfig(
                sandbox_timeout_seconds=2700,
                code_execution_timeout_seconds=2701.0,
            )

    def test_none_execution_timeout_skips_check(self) -> None:
        """Skip the comparison when execution timeout is unset (library default)."""
        config = CodeExecutionConfig(code_execution_timeout_seconds=None)
        assert config.code_execution_timeout_seconds is None


class TestAgentConfig:
    """Validation for reusable agent configs."""

    def test_root_instruction_is_required(self) -> None:
        """A reusable ADK agent needs explicit task instructions."""
        with pytest.raises(ValidationError, match="root agent"):
            AgentConfig()

    def test_context_retrieval_instruction_is_required_when_enabled(self) -> None:
        """Search agents should not be enabled without search instructions."""
        with pytest.raises(ValidationError, match="context retrieval agent"):
            AgentConfig(
                instruction="Forecast the target series.",
                context_retrieval=ContextRetrievalConfig(enabled=True, instruction=" "),
            )

    def test_skill_dirs_must_resolve_to_real_directories(self, tmp_path: Path) -> None:
        """Misspelled skill paths fail loudly at config time."""
        missing = tmp_path / "does_not_exist"

        with pytest.raises(ValidationError, match="Skill directories do not exist"):
            AgentConfig(instruction="Forecast.", skills_dirs=[missing])

    def test_proxy_fields_default_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """proxy_base_url and proxy_api_key pick up environment variables."""
        monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com/v1")
        monkeypatch.setenv("PROXY_API_KEY", "test-key-123")

        config = AgentConfig(instruction="Forecast.")

        assert config.proxy_base_url == "https://proxy.example.com/v1"
        assert config.proxy_api_key == "test-key-123"

    def test_proxy_fields_none_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proxy fields are None when the env vars are unset."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("PROXY_API_KEY", raising=False)

        config = AgentConfig(instruction="Forecast.")

        assert config.proxy_base_url is None
        assert config.proxy_api_key is None


class TestBuildAdkAgent:
    """build_adk_agent wires output_schema, proxy model, and skills correctly."""

    def test_output_schema_with_skills_registers_shim_on_proxy_path(self, tmp_path: Path) -> None:
        """Skills + output_schema + proxy: shim registered, schema cleared at ADK level."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: test\n---\n", encoding="utf-8")
        # Default LITE_MODEL is anthropic/ → Anthropic provider path via proxy.
        agent = build_adk_agent(
            AgentConfig(
                instruction="Forecast the supplied series.",
                skills_dirs=[skill_dir],
                proxy_base_url="https://proxy.example.com/v1",
                proxy_api_key="test-key",
            ),
            output_schema=ContinuousAgentForecastOutput,
        )

        # Proxy set + anthropic model → wrapped in LiteLlm → set_model_response
        # shim registered and output_schema cleared at ADK level.
        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" in tool_names
        assert agent.output_schema is None

    def test_anthropic_model_wrapped_in_litellm_when_proxy_set(self) -> None:
        """An anthropic/ model string is wrapped in LiteLlm when proxy_base_url is set."""
        config = AgentConfig(
            instruction="Forecast.",
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, LiteLlm)
        # anthropic/ prefix preserved — LiteLLM routes via the Anthropic provider.
        assert agent.model.model == "anthropic/claude-haiku-4-5-20251001"

    def test_gemini_model_wrapped_with_openai_prefix_when_proxy_set(self) -> None:
        """A bare Gemini model string is prefixed with openai/ for the proxy path."""
        config = AgentConfig(
            instruction="Forecast.",
            model="gemini-3.1-flash-lite-preview",  # explicit Gemini for OpenAI-proxy path
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, LiteLlm)
        # LiteLlm receives the "openai/" prefix so LiteLLM routes via the
        # OpenAI-compatible proxy path; the prefix is stripped before the
        # proxy sees the model name.
        assert agent.model.model == "openai/gemini-3.1-flash-lite-preview"

    def test_gemini_string_kept_as_string_without_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a proxy URL a native Gemini string is passed as-is to LlmAgent."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("PROXY_API_KEY", raising=False)

        config = AgentConfig(
            instruction="Forecast.",
            model="gemini-3.1-flash-lite-preview",  # explicit Gemini string
            proxy_base_url=None,
            proxy_api_key=None,
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, str)
        assert agent.model == "gemini-3.1-flash-lite-preview"

    def test_any_string_kept_as_string_without_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a proxy URL any model string is passed as-is (no LiteLlm wrapping)."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("PROXY_API_KEY", raising=False)

        config = AgentConfig(
            instruction="Forecast.",
            model="anthropic/claude-haiku-4-5-20251001",
            proxy_base_url=None,
            proxy_api_key=None,
        )
        agent = build_adk_agent(config)

        # Without a proxy the model string is passed through unchanged.
        # The build-cli use case always sets PROXY_BASE_URL so this path
        # is not reached in normal operation.
        assert isinstance(agent.model, str)
        assert agent.model == "anthropic/claude-haiku-4-5-20251001"

    def test_baselm_instance_bypasses_wrapping(self) -> None:
        """A pre-built BaseLlm instance is passed through unchanged."""
        custom_model = LiteLlm(model="gpt-4o-mini")
        config = AgentConfig(
            instruction="Forecast.",
            model=custom_model,
            proxy_base_url="https://proxy.example.com/v1",
        )
        agent = build_adk_agent(config)

        assert agent.model is custom_model

    def test_tools_auto_disable_automatic_function_calling(self) -> None:
        """ADK-orchestrated agents disable genai AFC to avoid mixed-tool warnings."""
        config = AgentConfig(
            instruction="Forecast the supplied series.",
            context_retrieval=ContextRetrievalConfig(
                enabled=True,
                instruction="Search for market news before the cutoff date.",
            ),
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        afc = agent.generate_content_config.automatic_function_calling
        assert afc is not None
        assert afc.disable is True

    def test_instruction_only_agent_leaves_automatic_function_calling_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Minimal interactive agents keep genai AFC at provider defaults."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        agent = build_adk_agent(AgentConfig(instruction="You are a helpful analyst.", proxy_base_url=None))

        assert agent.generate_content_config.automatic_function_calling is None

    def test_function_tools_are_attached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Conventional function tools in the config are appended to the agent."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)

        def my_tool(x: str) -> str:
            """Echo the input. Args: x: anything. Returns: the same string."""
            return x

        agent = build_adk_agent(
            AgentConfig(
                instruction="Forecast the supplied series.",
                function_tools=[my_tool],
                proxy_base_url=None,
            )
        )

        assert len(agent.tools) == 1


class TestSmrShimRegistration:
    """build_adk_agent registers the set_model_response shim on the proxy path.

    Any LiteLlm (proxy-routed) agent with an output_schema gets the flat-string
    shim, because ADK's native response_schema uses $defs/$ref/additionalProperties
    that Gemini rejects via the OpenAI-compatible proxy — regardless of whether the
    agent has other tools. So build_adk_agent swaps the shim in and clears
    output_schema at the ADK level. Direct-Gemini (non-LiteLlm) agents keep the
    native schema, which the Gemini API accepts.
    """

    def test_smr_shim_registered_and_output_schema_cleared_on_litellm_path(self) -> None:
        """Proxy + tools + schema: shim in tools, agent output_schema is None."""
        config = AgentConfig(
            instruction="Forecast the supplied series.",
            context_retrieval=ContextRetrievalConfig(
                enabled=True,
                instruction="Search for market news before the cutoff date.",
            ),
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" in tool_names, f"set_model_response shim not found in tools. Got: {tool_names}"
        assert agent.output_schema is None, (
            "output_schema must be cleared at ADK level when shim is active; "
            "AgentPredictor validates the JSON directly."
        )

    def test_smr_shim_registered_without_tools_on_litellm_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proxy + schema + NO other tools: shim still fires, output_schema cleared.

        Regression guard for the bare-AgentPredictor case (e.g. BoC's basic agent):
        without the shim, ADK would send the Pydantic schema as Gemini's
        response_schema and 400 on $defs/$ref/additionalProperties through the proxy.
        """
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        config = AgentConfig(
            instruction="Forecast.",
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" in tool_names, f"shim not registered without tools. Got: {tool_names}"
        assert agent.output_schema is None

    def test_output_schema_retained_on_direct_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No proxy, explicit Gemini string: native output_schema enforcement is preserved."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        # Explicitly use a bare Gemini model — not wrapped in LiteLlm, so ADK
        # enforces the schema natively and the shim is not needed.
        config = AgentConfig(instruction="Forecast.", model="gemini-3.1-flash-lite-preview")
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        assert agent.output_schema is ContinuousAgentForecastOutput
        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" not in tool_names


class TestBuildSearchTool:
    """_build_search_tool creates a correctly-shaped async Tavily-backed tool."""

    def test_returns_callable_with_expected_signature(self) -> None:
        """Returned function is async and accepts query + optional cutoff_date."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="You are a search assistant.",
            tavily_api_key="test-key",
        )
        tool = _build_search_tool(config)

        assert callable(tool)
        assert inspect.iscoroutinefunction(tool)
        sig = inspect.signature(tool)
        assert "query" in sig.parameters
        assert "cutoff_date" in sig.parameters
        assert sig.parameters["cutoff_date"].default is None

    @pytest.mark.asyncio
    async def test_cutoff_date_appended_to_query_when_enforce_cutoff_true(self) -> None:
        """Cutoff date is appended to the Tavily query when enforce_cutoff=True."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="Search assistant.",
            enforce_cutoff=True,
            tavily_api_key="test-key",
        )
        tool = _build_search_tool(config)

        captured_queries: list[str] = []

        async def _fake_search(query: str, **kwargs: object) -> dict:
            captured_queries.append(query)
            return {"results": [{"title": "T", "url": "https://example.com", "content": "c"}]}

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=_fake_search)

        with patch("tavily.AsyncTavilyClient", return_value=mock_client):
            await tool(query="WTI price", cutoff_date="2024-01-15")

        assert captured_queries, "Tavily search was not called"
        assert "2024-01-15" in captured_queries[0]

    @pytest.mark.asyncio
    async def test_cutoff_not_appended_when_enforce_cutoff_false(self) -> None:
        """No cutoff constraint when enforce_cutoff=False."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="Search assistant.",
            enforce_cutoff=False,
            tavily_api_key="test-key",
        )
        tool = _build_search_tool(config)

        captured_queries: list[str] = []

        async def _fake_search(query: str, **kwargs: object) -> dict:
            captured_queries.append(query)
            return {"results": [{"title": "T", "url": "https://example.com", "content": "c"}]}

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=_fake_search)

        with patch("tavily.AsyncTavilyClient", return_value=mock_client):
            await tool(query="WTI price", cutoff_date="2024-01-15")

        assert "2024-01-15" not in captured_queries[0]

    @pytest.mark.asyncio
    async def test_results_formatted_with_title_url_and_content(self) -> None:
        """Tavily results are formatted as markdown list items with title, URL, content."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="Search assistant.",
            tavily_api_key="test-key",
        )
        tool = _build_search_tool(config)

        async def _fake_search(query: str, **kwargs: object) -> dict:
            return {
                "results": [
                    {"title": "WTI Report", "url": "https://example.com/wti", "content": "Oil at $90."},
                    {"title": "OPEC News", "url": "https://example.com/opec", "content": "OPEC cuts."},
                ]
            }

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=_fake_search)

        with patch("tavily.AsyncTavilyClient", return_value=mock_client):
            result = await tool(query="WTI price")

        assert "https://example.com/wti" in result
        assert "https://example.com/opec" in result
        assert "WTI Report" in result
        assert "OPEC News" in result

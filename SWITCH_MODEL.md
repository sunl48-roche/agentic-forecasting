# Switching from Gemini (Vector proxy) to Claude (build-cli)

This document describes every change made to the codebase to support the Roche
build-cli AI Gateway, and what manual setup is still required before running
any notebooks or scripts.

---

## Status

| # | Change | State |
|---|---|---|
| 1 | `models.py` — Claude model constants | **Done** |
| 2 | `_client.py` — Anthropic routing + custom headers | **Done** |
| 3 | `agent_factory.py` — Anthropic model wrapping + Tavily search | **Done** |
| 4 | `pyproject.toml` — `tavily-python` dependency | **Done** |
| 5 | `00_environment_check.ipynb` — auto token refresh in setup cell | **Done** |
| 6 | `.env` — gateway URL, custom header, Tavily key | **Manual — you must do this** |

---

## Background

The repo previously routed all LLM calls through the Vector Institute proxy
(`proxy.vectorinstitute.ai`), which exposes Gemini models via an
OpenAI-compatible endpoint. The Roche build-cli gateway
(`us.build-cli.roche.com/proxy`) exposes Claude models via the Anthropic API
protocol — a different wire format, auth scheme, and model naming convention.

### Key differences

| Property | Vector proxy (old) | build-cli gateway (new) |
|---|---|---|
| Endpoint | `https://proxy.vectorinstitute.ai/v1` | `https://us.build-cli.roche.com/proxy` |
| Protocol | OpenAI-compatible (`/chat/completions`) | Anthropic (`/v1/messages`) |
| Auth header | `Authorization: Bearer <key>` | `x-api-key: <token>` (SDK) + `Authorization: Bearer <token>` (gateway) |
| Required custom header | none | `x-build-cli-tool: claude` |
| Token lifetime | long-lived | short-lived ID token — refreshed automatically in notebooks |
| LiteLLM model prefix | bare string (e.g. `gemini-3.5-flash`) | `anthropic/` prefix (e.g. `anthropic/claude-haiku-4-5-20251001`) |
| Web search | Gemini `googleSearch` server-side extension | Tavily REST API |

The `anthropic/` prefix tells LiteLLM to use the Anthropic provider path
(`/v1/messages`), which is what the Roche gateway speaks. The OpenAI-compatible
path (`/chat/completions`) exists on the gateway but is blocked by the WAF for
OpenAI SDK requests due to `X-Stainless-*` headers added by the SDK.

---

## What was changed in the code

### `aieng-forecasting/aieng/forecasting/models.py`

Model constants updated from Gemini to Claude:

```python
# Before
LITE_MODEL     = "gemini-3.1-flash-lite-preview"
ADVANCED_MODEL = "gemini-3.5-flash"

# After
LITE_MODEL     = "anthropic/claude-haiku-4-5-20251001"
ADVANCED_MODEL = "anthropic/claude-sonnet-4-6[1m]"
```

Every predictor and agent factory imports from this module — a single-file
change that propagates everywhere automatically.

### `aieng-forecasting/aieng/forecasting/methods/llm_processes/_client.py`

Two additions:

1. **`_parse_custom_headers(raw)`** — new helper that parses the
   `ANTHROPIC_CUSTOM_HEADERS` env var (`"key: value"` or `"k1: v1, k2: v2"`)
   into a `dict`. LiteLLM does not read this env var on its own.

2. **Anthropic-aware routing in `_one_completion_async`** — when `api_base` is
   set and the model starts with `anthropic/`, the function:
   - Keeps the `anthropic/` prefix so LiteLLM routes via the Anthropic provider
   - Adds `Authorization: Bearer {api_key}` to `extra_headers` (the Roche
     gateway requires Bearer; the Anthropic SDK sends `x-api-key`, so both are
     sent and the gateway accepts whichever it checks)
   - Merges any `ANTHROPIC_CUSTOM_HEADERS` into `extra_headers`

   Non-`anthropic/` models continue to use the existing OpenAI-compatible proxy
   path (model string prefixed with `openai/`).

### `aieng-forecasting/aieng/forecasting/methods/agentic/agent_factory.py`

Three changes:

1. **`ContextRetrievalConfig`** — `search_model` field (Gemini-only) replaced
   by `tavily_api_key` (reads `TAVILY_API_KEY` env var). The unused
   `temperature` and `max_output_tokens` fields (inner Gemini call only) are
   also removed.

2. **Model wrapping in `build_adk_agent`** — added `anthropic/` branch.
   ADK's `LlmAgent` only accepts native Gemini strings or `BaseLlm` instances,
   so all proxy-routed models are wrapped in `LiteLlm`. The new branch handles
   `anthropic/` models with the correct `api_base`, `api_key`, and
   `extra_headers` (Bearer + custom header). Non-`anthropic/` models continue
   through the existing `openai/` prefix path.

3. **`_build_search_tool`** — replaced Google Search (Gemini-only server-side
   `{"googleSearch": {}}` extension) with a Tavily `AsyncTavilyClient` call.
   The `search_web` tool signature is unchanged (`query`, optional `cutoff_date`).
   The call site in `build_adk_agent` no longer receives `proxy_base_url` /
   `proxy_api_key`.

### `aieng-forecasting/pyproject.toml`

`tavily-python>=0.5` added to the `agentic` optional dependency group.
Run `uv sync` after pulling this branch. `tavily-python==0.7.26` is already
in `uv.lock`.

### `aieng-forecasting/tests/`

Tests in `test_agent_factory.py` and `test_sampled_trajectory.py` updated to
reflect new model names, routing behaviour, and Tavily-based search tool.
**375 tests pass.**

### `implementations/getting_started/00_environment_check.ipynb`

The setup cell now auto-refreshes the build-cli token at the start of every
notebook run, solving two VS Code-specific problems:

- **PATH issue** — VS Code launched from the GUI (Dock/Spotlight) doesn't
  inherit `~/.local/bin` where `build-cli` lives. The cell probes
  `~/.local/bin/build-cli` directly before falling back to PATH lookup.
- **Cell order issue** — the token is fetched in the setup cell, before any
  check cell runs, so execution order doesn't matter.

The token refresh only fires when `PROXY_BASE_URL` contains
`build-cli.roche.com`, so the cell is a no-op for any other proxy setup.

The LLM inference check cell (check 3) is also updated to branch on
`LITE_MODEL`'s prefix: `anthropic/` models use the Anthropic provider path
with `Authorization: Bearer` + custom headers; other models continue to use
the old `openai/{model}` prefix path.

---

## Manual setup — do this once before running anything

### Step 1 — Prerequisites

```bash
build-cli login              # SSO login (opens browser)
build-cli claude setup       # writes gateway URL to ~/.claude/settings.json
build-cli doctor             # verify setup is healthy
```

A free Tavily API key is needed for web-search-enabled agents:

> https://tavily.com → sign up → copy your API key

### Step 2 — Update `.env`

Open the repo-root `.env` file and apply these changes:

```diff
-PROXY_BASE_URL=https://proxy.vectorinstitute.ai/v1
-PROXY_API_KEY=your_vector_proxy_key
+PROXY_BASE_URL=https://us.build-cli.roche.com/proxy
+PROXY_API_KEY=
+ANTHROPIC_CUSTOM_HEADERS=x-build-cli-tool: claude
+TAVILY_API_KEY=your_tavily_api_key
```

Leave `PROXY_API_KEY` blank — notebooks refresh it automatically. Keep all
other existing keys (`FRED_API_KEY`, `LANGFUSE_*`, `E2B_API_KEY`, etc.)
unchanged.

Or from the repo root in a terminal:

```bash
sed -i '' \
  -e 's|^PROXY_BASE_URL=.*|PROXY_BASE_URL=https://us.build-cli.roche.com/proxy|' \
  -e 's|^PROXY_API_KEY=.*|PROXY_API_KEY=|' \
  .env
grep -q "ANTHROPIC_CUSTOM_HEADERS" .env || \
  echo "ANTHROPIC_CUSTOM_HEADERS=x-build-cli-tool: claude" >> .env
```

### Step 3 — Running notebooks (VS Code or Jupyter Lab)

The setup cell in every notebook that calls the LLM will:

1. Read `PROXY_BASE_URL` from `.env`
2. Detect the build-cli gateway
3. Call `build-cli auth token` (using the full path to `~/.local/bin/build-cli`)
4. Set `PROXY_API_KEY` in the kernel's environment

**No manual `export` is needed before launching VS Code.**

If the setup cell prints `⚠️  build-cli auth token failed`, run
`build-cli login` in a terminal — your session has expired.

> **VS Code note:** if you see old fix messages (referencing
> `proxy.vectorinstitute.ai`) after pulling this branch, VS Code has a cached
> version of the notebook. Do `Cmd+Shift+P` → `Revert File`, then
> `Jupyter: Restart Kernel`.

### Step 4 — Running scripts / CLI (non-notebook)

For scripts, set the token in the shell before running:

```bash
export PROXY_API_KEY=$(build-cli auth token)
uv run python your_script.py
```

Tokens last a few hours of active use. After 5 days of inactivity,
`build-cli login` is required to get a fresh refresh token.

### Step 5 — Verify

Run the quick smoke test from the repo root:

```bash
export PROXY_API_KEY=$(build-cli auth token)

uv run python - <<'EOF'
import asyncio, os
from aieng.forecasting.models import LITE_MODEL
from aieng.forecasting.methods.llm_processes._client import _one_completion_async, bootstrap_litellm

bootstrap_litellm()

async def smoke():
    content, _, in_tok, out_tok = await _one_completion_async(
        model=LITE_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        response_format={"type": "text"},
        temperature=0.0,
        max_tokens=10,
        timeout_s=20.0,
        reasoning_effort=None,
        api_base=os.getenv("PROXY_BASE_URL"),
        api_key=os.getenv("PROXY_API_KEY"),
    )
    print(f"Model    : {LITE_MODEL}")
    print(f"Response : {content!r}")
    print(f"Tokens   : {in_tok} in / {out_tok} out")
    assert content and "OK" in content.upper()
    print("✓ passed")

asyncio.run(smoke())
EOF
```

Expected output:

```
Model    : anthropic/claude-haiku-4-5-20251001
Response : 'OK'
Tokens   : 14 in / 4 out
✓ passed
```

Then open and run top-to-bottom:

```
implementations/getting_started/00_environment_check.ipynb
```

---

## What does NOT need to change

| Component | Why |
|---|---|
| `LLMPredictorConfig` in `base.py` | `proxy_base_url` / `proxy_api_key` read `PROXY_BASE_URL` / `PROXY_API_KEY` — same env var names, new values |
| All `analyst_agent/`, `starter_agent/`, `adaptive_agent/` `agent.py` files | Reference `LITE_MODEL` / `ADVANCED_MODEL` constants — follow `models.py` automatically |
| `set_model_response` shim in `agent_factory.py` | Model-agnostic — fires for any `LiteLlm` agent with `output_schema`; Claude follows the instruction |
| `reasoning_effort=None` default | Correct for both Gemini and Claude |
| PDF report ingestion (food CPI) | Improves — Claude supports native PDF; Gemini was the blocked path |

---

## Features lost vs gained

| Feature | After switch |
|---|---|
| Gemini `googleSearch` grounding | Replaced by Tavily — different coverage, no cutoff metadata |
| `thinking_budget` / `thinking_level` on `AgentConfig` | Silently ignored for `LiteLlm` agents — do not set |
| PDF report ingestion in LLMP | Now works — Claude is the supported path |
| Backtest result cache (`data/predictions/`) | Predictor IDs embed model name — delete old Gemini cache files before re-running backtests |

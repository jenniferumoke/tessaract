# Architecture

Tessaract has three layers:

```
 your agent code
      │  canonical types (UserMessage, FunctionTool, ReasoningOptions, …)
      ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Tessaract   │ ──▶ │     Adapter      │ ──▶ │     Provider     │ ──▶ vendor SDK / API
│  (client.py) │     │ (translation)    │     │ (SDK client)     │
└──────────────┘     └──────────────────┘     └──────────────────┘
      ▲                       │
      │   Response / StreamEventUnion (canonical, with .raw)
      └───────────────────────┘
```

## Providers

A **provider** (`providers/`) holds credentials and the vendor SDK client. `OpenAIProvider` builds `openai.OpenAI(...)` from its typed fields plus `provider_args`, falls back to `OPENAI_API_KEY`, and imports the SDK lazily. `AnthropicProvider` does the same with `anthropic.Anthropic(...)` and `ANTHROPIC_API_KEY`, and `OpenRouterProvider` with `openrouter.OpenRouter(...)` and `OPENROUTER_API_KEY`, so Tessaract depends only on `pydantic` unless you install an extra.

## Adapters

An **adapter** (`adapters/`) translates in both directions between canonical types and one vendor's API. The `Adapter` base class defines the translation hooks:

| Method | Direction | Purpose |
| --- | --- | --- |
| `map_input_message(UserMessage \| SystemPrompt)` | canonical → native | User turns and system prompts |
| `map_assistant_message(AssistantMessage)` | canonical → native | Earlier assistant text |
| `map_output_item(OutputItem)` | canonical → native | Replays earlier reasoning, function call and provider items. The default returns `item.raw` |
| `map_tool_result(FunctionToolResult)` | canonical → native | Tool outputs |
| `map_function_schema(list[FunctionTool])` | canonical → native | Tool definitions |
| `map_reasoning_params(ReasoningOptions)` | canonical → native | Reasoning configuration |
| `map_reasoning(...)` | canonical → native | Reserved |

`OpenAIAdapter`, `AnthropicAdapter` and `OpenRouterAdapter` also implement:

- `_normalize_output_item` / `_normalize_output`, which turn native output items into canonical ones
- `_normalize_stream_event`, which turns native stream events into canonical ones
- `_build_request_kwargs`, which merges the canonical parameters with `provider_options` (canonical wins)
- `generate_sync(request) -> Response`
- `generate_stream(request) -> Iterator[StreamEventUnion]`

`AnthropicAdapter._build_request_kwargs` also reshapes the flat list of mapped items into what the Messages API expects. System items move to the top-level `system` parameter, after `request.instructions`. Neighbouring items with the same role are merged into one turn, so a response's thinking, text and `tool_use` blocks become one assistant turn, and the `tool_result`s that follow become one user turn with the results first. When `max_tokens` isn't set it sends `DEFAULT_ANTHROPIC_MAX_TOKENS` (16,000). Requests that use beta-only thinking features go to `client.beta.messages` with the beta header added.

`OpenRouterAdapter` uses OpenRouter's Responses API through the OpenRouter SDK, independently of `OpenAIAdapter`. OpenRouter stores nothing between requests, so it always asks for `reasoning.encrypted_content` and the history carries the reasoning. The SDK only accepts plain dicts as input, so output items are dumped with `model_dump(by_alias=True)` before they are replayed.

## The request lifecycle

1. `Tessaract.send()` splits `"oai/gpt-…"` into the prefix and the model name, and looks up the provider and adapter.
2. `_build_request_model` converts each `input` item:
   - `str` → `UserMessage` → `adapter.map_input_message`
   - `InputType` → `item.raw(adapter)`, which calls the matching `map_*` hook
   - `AssistantMessage` → `adapter.map_assistant_message`
   - other output items → `adapter.map_output_item`, which replays the stored `.raw` native object
3. A canonical `Request` goes to `adapter.generate_sync`, or to `adapter.generate_stream` when `stream=True`.
4. The adapter calls the SDK, then normalizes the result into a `Response` or a stream of events.

## Design principles

- **Canonical first, native always available.** Every canonical object keeps the native payload (`raw`, `raw_response`, `raw_event`), so nothing is lost and you can always go down to the SDK.
- **Lossless history.** Output items replay from `.raw`, so provider-specific state such as reasoning IDs and encrypted content survives across turns.
- **Unknowns pass through.** Output items that Tessaract can't model become `ProviderOutputItem`, and stream events it can't model become `CustomProviderEvent`. Neither is dropped.
- **Canonical parameters win.** `request_options` and `provider_options` can add native parameters, but they can't override the canonical ones.

## Adding a provider

1. Subclass `Provider` and create the SDK client in `__post_init__` from `self._client_kwargs()`. Import the SDK lazily and raise a helpful `ImportError`.
2. Subclass `Adapter` and implement the `map_*` hooks, output normalization, stream normalization, `generate_sync` and `generate_stream`.
3. Register the pair in `Tessaract.register_adapter()`.
4. Add an optional dependency group to `pyproject.toml`.
5. Export the provider from `tessaract/__init__.py`.

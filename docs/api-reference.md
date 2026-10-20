# API reference

Everything listed as exported can be imported from the top-level package:

```python
from tessaract import (
    Tessaract,
    OpenAIProvider, AnthropicProvider,
    FunctionTool, InputSchema, Property,
    UserMessage, SystemPrompt, FunctionToolResult,
    ReasoningOptions, CacheOptions,
    DEFAULT_ANTHROPIC_MAX_TOKENS,
)
```

Other types can be imported from their modules, shown under each heading.

---

## Client

### `Tessaract`

`tessaract.client.Tessaract`

```python
Tessaract(providers: dict[str, OpenAIProvider | AnthropicProvider])
```

This creates a client and builds one adapter per registered provider. Any provider type without an adapter raises `NotImplementedError`.

#### `send()`

```python
send(
    model: str,
    input: str | list[str | InputType | OutputItem | AssistantMessage],
    stream: bool = False,
    reasoning: ReasoningOptions | None = None,
    tools: list[FunctionTool] | None = None,
    request_options: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    instructions: str | None = None,
    cache: CacheOptions | None = None,
) -> Response | Iterator[StreamEventUnion]
```

| Parameter | Description |
| --- | --- |
| `model` | `"<prefix>/<model-name>"`. The prefix selects the provider, and the model name is **everything** after the first `/`, so `"oai/org/model"` sends `"org/model"`. |
| `input` | A prompt string, or a history list. Strings become `UserMessage`s, and output items are replayed from `.raw`. |
| `stream` | If `False`, returns a `Response`. If `True`, returns `Iterator[StreamEventUnion]`. |
| `reasoning` | See [`ReasoningOptions`](#reasoningoptions). |
| `tools` | See [`FunctionTool`](#functiontool). |
| `request_options` | Native request parameters merged into the provider call. Canonical parameters take precedence. |
| `max_tokens` | Maximum output tokens. OpenAI: `max_output_tokens`, left out when `None`. Anthropic: `max_tokens`, defaulting to `DEFAULT_ANTHROPIC_MAX_TOKENS` (16,000) when `None`; setting it explicitly is recommended. |
| `instructions` | System instructions. OpenAI: the native `instructions` parameter. Anthropic: the first `system` block. |
| `cache` | See [`CacheOptions`](#cacheoptions). |

**Raises:**

- `ValueError` if `model` isn't `"<prefix>/<model-name>"`, or if the prefix isn't registered
- `TypeError` if an item in `input` has an unsupported type
- `ValueError` if a canonical value can't be used by the provider, such as `mode="pro"` on Anthropic
- any exception raised by the provider SDK, such as `openai.APIError` or `anthropic.APIError`

---

## Providers

### `Provider`

`tessaract.providers.Provider`

This is the base dataclass for providers.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `api_key` | `str \| None` | `None` | Falls back to the provider's environment variable |
| `base_url` | `str \| None` | `None` | API base URL |
| `timeout` | `float \| None` | `None` | Request timeout in seconds |
| `max_retries` | `int \| None` | `None` | SDK retry count |
| `default_headers` | `dict[str, str] \| None` | `None` | Headers sent with every request |
| `default_query` | `dict[str, object] \| None` | `None` | Query parameters sent with every request |
| `provider_args` | `dict` | `{}` | Extra keyword arguments for the SDK client constructor. These override the fields above. |

Fields left as `None` aren't passed to the SDK, so the SDK's own defaults apply.

### `OpenAIProvider`

`tessaract.OpenAIProvider`

```python
OpenAIProvider(api_key="sk-...", base_url=..., timeout=..., provider_args={"organization": ...})
```

This creates an `openai.OpenAI` client from the fields that are set, plus `provider_args`. The `.client` property returns that SDK client. If `api_key` is `None`, it reads `OPENAI_API_KEY`.

**Raises:** `ImportError` if the `openai` package isn't installed, and `ValueError` if no API key is found.

### `AnthropicProvider`

`tessaract.AnthropicProvider`

```python
AnthropicProvider(api_key="sk-ant-...", base_url=..., timeout=..., provider_args={...})
```

This creates an `anthropic.Anthropic` client from the fields that are set, plus `provider_args`. The `.client` property returns that SDK client. If `api_key` is `None`, it reads `ANTHROPIC_API_KEY`.

### `DEFAULT_ANTHROPIC_MAX_TOKENS`

`tessaract.DEFAULT_ANTHROPIC_MAX_TOKENS` (`16_000`). The `max_tokens` the Anthropic adapter sends when `send()` gets `max_tokens=None`. A thinking `budget` must be less than the `max_tokens` that's sent.

**Raises:** `ImportError` if the `anthropic` package isn't installed, and `ValueError` if no API key is found.

---

## Requests

### `ReasoningOptions`

`tessaract.ReasoningOptions` (a dataclass)

| Field | Type | Default |
| --- | --- | --- |
| `effort` | `"none" \| "minimal" \| "low" \| "medium" \| "high" \| "extra_high" \| "max" \| None` | `None` |
| `summary` | `"concise" \| "auto" \| "detailed" \| "omitted" \| "updates" \| None` | `None` |
| `mode` | `"standard" \| "pro" \| "enabled" \| "adaptive" \| "between_tools" \| "disabled" \| None` | `None` |
| `budget` | `int \| None` | `None` |
| `native_options` | `dict \| None` | `None` |

See [Reasoning](reasoning.md).

### `CacheOptions`

`tessaract.CacheOptions` (a dataclass)

| Field | Type | Default | OpenAI | Anthropic |
| --- | --- | --- | --- | --- |
| `mode` | `"implicit" \| "explicit" \| None` | `None` | `prompt_cache_options.mode`; `gpt-5.6`+ only, older models only cache automatically | `"explicit"` leaves out the top-level `cache_control` |
| `retention` | `"short" \| "long" \| None` | `None` | Ignored | `ttl`: `"5m"` / `"1h"`, on the automatic breakpoint and on breakpoints without their own `ttl` |
| `key` | `str \| None` | `None` | `prompt_cache_key` | Ignored |

On Anthropic, passing any `CacheOptions` except `mode="explicit"` sends a top-level `cache_control`, which turns on automatic caching. OpenAI caches automatically, so only `mode` and `key` are sent there, and `mode` only works on `gpt-5.6` and later; OpenAI's lifetime fields differ by model, so set them in `request_options` if you need them. Fields left as `None` are not sent. The typed values take precedence over the same keys in `request_options`. See [Prompt caching](getting-started.md#prompt-caching).

### `Request`

`tessaract.types.request.Request` (a dataclass)

This is the internal canonical request that `send()` builds and passes to an adapter. You only need it if you're writing an adapter.

| Field | Type |
| --- | --- |
| `model` | `str` (without the prefix) |
| `input` | `list`, already converted to native items |
| `instructions` | `str \| None`, from `send(instructions=...)` |
| `tools` | `list[FunctionTool] \| None` |
| `reasoning` | `ReasoningOptions \| None` |
| `cache` | `CacheOptions \| None` |
| `stream` | `bool` |
| `provider_options` | `dict \| None` |
| `max_tokens` | `int \| None` |

---

## Tools

### `FunctionTool`

`tessaract.FunctionTool` (a Pydantic model)

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | required |
| `description` | `str` | required |
| `input_schema` | `InputSchema \| None` | `None` |
| `strict` | `bool \| None` | `None` (both adapters treat it as `True`) |
| `cache` | `bool` | `False`. Anthropic: `cache_control` on the tool definition. Ignored on OpenAI. |
| `provider_options` | `dict[str, Any]` | `{}` |

### `InputSchema`

`tessaract.InputSchema` (a Pydantic model with strict validation)

| Field | Type | Default |
| --- | --- | --- |
| `type` | `"object"` | `"object"` |
| `properties` | `dict[str, Property]` | `{}` |
| `required` | `list[str]` | `[]` |
| `additionalProperties` | `bool \| None` | `None`, and left out of serialization when `None` |

Validation raises `ValueError` if `required` names a property that isn't in `properties`.

### `Property`

`tessaract.Property` (a Pydantic model)

| Field | Type |
| --- | --- |
| `type` | `"string" \| "number" \| "integer" \| "boolean" \| "array" \| "object" \| "null"`, or a list of these, e.g. `["string", "null"]` |
| `description` | `str \| None` |
| `enum` | `list \| None` |
| `items` | `Property \| None` (for `"array"` only) |
| `properties` | `dict[str, Property] \| None` (for `"object"` only) |
| `required` | `list[str] \| None` (for `"object"` only) |
| `additionalProperties` | `bool \| None` (for `"object"` only; the OpenAI adapter sends `False` when it's `None`) |

Validation raises `ValueError` if `items` is set on a non-array type, if the object fields are set on a non-object type, or if `required` names an undefined property.

### `WebSearchTool`

`tessaract.WebSearchTool` (a Pydantic model). A web search the provider runs. See [Server tools](tool-calling.md#server-tools).

| Field | Type | Default |
| --- | --- | --- |
| `max_uses` | `int \| None` (at least 1) | `None`. Anthropic only |
| `allowed_domains` | `list[str] \| None` | `None` |
| `blocked_domains` | `list[str] \| None` | `None`. Anthropic only; cannot be combined with `allowed_domains` |
| `user_location` | `UserLocation \| None` | `None` |
| `cache` | `bool` | `False`. Anthropic: `cache_control` on the tool definition. Ignored on OpenAI |
| `provider_options` | `dict[str, Any]` | `{}`. Native tool fields; `"type"` pins the native tool version |

### `UserLocation`

`tessaract.UserLocation` (a Pydantic model): `city`, `region`, `country` (a two-letter ISO code), `timezone` (an IANA timezone), all `str | None`. Both adapters send it as an `approximate` location.

### `ProviderTool`

`tessaract.ProviderTool` (a Pydantic model). A native tool definition, sent unchanged.

| Field | Type |
| --- | --- |
| `provider` | `"openai" \| "anthropic"`: the provider the definition was written for. Sending it to another provider raises `ValueError` |
| `definition` | `dict[str, Any]` |

### `Tool`

`tessaract.Tool` is `FunctionTool | WebSearchTool | ProviderTool`, the type `send(tools=...)` accepts.

### `UnsupportedToolOptionError`

`tessaract.UnsupportedToolOptionError`, a `ValueError` raised when a canonical tool option has no equivalent on the target provider, such as `WebSearchTool(max_uses=...)` on OpenAI.

See [Tool calling](tool-calling.md).

---

## Input items

`tessaract.types.input_types`. Every input item subclasses `InputType` and implements `raw(adapter)`, which returns the provider-native form.

### `UserMessage`

| Field | Type | Default |
| --- | --- | --- |
| `role` | `"user"` | `"user"` |
| `content` | `str \| list[dict]` | required |
| `cache` | `bool` | `False`. Puts a cache breakpoint on the last content part. On OpenAI, needs `gpt-5.6` or later. |

`content` can be a list of native content parts, such as OpenAI `input_text` / `input_image` dicts, for multimodal input.

### `SystemPrompt`

| Field | Type | Default |
| --- | --- | --- |
| `role` | `"system"` | `"system"` |
| `content` | `str \| list[dict]` | required |
| `cache` | `bool` | `False`. Puts a cache breakpoint on the last content part. On OpenAI, needs `gpt-5.6` or later. |

Use `SystemPrompt(content="...")` in an input list to provide system instructions. It is mapped to an OpenAI input message with the `system` role, and moved to Anthropic's top-level `system` parameter.

### `FunctionToolResult`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `"function_tool_result"` | `"function_tool_result"` |
| `call_id` | `str` | required |
| `result` | `Any` | required. Strings are sent as-is, and other values are JSON-encoded. |
| `is_error` | `bool` | `False`. When `True`, OpenAI receives `{"error": result}`, and Anthropic receives `"is_error": true`. |
| `cache` | `bool` | `False`. Puts a cache breakpoint on the result. On OpenAI, needs `gpt-5.6` or later. |
| `provider_options` | `dict[str, Any]` | `{}`. Extra native fields merged into the result block, such as Anthropic `cache_control`. |

---

## Responses

### `Response`

`tessaract.types.response.Response` (a dataclass). The OpenAI adapter returns the subclass `OpenAIResponse`, and the Anthropic adapter returns `AnthropicResponse`.

| Field / property | Type | Description |
| --- | --- | --- |
| `id` | `str` | Provider response ID |
| `model` | `str` | Model name |
| `provider` | `"openai" \| "anthropic" \| None` | Which provider produced the response |
| `status` | `ResponseStatus \| None` | Canonical status: `"completed"` when the model finished its turn (including when it stops to call tools), `"incomplete"` when the output stopped early. `None` when the provider reports nothing Tessaract can map |
| `incomplete_reason` | `IncompleteReason \| None` | Why the response is incomplete. `None` unless `status` is `"incomplete"`, and also `None` when the provider gives no reason |
| `output` | `list[OutputType]` | Typed output items, in order |
| `usage` | `Usage \| None` | Normalized token counts when the provider reports usage |
| `error` | `ResponseError \| None` | Set if the provider reports an error |
| `raw_response` | `Any` | The native SDK response object |
| `output_text` *(property)* | `str` | All assistant message text joined together |
| `response_id` *(property)* | `str` | An alias for `id` |

`Usage` includes `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `reasoning_tokens`, and `server_tool_requests` (server tool calls by tool name, e.g. `{"web_search": 2}`, or `None` when no server tools ran). The detail fields are `None` when the provider does not report them. `OpenAIResponse.provider_usage` holds the native OpenAI SDK usage object, or `None` when usage is unavailable. Completed streaming responses expose both fields through `ResponseCompletedEvent.response`.

`OpenAIResponse` takes `status` from OpenAI's response status, and `incomplete_reason` from `incomplete_details.reason`. `OpenAIResponse.input_tokens` and `OpenAIResponse.cached_input_tokens` provide direct access to the corresponding normalized usage counts. Each returns `None` when usage is unavailable; `cached_input_tokens` is also `None` when the provider does not report that detail.

`AnthropicResponse` adds `stop_reason` (a `StopReason`, or the raw string for a stop reason Tessaract doesn't know yet). Anthropic has no response status, so `status` and `incomplete_reason` are derived from `stop_reason`:

| `stop_reason` | `status` | `incomplete_reason` |
| --- | --- | --- |
| `end_turn`, `stop_sequence`, `tool_use` | `completed` | `None` |
| `max_tokens` | `incomplete` | `max_tokens` |
| `model_context_window_exceeded` | `incomplete` | `context_window` |
| `refusal` | `incomplete` | `content_filter` |
| `pause_turn`, `compaction` | `incomplete` | `paused` |
| unknown or `None` | `None` | `None` |

`stop_reason` stays available for detail that `status` doesn't carry, such as whether the model stopped to call a tool. `AnthropicResponse` also has `provider_usage` holds the native Anthropic `Usage`. Its normalized `input_tokens` includes cache reads and cache writes, so it means the same as OpenAI's; `provider_usage.input_tokens` is Anthropic's uncached count. `cached_input_tokens` is Anthropic's `cache_read_input_tokens`, `cache_write_input_tokens` is `cache_creation_input_tokens`, and `reasoning_tokens` is the thinking token count when Anthropic reports it. `AnthropicResponse` has the same `input_tokens` and `cached_input_tokens` properties as `OpenAIResponse`.

`status` and `incomplete_reason` are on every `Response`. `send()` is typed to return `Response`, so use `isinstance(response, AnthropicResponse)` (or `OpenAIResponse`) before reading provider-specific fields such as `stop_reason`. Both classes are in `tessaract.types.response`.

### `StopReason`

A `str` enum with the values `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `compaction`, `refusal` and `model_context_window_exceeded`.

### `ResponseStatus`

A `str` enum with the values `queued`, `in_progress`, `completed`, `incomplete`, `failed` and `cancelled`.

### `IncompleteReason`

A `str` enum, in `tessaract.types.response`:

| Value | Meaning | OpenAI | Anthropic |
| --- | --- | --- | --- |
| `max_tokens` | Output hit the `max_tokens` limit | `max_output_tokens` | `max_tokens` |
| `context_window` | The conversation filled the model's context window | — | `model_context_window_exceeded` |
| `content_filter` | Output was stopped by a safety filter or refusal | `content_filter` | `refusal` |
| `paused` | The turn isn't finished; send the output back to let the model continue | — | `pause_turn`, `compaction` |

### `ResponseError`

| Field | Type |
| --- | --- |
| `message` | `str` |
| `code` | `str \| None` |

---

## Output items

`tessaract.types.output_types`. Every item has a `type` discriminator and a `raw` field that holds the native provider object, which is used to replay the item in the history.

| Class | `type` | Fields |
| --- | --- | --- |
| `AssistantMessage` | `"assistant_message"` | `role="assistant"`, `content: list[TextOutputItem]`, `raw` |
| `TextOutputItem` | `"text"` | `text: str`, `annotations: list[Annotation]`, `raw` |
| `ReasoningOutputItem` | `"reasoning"` | `id`, `text` (the summary), `content` (the raw reasoning), `raw` |
| `FunctionCallOutputItem` | `"function_call"` | `call_id`, `name`, `arguments: dict`, `raw` |
| `WebSearchCallOutputItem` | `"web_search_call"` | `call_id`, `status` (`"in_progress"`, `"searching"`, `"completed"` or `"failed"`), `action` (`"search"`, `"open_page"` or `"find_in_page"`), `queries: list[str]`, `url`, `pattern`, `sources: list[WebSource]` (`url`, `title`, `page_age`), `error`, `provider`, `raw` |
| `ProviderOutputItem` | `"provider_output"` | `provider_type` (the native item type), `raw` |

The OpenAI adapter returns an assistant message as a `ProviderOutputItem` (`provider_type="message"`) when it contains anything other than plain `output_text`, such as refusals or annotations and citations. It still replays correctly, and you can inspect it through `.raw`.

The Anthropic adapter returns neighbouring text blocks as one `AssistantMessage` whose `raw` is the list of native text blocks. Thinking blocks become `ReasoningOutputItem`s with `id=None`. A web search's `server_tool_use` and `web_search_tool_result` blocks become one `WebSearchCallOutputItem` whose `raw` is the list of both blocks. `redacted_thinking`, other server tool blocks and any other block types become `ProviderOutputItem`s. Unknown OpenAI output item types are also returned as `ProviderOutputItem`s.

`TextOutputItem.annotations` holds citations as `Annotation` objects (`provider`, `provider_type`, `source`, `title`, `cited_text`, `provider_metadata`). On Anthropic they come from a text block's `citations`, and `source` is the URL, source or file ID.

`OutputType` is the union of all of these.

---

## Stream events

`tessaract.types.streaming.event_types`. `StreamEventUnion` is the union of all event classes. Every event has a `raw_event` field.

| Class | `type` |
| --- | --- |
| `ResponseStartedEvent` | `response_started` |
| `ResponseCompletedEvent` | `response.completed` |
| `ResponseFailedEvent` | `response.failed` |
| `TextDeltaEvent` | `text.delta` |
| `ReasoningStartedEvent` | `reasoning.started` |
| `ReasoningSummaryDeltaEvent` | `reasoning_summary.delta` |
| `ReasoningTextDeltaEvent` | `reasoning_text.delta` |
| `FunctionCallArgumentDeltaEvent` | `tool_arguments.delta` |
| `ToolCallStartedEvent` | `tool_call.started` |
| `ServerToolCallStartedEvent` | `server_tool_call.started` |
| `OutputItemCompletedEvent` | `output_item.done` |
| `CustomProviderEvent` | `custom_provider_event` (the native event name is in `provider_type`) |

Field details are in [Streaming](streaming.md#event-reference).

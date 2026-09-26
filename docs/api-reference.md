# API reference

Everything listed as exported can be imported from the top-level package:

```python
from tessaract import (
    Tessaract,
    OpenAIProvider,
    FunctionTool, InputSchema, Property,
    UserMessage, SystemPrompt, FunctionToolResult,
    ReasoningOptions,
)
```

Other types can be imported from their modules, shown under each heading.

---

## Client

### `Tessaract`

`tessaract.client.Tessaract`

```python
Tessaract(providers: dict[str, OpenAIProvider])
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

**Raises:**

- `ValueError` if `model` isn't `"<prefix>/<model-name>"`, or if the prefix isn't registered
- `TypeError` if an item in `input` has an unsupported type
- any exception raised by the provider SDK, such as `openai.APIError`

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

---

## Requests

### `ReasoningOptions`

`tessaract.ReasoningOptions` (a dataclass)

| Field | Type | Default |
| --- | --- | --- |
| `effort` | `"none" \| "minimal" \| "low" \| "medium" \| "high" \| "extra_high" \| "max" \| None` | `None` |
| `summary` | `"concise" \| "auto" \| "detailed" \| None` | `None` |
| `mode` | `"standard" \| "pro" \| None` | `None` |

See [Reasoning](reasoning.md).

### `Request`

`tessaract.types.request.Request` (a dataclass)

This is the internal canonical request that `send()` builds and passes to an adapter. You only need it if you're writing an adapter.

| Field | Type |
| --- | --- |
| `model` | `str` (without the prefix) |
| `input` | `list`, already converted to native items |
| `instructions` | `str \| None` (not yet populated by `send()`; use `request_options={"instructions": ...}`) |
| `tools` | `list[FunctionTool] \| None` |
| `reasoning` | `ReasoningOptions \| None` |
| `stream` | `bool` |
| `provider_options` | `dict \| None` |

---

## Tools

### `FunctionTool`

`tessaract.FunctionTool` (a Pydantic model)

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | required |
| `description` | `str` | required |
| `input_schema` | `InputSchema \| None` | `None` |
| `strict` | `bool \| None` | `None` (the OpenAI adapter treats it as `True`) |
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

See [Tool calling](tool-calling.md).

---

## Input items

`tessaract.types.input_types`. Every input item subclasses `InputType` and implements `raw(adapter)`, which returns the provider-native form.

### `UserMessage`

| Field | Type | Default |
| --- | --- | --- |
| `role` | `"user"` | `"user"` |
| `content` | `str \| list[dict]` | required |

`content` can be a list of native content parts, such as OpenAI `input_text` / `input_image` dicts, for multimodal input.

### `SystemPrompt`

| Field | Type | Default |
| --- | --- | --- |
| `role` | `"system"` | `"system"` |
| `content` | `str \| list[dict]` | required |

Use `SystemPrompt(content="...")` in an input list to provide system instructions. It is mapped to an OpenAI input message with the `system` role.

### `FunctionToolResult`

| Field | Type | Default |
| --- | --- | --- |
| `type` | `"function_tool_result"` | `"function_tool_result"` |
| `call_id` | `str` | required |
| `result` | `Any` | required. Strings are sent as-is, and other values are JSON-encoded. |
| `is_error` | `bool` | `False`. When `True`, OpenAI receives `{"error": result}`. |

---

## Responses

### `Response`

`tessaract.types.response.Response` (a dataclass). The OpenAI adapter returns the subclass `OpenAIResponse`.

| Field / property | Type | Description |
| --- | --- | --- |
| `id` | `str` | Provider response ID |
| `model` | `str` | Model name |
| `status` | `ResponseStatus \| str` | e.g. `"completed"` or `"incomplete"` |
| `provider` | `Provider \| None` | The provider that produced the response |
| `output` | `list[OutputType]` | Typed output items, in order |
| `usage` | `Usage \| None` | Normalized token counts when the provider reports usage |
| `error` | `ResponseError \| None` | Set if the provider reports an error |
| `raw_response` | `Any` | The native SDK response object |
| `output_text` *(property)* | `str` | All assistant message text joined together |
| `response_id` *(property)* | `str` | An alias for `id` |

`Usage` includes `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, and `reasoning_tokens`. The detail fields are `None` when the provider does not report them. `OpenAIResponse.provider_usage` holds the native OpenAI SDK usage object, or `None` when usage is unavailable. Completed streaming responses expose both fields through `ResponseCompletedEvent.response`.

`OpenAIResponse.input_tokens` and `OpenAIResponse.cached_input_tokens` provide direct access to the corresponding normalized usage counts. Each returns `None` when usage is unavailable; `cached_input_tokens` is also `None` when the provider does not report that detail.

### `ResponseStatus`

A `str` enum with the values `queued`, `in_progress`, `completed`, `incomplete`, `failed` and `cancelled`.

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
| `ProviderOutputItem` | `"provider_output"` | `provider_type` (the native item type), `raw` |

The OpenAI adapter returns an assistant message as a `ProviderOutputItem` (`provider_type="message"`) when it contains anything other than plain `output_text`, such as refusals or annotations and citations. It still replays correctly, and you can inspect it through `.raw`.

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
| `OutputItemCompletedEvent` | `output_item.done` |
| `CustomProviderEvent` | the native event type |

Field details are in [Streaming](streaming.md#event-reference).

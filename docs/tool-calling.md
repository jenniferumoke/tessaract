# Tool calling

Tessaract has one canonical way to define **function tools**. Each provider adapter converts them to the native format: a Responses API `function` tool for OpenAI, and a Messages API `custom` tool with an `input_schema` for Anthropic.

```python
from tessaract import FunctionTool, InputSchema, Property
```

Tools the provider runs on its own servers, such as web search, are covered in [Server tools](#server-tools).

## Defining a tool

```python
get_weather = FunctionTool(
    name="get_weather",
    description="Get the current temperature for a city.",
    input_schema=InputSchema(
        properties={
            "city": Property(type="string", description="City name, e.g. Paris."),
            "units": Property(type="string", description="Either 'celsius' or 'fahrenheit'."),
        },
        required=["city", "units"],
        additionalProperties=False,
    ),
    strict=True,
)
```

### `FunctionTool`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | The name the model uses to call the tool. Use it as the key in your dispatch table. |
| `description` | `str` | required | Tells the model what the tool does and when to use it. |
| `input_schema` | `InputSchema \| None` | `None` | The arguments. Leave it out for tools that take no arguments. |
| `strict` | `bool \| None` | `None` | Enforce the schema exactly. Both adapters treat `None` as **`True`**. |
| `cache` | `bool` | `False` | Anthropic: a prompt-caching breakpoint on the tool definition. Ignored on OpenAI. |
| `provider_options` | `dict` | `{}` | Extra native fields merged into the provider's tool definition. |

### `InputSchema`

A JSON Schema `object`. It's validated strictly with Pydantic, and every name in `required` must also appear in `properties`. Otherwise you get a `ValueError`.

| Field | Type | Default |
| --- | --- | --- |
| `type` | `"object"` | `"object"` |
| `properties` | `dict[str, Property]` | `{}` |
| `required` | `list[str]` | `[]` |
| `additionalProperties` | `bool \| None` | `None` (the OpenAI adapter sends `False`) |

### `Property`

| Field | Type |
| --- | --- |
| `type` | `"string"`, `"number"`, `"integer"`, `"boolean"`, `"array"`, `"object"` or `"null"`, or a list of them such as `["string", "null"]` |
| `description` | `str \| None` |
| `enum` | `list \| None`: the allowed values |
| `items` | `Property \| None`: the element schema, for `"array"` |
| `properties` | `dict[str, Property] \| None`: nested fields, for `"object"` |
| `required` | `list[str] \| None`: required nested fields, for `"object"` |
| `additionalProperties` | `bool \| None`, for `"object"`. The OpenAI adapter sends `False` when it's `None`, which strict mode requires. |

Setting `items` on a non-array type, or the object fields on a non-object type, raises `ValueError`.

### A richer schema

```python
InputSchema(
    properties={
        "units": Property(type="string", description="Temperature units.", enum=["celsius", "fahrenheit"]),
        "cities": Property(
            type="array",
            description="Cities to look up.",
            items=Property(type="string"),
        ),
        "window": Property(
            type="object",
            description="Forecast window.",
            properties={
                "start": Property(type="string", description="ISO-8601 date."),
                "days": Property(type="integer", description="Number of days."),
            },
            required=["start", "days"],
        ),
        "note": Property(type=["string", "null"], description="Optional note, or null."),
    },
    required=["units", "cities", "window", "note"],
    additionalProperties=False,
)
```

## Strict mode

With `strict=True`, which both adapters use when `strict` is `None`, OpenAI uses structured outputs to guarantee that the arguments match the schema. OpenAI requires that:

- every key in `properties` is listed in `required`
- `additionalProperties` is `False`

For an optional argument in strict mode, keep it in `required` and make it nullable with `type=["string", "null"]`. Nested objects follow the same rules.

On Anthropic, `strict=True` turns on strict tool use, which also guarantees schema-valid arguments. It needs `additionalProperties: false` on every object, which both adapters send by default. Anthropic doesn't require every property to be in `required`, but listing them all keeps a tool portable across providers.

For a tool with **no arguments**, leave out `input_schema` and set `strict=False`. Anthropic requires a schema on every tool, so the Anthropic adapter sends an empty object schema:

```python
FunctionTool(name="get_time", description="Get the current UTC time.", strict=False)
```

## Sending tools

```python
response = client.send(model="oai/gpt-5.6-luna", input=history, tools=[get_weather, get_time])
```

## Handling function calls

The model's calls come back as `FunctionCallOutputItem`s in `response.output`:

```python
for item in response.output:
    if item.type == "function_call":
        item.call_id     # "call_abc123": links the call to its result
        item.name        # "get_weather"
        item.arguments   # {"city": "Paris", "units": "celsius"}, already parsed from JSON
        item.raw         # the native OpenAI ResponseFunctionToolCall, or Anthropic ToolUseBlock
```

A single response can contain **several** function calls, which is parallel tool calling. Handle them all.

## Returning results

Send each result back as a `FunctionToolResult` with the matching `call_id`, **after** the call itself is in the history:

```python
from tessaract import FunctionToolResult

history.extend(response.output)          # includes the function_call items

for call in (i for i in response.output if i.type == "function_call"):
    output = FUNCTIONS[call.name](**call.arguments)
    history.append(FunctionToolResult(call_id=call.call_id, result=output))

response = client.send(model="oai/gpt-5.6-luna", input=history, tools=tools)
```

### `FunctionToolResult`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `call_id` | `str` | required | The `call_id` of the `FunctionCallOutputItem` this result answers |
| `result` | `Any` | required | The tool's output. |
| `is_error` | `bool` | `False` | Marks the call as failed. |
| `cache` | `bool` | `False` | Puts a prompt-caching breakpoint on the result. On OpenAI (`gpt-5.6` or later), the output is sent as an `input_text` part so it can carry the breakpoint. |
| `provider_options` | `dict` | `{}` | Extra native fields merged into the result, such as Anthropic `cache_control`. They can't override the canonical fields. |

The OpenAI adapter maps it to `{"type": "function_call_output", "call_id": ..., "output": ...}`. The `output` depends on `result` and `is_error`:

| `result` | `is_error` | `output` sent to OpenAI |
| --- | --- | --- |
| `"sunny"` | `False` | `"sunny"` |
| `{"temp": 19}` or any other non-string | `False` | `'{"temp": 19}'` (JSON-encoded) |
| a list of content-part dicts, e.g. `[{"type": "input_text", ...}]` | `False` | passed through as-is |
| anything | `True` | `'{"error": <result>}'` (JSON-encoded) |

The Anthropic adapter maps it to a `tool_result` block, `{"type": "tool_result", "tool_use_id": ..., "content": ...}`, inside a user turn. Strings and lists of content blocks such as `[{"type": "text", ...}]` are sent as-is, and other values are JSON-encoded. Anthropic has a native error flag, so `is_error=True` sends `"is_error": true` with the result unchanged. All results that answer one response are grouped into a single user turn, as Anthropic requires.

OpenAI has no error flag on function call outputs, so Tessaract puts the error inside the output where the model can see it:

```python
try:
    output = FUNCTIONS[call.name](**call.arguments)
    history.append(FunctionToolResult(call_id=call.call_id, result=output))
except Exception as exc:
    history.append(FunctionToolResult(call_id=call.call_id, result=str(exc), is_error=True))
```

## Native tool fields (`provider_options`)

Everything in `provider_options` is merged into the native tool definition, except the canonical keys `type`, `name`, `description` and `strict`, plus `parameters` (OpenAI) or `input_schema` (Anthropic). The canonical fields always control those.

For an Anthropic prompt-caching breakpoint, use `cache=True` on the last tool to cache all the tool definitions. To set a native field such as a per-breakpoint `ttl`, pass `cache_control` in `provider_options`; it is kept as written:

```python
FunctionTool(
    name="get_weather",
    description="Get the current temperature for a city.",
    input_schema=...,
    provider_options={"cache_control": {"type": "ephemeral", "ttl": "1h"}},
)
```

Any other native field works the same way, for example an `output_schema`:

```python
FunctionTool(
    name="get_weather",
    description="Get the current temperature for a city.",
    input_schema=InputSchema(
        properties={"city": Property(type="string", description="City name.")},
        required=["city"],
        additionalProperties=False,
    ),
    provider_options={
        "output_schema": {
            "type": "object",
            "properties": {"temperature": {"type": "string"}},
            "required": ["temperature"],
            "additionalProperties": False,
        }
    },
)
```

## Controlling tool use

Tool choice isn't part of the canonical API yet. Pass the native parameters through `request_options`. For OpenAI:

```python
client.send(
    model="oai/gpt-5.6-luna",
    input=history,
    tools=tools,
    request_options={
        "tool_choice": "required",        # "auto" | "none" | "required" | {"type": "function", "name": "..."}
        "parallel_tool_calls": False,     # at most one call per turn
    },
)
```

For Anthropic, use `request_options={"tool_choice": {"type": "auto"}}` (or `"any"`, `"none"`, `{"type": "tool", "name": "..."}`). Anthropic only allows `"auto"` and `"none"` while thinking is on.

## Server tools

Server tools run on the provider's servers, not in your code. The model calls them during its turn, and the results come back in the response, so you don't send a result back. `tools=` accepts them next to function tools, in any order:

```python
from tessaract import UserLocation, WebSearchTool

tools = [
    WebSearchTool(max_uses=5, user_location=UserLocation(country="NL", city="Amsterdam")),
    get_weather_tool,
]
```

### Web search

`WebSearchTool` maps to OpenAI's `web_search` tool and Anthropic's `web_search_20250305` tool. Each search comes back as a `WebSearchCallOutputItem`:

```python
response = client.send(model="ant/claude-opus-5-5", input="What changed in Python 3.15?", tools=[WebSearchTool()])

for item in response.output:
    if item.type == "web_search_call":
        print(item.status, item.queries)            # "completed", ["python 3.15 what's new"]
        for source in item.sources:
            print(source.title, source.url)

    elif item.type == "assistant_message":
        for part in item.content:
            for citation in part.annotations:       # where the answer cites a source
                print(citation.title, citation.source)

print(response.usage.server_tool_requests)          # {"web_search": 1}
```

| Option | OpenAI | Anthropic |
| --- | --- | --- |
| `allowed_domains` | `filters.allowed_domains` | `allowed_domains` |
| `blocked_domains` | not supported | `blocked_domains` |
| `max_uses` | not supported | `max_uses` |
| `user_location` | `user_location` | `user_location` |
| `cache` | ignored (caching is automatic) | `cache_control` on the tool |

An option the target provider doesn't support raises `UnsupportedToolOptionError` (a `ValueError`) instead of being dropped. `allowed_domains` and `blocked_domains` cannot be combined.

Native fields go in `provider_options`, as on function tools. `"type"` pins the native tool version:

```python
WebSearchTool(provider_options={"search_context_size": "high"})     # OpenAI only
WebSearchTool(provider_options={"type": "web_search_20260209"})     # a newer Anthropic version
```

Provider differences to know about:

- OpenAI only returns sources when the request includes `web_search_call.action.sources`. The OpenAI adapter adds it to `include` when you send a `WebSearchTool`, next to any `include` values in `request_options`. OpenAI returns source URLs without titles.
- OpenAI can also browse a result. Those calls have `action="open_page"` or `"find_in_page"`, with `url` (and `pattern`) set and no `queries`.
- Anthropic returns each search as a `server_tool_use` block followed by a `web_search_tool_result` block. Tessaract merges the pair into one `WebSearchCallOutputItem`, and `raw` holds both blocks, which are replayed unchanged. When parallel searches interleave their blocks, each block stays its own item so the native order is kept.
- A long server-tool turn on Anthropic can pause (`incomplete_reason` is `"paused"`). Append the output to the history and send it again to let the model continue. A search that hadn't finished has `status="in_progress"`.

`Usage.server_tool_requests` counts server tool calls by tool name. Anthropic reports these counts; for OpenAI they are the `web_search_call` items in the output.

### Other provider tools (`ProviderTool`)

For a native tool without a canonical type yet, pass its native definition in a `ProviderTool`. It is sent unchanged, and its output comes back as `ProviderOutputItem`s, which replay through the history like any other item:

```python
from tessaract import ProviderTool

web_fetch = ProviderTool(
    provider="anthropic",
    definition={"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3},
)
```

`provider` is the provider the definition was written for. Sending a `ProviderTool` to a model from a different provider raises `ValueError`.

Native tools in `request_options={"tools": [...]}` are also sent when `tools=` is empty. When `tools=` is set, it wins.

## Streaming tool calls

When streaming, a `tool_call.started` event (with `call_id` and `name`) fires as soon as the model starts a call. The arguments then arrive as `tool_arguments.delta` events, which hold raw JSON fragments. The fully parsed call arrives in an `output_item.done` event and in the final `response.completed` event. In most cases, act on the completed items and don't parse the deltas.

For server tools, a `server_tool_call.started` event (with `call_id` and `name`, e.g. `"web_search"`) fires when the call starts, and the finished call arrives in `output_item.done`. Server tool input never arrives as `tool_arguments.delta` events. See [Streaming](streaming.md).

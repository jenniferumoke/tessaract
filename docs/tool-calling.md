# Tool calling

Tessaract has one canonical way to define **function tools**. Each provider adapter converts them to the native format: a Responses API `function` tool for OpenAI, and a Messages API `custom` tool with an `input_schema` for Anthropic.

```python
from tessaract import FunctionTool, InputSchema, Property
```

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

With `strict=True`, which is the OpenAI default here, OpenAI uses structured outputs to guarantee that the arguments match the schema. OpenAI requires that:

- every key in `properties` is listed in `required`
- `additionalProperties` is `False`

For an optional argument in strict mode, keep it in `required` and make it nullable with `type=["string", "null"]`. Nested objects follow the same rules.

Anthropic's strict tool use also needs `additionalProperties: false` on every object, which both adapters send by default. Anthropic doesn't require every property to be in `required`.

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

Everything in `provider_options` is merged into the native tool definition, except the canonical keys `type`, `name`, `description` and `strict`, plus `parameters` (OpenAI) or `input_schema` (Anthropic). The canonical fields always control those. A common Anthropic use is a prompt-caching breakpoint, `provider_options={"cache_control": {"type": "ephemeral"}}`:

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

## Streaming tool calls

When streaming, a `tool_call.started` event (with `call_id` and `name`) fires as soon as the model starts a call. The arguments then arrive as `tool_arguments.delta` events, which hold raw JSON fragments. The fully parsed call arrives in an `output_item.done` event and in the final `response.completed` event. In most cases, act on the completed items and don't parse the deltas. See [Streaming](streaming.md).

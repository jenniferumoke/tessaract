# Tool calling

Tessaract has one canonical way to define **function tools**. Each provider adapter converts them to the native format: a Responses API `function` tool for OpenAI, and a Messages API `custom` tool with an `input_schema` for Anthropic.

```python
from tessaract import FunctionTool, InputSchema, Property
```

Tools the provider runs on its own servers, such as web search, are covered in [Server tools](#server-tools). Provider-defined tools your code runs, computer use and shell, are covered in [Client-executed provider tools](#client-executed-provider-tools).

For complete agents built with these tools, and workflows that combine providers, see [Provider tools](provider-tools.md).

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

## Client-executed provider tools

Computer use and shell are tools the provider defines but your code runs. The model calls them like function tools: each call comes back as an output item, your code runs it, and you append a result to the history.

| Tool | OpenAI | Anthropic | Call | Result |
| --- | --- | --- | --- | --- |
| `ComputerUseTool` | `computer` | `computer_toolset_20260801` | `ComputerCallOutputItem` (`"computer_call"`) | `ComputerToolResult` |
| `ShellTool` | `shell` with a local environment | `bash_20250124` | `ShellCallOutputItem` (`"shell_call"`) | `ShellToolResult` |

`ShellTool` can also run commands in a container the provider manages, so the provider executes them; see [Provider-managed shell](#provider-managed-shell).

### Computer use

```python
from tessaract import ComputerToolResult, ComputerUseTool

tools = [ComputerUseTool()]

def run_turn(history: list) -> bool:
    response = client.send(model="ant/claude-opus-5-5", input=history, tools=tools)
    history.extend(response.output)

    calls = [item for item in response.output if item.type == "computer_call"]
    for call in calls:
        for action in call.actions:          # always a list
            match action.type:
                case "click":
                    desktop.click(action.x, action.y, button=action.button, clicks=action.clicks, hold=action.modifiers)
                case "type":
                    desktop.type(action.text)
                case "keypress":
                    desktop.press(action.keys, repeat=action.repeat)
                case "scroll":
                    desktop.scroll(action.x, action.y, action.dx, action.dy, unit=action.unit)
                case "screenshot" | "wait" | "move" | "drag" | "zoom":
                    ...
        history.append(ComputerToolResult(
            call_id=call.call_id,
            screenshot=desktop.screenshot_png(),
            acknowledged_safety_checks=call.safety_checks,   # after you confirmed them; always empty on Anthropic
        ))

    return bool(calls)
```

Sending a screenshot in every result works on both providers. The actions are canonical types in `tessaract.types.computer_actions`, matched on `action.type`:

| `type` | Class | Fields | OpenAI | Anthropic |
| --- | --- | --- | --- | --- |
| `click` | `Click` | `x`, `y` (`None` at the cursor), `button`, `clicks` (2 = double, 3 = triple), `modifiers` | `click`, `double_click` | `left_click`, `right_click`, `middle_click`, `double_click`, `triple_click` |
| `scroll` | `Scroll` | `x`, `y`, `dx`, `dy` (positive is right / down), `unit`, `modifiers` | `scroll`, in `"pixels"` | `scroll`, in wheel `"clicks"` |
| `type` | `TypeText` | `text` | `type` | `type` |
| `keypress` | `KeyPress` | `keys`, `repeat` | `keypress` | `key` |
| `move` | `MoveMouse` | `x`, `y`, `modifiers` | `move` | `mouse_move` |
| `drag` | `Drag` | `path: list[Point]`, `modifiers` | `drag` | `left_click_drag` (a two-point path) |
| `screenshot` | `Screenshot` | | `screenshot` | `screenshot` |
| `wait` | `Wait` | `duration` in seconds (`None` on OpenAI) | `wait` | `wait` |
| `mouse_down` / `mouse_up` | `MouseDown` / `MouseUp` | `button` | | `left_mouse_down` / `left_mouse_up` |
| `hold_key` | `HoldKey` | `keys`, `duration` | | `hold_key` |
| `zoom` | `Zoom` | `region: (x0, y0, x1, y1)`; return a screenshot of that region | | `zoom` |
| `cursor_position` | `CursorPosition` | return the position as text in `output` | | `cursor_position` |
| `provider_action` | `ProviderComputerAction` | `provider_type`, `payload`: a native action Tessaract doesn't map yet | any | any |

Coordinates are in the pixel space of the screenshots you return. Key names are passed through as the provider sends them, e.g. `"ctrl"` and `"Return"` from Anthropic, `"CTRL"` and `"ENTER"` from OpenAI.

The providers batch differently, and the results follow the calls:

- **OpenAI** can put several actions in one call. Run them all, then send one result with a screenshot. Every OpenAI result needs a `screenshot`, and `is_error=True` raises `UnsupportedToolOptionError`, since OpenAI has no error flag; send the screenshot of the current screen instead. `output` is ignored. A call can carry `safety_checks`; pass the ones you confirmed back in `acknowledged_safety_checks`.
- **Anthropic** sends one action per call, often several calls per turn, and every call needs its own result. Only `screenshot` and `zoom` actions need a `screenshot`; for the others `output` (or the default `"OK"`) is enough. Run the calls in order and stop at the first failure: answer the failed call with `is_error=True` and a description in `output`, and each later call with `is_error=True` and `output="Not executed: an earlier computer action in this turn failed."`. Tessaract adds the `toolset_name` that Anthropic requires on these results.

Native fields go in `provider_options`, e.g. `ComputerUseTool(provider_options={"configs": {"zoom": {"enabled": False}}})` to withhold Anthropic's zoom action. `ComputerUseTool` doesn't support Anthropic's earlier `computer_20251124` tool, whose calls have a different shape; use a `ProviderTool` for it.

### Shell

```python
import subprocess

from tessaract import ShellCommandOutput, ShellTool, ShellToolResult

for call in (item for item in response.output if item.type == "shell_call"):
    if call.executed_by == "provider":       # already ran in a provider container; nothing to send back
        continue

    if call.restart:                         # Anthropic: restart your shell session
        shell = start_shell()
        history.append(ShellToolResult(call_id=call.call_id))
        continue

    outputs = []
    for command in call.commands:            # OpenAI can send several, Anthropic sends one
        timeout = call.timeout_ms / 1000 if call.timeout_ms else 60
        try:
            done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
            outputs.append(ShellCommandOutput(stdout=done.stdout, stderr=done.stderr, exit_code=done.returncode))
        except subprocess.TimeoutExpired as exc:
            outputs.append(ShellCommandOutput(stdout=exc.stdout or "", timed_out=True))

    history.append(ShellToolResult(call_id=call.call_id, output=outputs, max_output_length=call.max_output_length))
```

The model runs commands on your machine, so run them in a sandbox you control.

- OpenAI receives each command's `stdout`, `stderr` and exit or timeout outcome. Pass the call's `max_output_length` back on the result. `is_error=True` raises `UnsupportedToolOptionError`; report failures in `stderr` with a non-zero `exit_code`.
- Anthropic receives one text block: `stdout`, then `stderr`, then `Exit code: N` when it isn't 0 (or `Command timed out.`). A result with no output answers a restart. `is_error=True` sets Anthropic's error flag; use it when a command couldn't run at all.
- On Anthropic, a `bash` tool call is only read as a `shell_call` when the request includes a `ShellTool`. Otherwise it stays a `function_call`, so a function tool of your own named `bash` keeps working.
- On OpenAI, the fields of `provider_options={"environment": {...}}` are merged into the environment, e.g. to add skills.

### Provider-managed shell

With a `ContainerEnvironment`, the provider runs the commands in a container it manages: OpenAI's hosted shell, or Anthropic's code execution tool. The same code works on both:

```python
from tessaract import ContainerEnvironment, ShellTool

tools = [ShellTool(environment=ContainerEnvironment())]

for model in ("oai/gpt-6.1-sol", "ant/claude-opus-5-5"):
    response = client.send(model=model, input="Which Python version is installed?", tools=tools)

    for call in (item for item in response.output if item.type == "shell_call"):
        print(call.executed_by)              # "provider"
        for command, output in zip(call.commands, call.output):
            print(command, output.exit_code, output.stdout)

    # keep using the same container, with its files, on the next request
    tools = [ShellTool(environment=ContainerEnvironment(container_id=response.container_id))]
```

These calls have `executed_by="provider"` and their commands' `output` already filled in. You send nothing back; append the output to the history as usual. A call whose output hasn't arrived yet has an empty `output`. `Response.container_id` names the container the calls ran in.

Each provider returns a call as two native pieces: OpenAI a `shell_call` item and a `shell_call_output` item, Anthropic a `server_tool_use` block (`bash_code_execution`) and a `bash_code_execution_tool_result` block. Tessaract merges the pair into one `ShellCallOutputItem` whose `raw` holds both, and both are replayed unchanged.

| `ContainerEnvironment` field | OpenAI | Anthropic |
| --- | --- | --- |
| *(none set)* | `{"type": "container_auto"}`, a new container | `code_execution_20250825`, a new container |
| `container_id` | `{"type": "container_reference"}`: reuse a container. It cannot be combined with the fields below | the top-level `container` parameter |
| `file_ids` | `file_ids`, files copied into a new container | `container_upload` blocks, added at the start of the first user message |
| `memory_limit` | `memory_limit`: `"1g"`, `"4g"`, `"16g"` or `"64g"` | not supported (fixed at 5 GiB) |
| `allowed_domains` | `network_policy` `{"type": "allowlist"}` | not supported (no network access) |
| `network_disabled` | `network_policy` `{"type": "disabled"}` | how Anthropic containers always run |
| `provider_options` | merged into the environment, e.g. `skills`, or a `network_policy` with `domain_secrets` | sent as the top-level `container` object, e.g. `skills` |

Options a provider doesn't support raise `UnsupportedToolOptionError`. On Anthropic:

- `ShellTool.provider_options={"type": ...}` pins the code execution version, e.g. `"code_execution_20260120"` for programmatic tool calling. Anthropic's current web search and web fetch versions need `code_execution_20260120` or later.
- `ShellCallOutputItem.file_ids` lists the files a command left in the container's output directory, for the Files API. `error` holds the error code when the call failed; `"execution_time_exceeded"` also gives a timed-out output.
- Code execution can also view and edit files. Those `text_editor_code_execution` calls come back as `ProviderOutputItem`s.
- One request uses one container, so all `ShellTool`s with a `ContainerEnvironment` in a request must have the same one.

When streaming, a provider-managed call starts with a `server_tool_call.started` event (`name="shell"`), not `tool_call.started`, and its `output_item.done` arrives with its output.

### Results for other provider tools (`ProviderToolResult`)

When a `ProviderTool` defines a tool your code runs, such as OpenAI's `apply_patch` or Anthropic's text editor, answer its calls with a `ProviderToolResult`, which is sent unchanged:

```python
from tessaract import ProviderToolResult

# OpenAI: a native input item
ProviderToolResult(provider="openai", item={"type": "apply_patch_call_output", "call_id": call_id, "status": "completed"})

# Anthropic: a content block, sent in a user turn
ProviderToolResult(provider="anthropic", item={"type": "tool_result", "tool_use_id": call_id, "content": "Edited."})
```

Sending it to a model from a different provider raises `ValueError`.

## Type names

The `type` values your code matches on are public API, and they follow one rule:

- an output item for a tool call is `<tool>_call`: `function_call`, `web_search_call`, `computer_call`, `shell_call`
- a result you send back is `<tool>_tool_result`: `function_tool_result`, `computer_tool_result`, `shell_tool_result`, `provider_tool_result`
- `tool_call.started` fires for every call your code runs, and its `call_type` is the type of the output item the call becomes; `server_tool_call.started` fires for calls the provider runs

`ProviderOutputItem` holds native items that have no canonical type yet. When Tessaract adds one, those items come back as the new type instead, so code that looks for a native type inside `ProviderOutputItem` (for example `item.type == "provider_output" and item.provider_type == "web_search_call"`) has to match the new type. Every such change is listed in the [changelog](https://github.com/jenniferumoke/tessaract/blob/main/CHANGELOG.md). Replaying output into the history works the same either way.

## Streaming tool calls

When streaming, a `tool_call.started` event (with `call_id`, `name` and `call_type`) fires as soon as the model starts a call your code runs. `call_type` is `"function_call"`, `"computer_call"` or `"shell_call"`, and `name` is the function name, or `"computer"` or `"shell"`. The arguments then arrive as `tool_arguments.delta` events, which hold raw JSON fragments. The fully parsed call arrives in an `output_item.done` event and in the final `response.completed` event. In most cases, act on the completed items and don't parse the deltas.

For server tools, a `server_tool_call.started` event (with `call_id` and `name`, e.g. `"web_search"`) fires when the call starts, and the finished call arrives in `output_item.done`. Server tool input never arrives as `tool_arguments.delta` events. See [Streaming](streaming.md).

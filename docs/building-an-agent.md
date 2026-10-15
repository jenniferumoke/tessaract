# Building an agent with reasoning and tool calling

This tutorial builds a small agent with Tessaract. It uses OpenAI by default, and the same code runs on Anthropic's Claude models by changing the model prefix (see [Run it on Anthropic](#run-it-on-anthropic)). By the end, the agent will:

- think before answering, and show you a summary of its reasoning
- decide when to call your Python functions, and with which arguments
- run several tool calls in one turn, and chain calls across turns
- keep a multi-turn conversation with the user
- stream reasoning and text as they're generated

Runnable code for this guide:

- [`examples/agent.py`](https://github.com/jenniferumoke/tessaract/blob/main/examples/agent.py) is the synchronous agent
- [`examples/streaming_agent.py`](https://github.com/jenniferumoke/tessaract/blob/main/examples/streaming_agent.py) is the streaming agent

Both read the model from `TESSARACT_MODEL`, so you can run them on Anthropic with:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
TESSARACT_MODEL=ant/claude-opus-5-5 uv run --extra anthropic python examples/agent.py
```

---

## How the loop works

```
            ┌───────────────────────────────────────────────┐
            │                                               │
 user ──▶ history ──▶ client.send(tools, reasoning) ──▶ response.output
                                                            │
                               ┌────────────────────────────┤
                               ▼                            ▼
                      any function_call?           no → answer the user
                               │ yes
                               ▼
                 run each function in Python
                 append FunctionToolResult(call_id, result)
                               │
                               └──────────▶ back to client.send
```

The model never runs your code. It returns `function_call` items, your code runs them, and you send the results back. Each round trip is one `client.send()` call, and the **history list** is the agent's entire memory.

---

## Step 1: Set up the client

```python
import json
import os
from datetime import datetime, timezone

from tessaract import (
    FunctionTool,
    FunctionToolResult,
    InputSchema,
    OpenAIProvider,
    Property,
    ReasoningOptions,
    Tessaract,
    UserMessage,
)

client = Tessaract(
    providers={"oai": OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])}
)

MODEL = "oai/gpt-5.6-luna"   # any OpenAI reasoning model, prefixed with your provider key
```

## Step 2: Write the tool functions

Tools are plain Python functions. Two rules make them work well with the agent loop:

1. **Return a string or JSON-serializable data.** OpenAI expects a function call's output to be a string, and Anthropic expects a string or content blocks. Tessaract sends strings as-is and JSON-encodes anything else.
2. **Don't raise.** Catch errors and return an error message instead, so the model can recover. An exception would stop your loop.

```python
def get_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_weather(city: str) -> str:
    temps = {"paris": "19C", "amsterdam": "20C", "lagos": "31C"}
    temp = temps.get(city.lower())
    if temp is None:
        return json.dumps({"error": f"No weather data for {city!r}"})
    return json.dumps({"city": city, "temperature": temp})


FUNCTIONS = {
    "get_time": get_time,
    "get_weather": get_weather,
}
```

## Step 3: Describe the tools to the model

The model sees only the name, the description and the input schema. Write descriptions for the model to read: say what the tool does and when to use it.

```python
tools = [
    FunctionTool(
        name="get_time",
        description="Get the current UTC time as an ISO-8601 string. Use for any question about the current time or date.",
        strict=False,
    ),
    FunctionTool(
        name="get_weather",
        description="Get the current temperature for a city.",
        input_schema=InputSchema(
            properties={
                "city": Property(type="string", description="City name, e.g. Paris."),
            },
            required=["city"],
            additionalProperties=False,
        ),
        strict=True,
    ),
]
```

Notes:

- **No arguments?** Leave out `input_schema` and set `strict=False`, as `get_time` does.
- **`strict=True`** (the default on both adapters) makes the provider guarantee that the arguments match your schema exactly. OpenAI's strict mode also requires **every** property to be listed in `required`, so do that for portable tools. Both adapters already default `additionalProperties` to `False`.
- `Property` supports `"string"`, `"number"`, `"integer"`, `"boolean"`, `"array"` (with `items`), `"object"` (with nested `properties`), `"null"`, nullable types such as `["string", "null"]`, and `enum`.

More options are in [Tool calling](tool-calling.md).

## Step 4: Choose reasoning settings

```python
reasoning = ReasoningOptions(effort="medium", summary="auto")
```

- `effort` sets how long the model thinks: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"extra_high"` or `"max"`. Higher effort gives better answers on hard problems, at the cost of more latency and tokens. `"low"` or `"medium"` is enough for tool routing.
- `summary` asks the model for a human-readable summary of its reasoning (`"auto"`, `"concise"` or `"detailed"`). Without it, reasoning items come back with an empty summary.

On Anthropic, these settings become adaptive thinking with `display="summarized"` and `output_config.effort="medium"`. Which effort levels are supported depends on the model. See [Reasoning](reasoning.md).

## Step 5: Write the agent loop

```python
INSTRUCTIONS = "You are a helpful assistant. Use tools when they help; be concise."
MAX_STEPS = 10


def execute(call) -> FunctionToolResult:
    """Run one function call and wrap its output (or error) for the model."""
    fn = FUNCTIONS.get(call.name)
    if fn is None:
        return FunctionToolResult(call_id=call.call_id, result=f"Unknown tool {call.name!r}", is_error=True)
    try:
        result = fn(**call.arguments)          # arguments is already a parsed dict
    except Exception as exc:                   # report the error to the model instead of crashing
        return FunctionToolResult(call_id=call.call_id, result=str(exc), is_error=True)
    return FunctionToolResult(call_id=call.call_id, result=result)


def run_turn(history: list, user_text: str) -> str:
    history.append(UserMessage(content=user_text))

    for _ in range(MAX_STEPS):
        response = client.send(
            model=MODEL,
            input=history,
            tools=tools,
            reasoning=reasoning,
            instructions=INSTRUCTIONS,
        )

        # (a) Save EVERYTHING the model produced: reasoning, messages and calls
        history.extend(response.output)

        # (b) Show the reasoning summaries
        for item in response.output:
            if item.type == "reasoning" and item.text:
                print(f"  [thinking] {item.text}")

        # (c) No function calls means the model has answered
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        # (d) Run every call and append one result per call_id
        for call in calls:
            print(f"  [tool] {call.name}({call.arguments})")
            history.append(execute(call))

    return "Stopped: too many tool-calling steps."
```

A few things in this loop matter:

- **(a) Extend the history with all of `response.output`.** The `function_call` items must be in the history before their `FunctionToolResult`s, or the provider rejects the request. Keeping the `reasoning` items lets the model build on its earlier thinking across tool calls, and Anthropic requires them when thinking is on.
- **(c) Stop when there are no calls.** A turn can contain reasoning, text and function calls together. If there are no function calls, the model is done.
- **(d) Answer every call.** The model can ask for several tools at once, which is parallel tool calling. Each `call_id` needs exactly one `FunctionToolResult`. `execute()` returns one even when the tool fails, marked with `is_error=True`.
- **`MAX_STEPS`** limits the loop, so a model that keeps calling tools can't run forever.

## Step 6: Chat with the agent

```python
def main():
    history: list = []
    print("Type 'q' to quit.")
    while True:
        user_text = input("\nyou > ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        answer = run_turn(history, user_text)
        print(f"agent > {answer}")


if __name__ == "__main__":
    main()
```

Example session:

```
you > What's the weather in Paris and Amsterdam, and what time is it?
  [thinking] The user wants weather for two cities plus the time; I'll call all three tools.
  [tool] get_weather({'city': 'Paris'})
  [tool] get_weather({'city': 'Amsterdam'})
  [tool] get_time({})
agent > It's 19C in Paris and 20C in Amsterdam. The current UTC time is 14:02.

you > Which one is warmer?
agent > Amsterdam, by one degree.
```

The second question works without calling a tool again because the earlier results are still in `history`.

---

## Stream the agent

For a responsive UI, stream the model's reasoning and text as they're generated. The loop is the same. The only change is where the `Response` comes from: the stream's final `response.completed` event.

```python
def run_turn_streaming(history: list, user_text: str) -> None:
    history.append(UserMessage(content=user_text))

    for _ in range(MAX_STEPS):
        completed = None

        for event in client.send(
            model=MODEL,
            input=history,
            tools=tools,
            reasoning=reasoning,
            instructions=INSTRUCTIONS,
            stream=True,
        ):
            match event.type:
                case "reasoning.started":
                    print("\n  [thinking] ", end="", flush=True)
                case "reasoning_summary.delta":
                    print(event.delta, end="", flush=True)
                case "text.delta":
                    print(event.delta, end="", flush=True)
                case "tool_call.started":
                    print(f"\n  [tool] calling {event.name}...", flush=True)
                case "output_item.done" if event.item.type == "function_call":
                    print(f"  [tool] {event.item.name}({event.item.arguments})")
                case "response.failed":
                    raise RuntimeError(f"Response failed: {event.message}")
                case "response.completed":
                    completed = event.response

        if completed is None:
            raise RuntimeError("Stream ended without a response.completed event")

        history.extend(completed.output)

        calls = [item for item in completed.output if item.type == "function_call"]
        if not calls:
            print()
            return

        for call in calls:
            history.append(execute(call))
```

- `tool_call.started` fires as soon as the model starts a function call, before its arguments stream in. `output_item.done` fires when each item is finished, with the fully parsed item.
- `response.failed` fires if the provider reports a failure or sends an error event.
- `response.completed.response.output` holds the same typed items as a non-streaming response. Use it for the history and the tool calls instead of rebuilding them from deltas.
- To show raw reasoning text as well (on models that expose it), handle `reasoning_text.delta`.

All the event types are listed in [Streaming](streaming.md).

---

## Run it on Anthropic

Nothing in the loop is OpenAI-specific. Register an `AnthropicProvider` and change the model:

```python
from tessaract import AnthropicProvider

client = Tessaract(providers={"ant": AnthropicProvider()})   # reads ANTHROPIC_API_KEY
MODEL = "ant/claude-opus-5-5"
```

Then pass `max_tokens` to `client.send()`. Anthropic requires it, and Tessaract sends 16,000 (`DEFAULT_ANTHROPIC_MAX_TOKENS`) when you leave it out, but output limits differ between models, so set it explicitly:

```python
response = client.send(
    model=MODEL,
    input=history,
    tools=tools,
    reasoning=reasoning,
    instructions=INSTRUCTIONS,
    max_tokens=16_000,
)
```

What the Anthropic adapter does for you:

- **Turns.** Each response's thinking, text and `tool_use` blocks are grouped into one assistant turn, and the `FunctionToolResult`s that follow are grouped into one user turn, as the Messages API requires.
- **Instructions.** `instructions=` and any `SystemPrompt` items are sent as the top-level `system` parameter.
- **Errors.** `is_error=True` is sent as Anthropic's native `is_error` flag on the `tool_result`.
- **Thinking.** Thinking blocks come back as `ReasoningOutputItem`s with their signature in `.raw`, and are replayed unchanged. Keep them in the history: Anthropic rejects a tool-use turn whose thinking is missing or modified.

To stop the loop on the right condition, you can also check `response.stop_reason == "tool_use"` on Anthropic responses, but looking for `function_call` items works on both providers.

You can register both providers on one client and pick per request. Keep each conversation's history on one provider, because output items replay in their native form.

---

## Production tips

| Concern | Recommendation |
| --- | --- |
| **Tool errors** | Catch exceptions in `execute()` and return `FunctionToolResult(call_id=..., result=str(exc), is_error=True)`. Models usually recover by retrying or asking the user. |
| **Runaway loops** | Keep a `MAX_STEPS` limit. |
| **Growing context** | `history` grows with every turn. Trim or summarize old turns in long sessions, but never separate a `function_call` from its `FunctionToolResult`. |
| **Bad arguments** | Use `strict=True` with a complete `required` list so the provider enforces the schema. Validate arguments again in your functions if they have side effects. |
| **Dangerous tools** | Ask the user to confirm before running tools that write, delete, spend money or send messages. |
| **Stateless mode** | If you run OpenAI with `request_options={"store": False}`, it can't look up earlier reasoning items by ID. Add `"include": ["reasoning.encrypted_content"]` so the reasoning items carry their encrypted content back in the history. Anthropic is stateless already, and thinking blocks carry their signature. |
| **Truncated output** | `response.status == "incomplete"` means the answer stopped early. Check `response.incomplete_reason`: for `"max_tokens"`, raise `max_tokens`; for `"paused"`, append the output to the history and send again to let the model continue. |
| **Prompt caching** | Pass `cache=CacheOptions()` to `send()` for automatic caching on both providers, and `cache=True` on the system prompt or last tool for a breakpoint on the stable prefix (on OpenAI, breakpoints need `gpt-5.6` or later). See [Prompt caching](getting-started.md#prompt-caching). |
| **Debugging** | `response.raw_response` (and `event.raw_event` when streaming) holds the untouched OpenAI or Anthropic SDK object. |
| **Unmodeled output** | Items Tessaract doesn't model yet, such as messages with citations, come back as `ProviderOutputItem` with the native object in `.raw`. They still replay correctly in the history. |

## Where to go next

- [Tool calling reference](tool-calling.md)
- [Reasoning reference](reasoning.md)
- [Streaming events](streaming.md)
- [API reference](api-reference.md)

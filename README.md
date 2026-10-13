# Tessaract

A provider-agnostic SDK for building AI agents natively.

Tessaract exists to make it straightforward to build agents that work across multiple providers. It is an unopinionated portal to multi-model intelligence giving you model diversity and full control over how you compose models in your applications. 

Today, Tessaract supports OpenAI's Responses API and Anthropic's Messages API, including streaming, reasoning, and tool calling. More provider adapters are planned.

More concretely, Tessaract exposes a **canonical model of the LLM ecosystem**: requests, responses, content, streaming events, tools, usage, errors and reasoning. You write an agent loop once against Tessaract's types, and a provider **adapter** translates to and from each vendor's native API.


> **Status:** OpenAI and Anthropic are supported today. The same agent code runs on both; switch providers by changing the model prefix.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Build an agent with reasoning and tool calling](#build-an-agent-with-reasoning-and-tool-calling)
- [Streaming](#streaming)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Status and roadmap](#status-and-roadmap)

---

## Features

- **Provider prefixes.** Register a provider under a prefix and address models as `"<prefix>/<model>"`, e.g. `"oai/gpt-5.6-luna"`.
- **Canonical tools.** Define function tools once with `FunctionTool` / `InputSchema` / `Property`, including nested objects, arrays, enums and nullable fields. The adapter emits the provider's native schema.
- **Canonical reasoning.** Control effort and summaries with `ReasoningOptions`; read reasoning back as `ReasoningOutputItem`s.
- **Typed output.** Responses contain `AssistantMessage`, `ReasoningOutputItem`, `FunctionCallOutputItem` and `ProviderOutputItem` objects. Function-call arguments are already parsed into a `dict`.
- **Canonical streaming events.** Text, reasoning and tool-argument deltas are normalized into typed events (`text.delta`, `reasoning_summary.delta`, `response.completed`, …).
- **Lossless multi-turn history.** Every output item keeps its provider-native `raw` payload, so you can append a response's output straight back into the conversation history.
- **Escape hatches everywhere.** `request_options` passes any native request parameter through, `FunctionTool.provider_options` adds native tool fields, and `raw_response` / `raw_event` expose the original SDK objects.

---

## Installation

Tessaract requires **Python 3.11+**. Install the extra for each provider you use: `openai`, `anthropic`, or both.

With pip:

```bash
python -m pip install "tessaract[openai]"             # OpenAI
python -m pip install "tessaract[anthropic]"          # Anthropic
python -m pip install "tessaract[openai,anthropic]"   # both
```

With uv:

```bash
uv add "tessaract[openai,anthropic]"
```

---

## Quickstart

Set the API key for each provider you use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Then send your first request:

```python
from tessaract import AnthropicProvider, OpenAIProvider, Tessaract

client = Tessaract(providers={"oai": OpenAIProvider(), "ant": AnthropicProvider()})

response = client.send(model="oai/gpt-6-luna", input="Say hello in five words.")
print(response.output_text)

response = client.send(model="ant/claude-opus-5-5", input="Say hello in five words.")
print(response.output_text)
```

`OpenAIProvider` reads `OPENAI_API_KEY` and `AnthropicProvider` reads `ANTHROPIC_API_KEY` from the environment. Register only the providers you need. The `oai` and `ant` prefixes are your choice; Tessaract sends the model name after `/` to the provider registered under that prefix. For multi-turn conversations, try the [simple chat loop](https://tessaractai.com/getting-started/#a-simple-chat-loop).

---

## Build an agent with reasoning and tool calling

An agent is a loop:

1. Send the conversation history, plus the tools the model may use.
2. Append the model's output (reasoning, messages and function calls) to the history.
3. If the model asked for function calls, run them and append a `FunctionToolResult` for each one.
4. Repeat until the model replies without calling a tool.

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

# 1. Plain Python functions the agent can call
def get_time() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_weather(city: str) -> str:
    temps = {"paris": "19C", "amsterdam": "20C"}
    return json.dumps({"temperature": temps.get(city.lower(), "unknown")})

FUNCTIONS = {"get_time": get_time, "get_weather": get_weather}

# 2. Describe them to the model
tools = [
    FunctionTool(
        name="get_time",
        description="Get the current UTC time as an ISO-8601 string.",
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
    ),
]

client = Tessaract(
    providers={"oai": OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])}
)

# 3. The agent loop
def run_agent(history: list, user_text: str) -> str:
    history.append(UserMessage(content=user_text))

    while True:
        response = client.send(
            model="oai/gpt-5.6-luna",
            input=history,
            tools=tools,
            reasoning=ReasoningOptions(effort="medium", summary="auto"),
            instructions="You are a concise, helpful assistant.",
        )

        # Keep reasoning, messages and calls in the history for the next turn
        history.extend(response.output)

        for item in response.output:
            if item.type == "reasoning" and item.text:
                print(f"[thinking] {item.text}")

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        for call in calls:
            result = FUNCTIONS[call.name](**call.arguments)  # arguments is already a dict
            history.append(FunctionToolResult(call_id=call.call_id, result=result))


history: list = []
print(run_agent(history, "What's the weather in Paris, and what time is it?"))
```

To run the same agent on Anthropic, register `AnthropicProvider` and change `model` to `"ant/claude-opus-5-5"`. Nothing else in the loop changes. Anthropic requires `max_tokens`; Tessaract sends 16,000 when you don't set it, but passing `max_tokens=` explicitly is recommended.

The step-by-step walkthrough is in the [building an agent guide](https://tessaractai.com/building-an-agent/), with runnable versions in [`examples/`](https://github.com/jenniferumoke/tessaract/tree/main/examples).

---

## Streaming

Pass `stream=True` to get an iterator of canonical events. The final event is `response.completed`, which carries the same `Response` object that a non-streaming call returns, so the agent loop doesn't change:

```python
completed = None

for event in client.send(
    model="oai/gpt-5.6-luna",
    input=history,
    tools=tools,
    stream=True,
    reasoning=ReasoningOptions(effort="medium", summary="auto"),
):
    if event.type == "reasoning.started":
        print("\n[thinking] ", end="")
    elif event.type == "reasoning_summary.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.completed":
        completed = event.response

history.extend(completed.output)
```

See the [streaming guide](https://tessaractai.com/streaming/) for the full event list.

---

## Documentation

The documentation site is [tessaractai.com](https://tessaractai.com/).

| Guide | What it covers |
| --- | --- |
| [Getting started](https://tessaractai.com/getting-started/) | Installing, configuring providers, sending your first request, and building multi-turn history |
| [Building an agent](https://tessaractai.com/building-an-agent/) | A step-by-step tutorial for an agent with reasoning and tool calling, both synchronous and streaming |
| [Tool calling](https://tessaractai.com/tool-calling/) | `FunctionTool`, `InputSchema`, `Property`, strict mode, tool results and `provider_options` |
| [Reasoning](https://tessaractai.com/reasoning/) | `ReasoningOptions`, effort levels, summaries, and reading and preserving reasoning items |
| [Streaming](https://tessaractai.com/streaming/) | Every canonical stream event, and how to build a streaming agent loop |
| [API reference](https://tessaractai.com/api-reference/) | Every public class, field and method |
| [Architecture](https://tessaractai.com/architecture/) | Canonical types, adapters and providers, and how to add a new provider |

---

## Project layout

```
src/tessaract/
├── client.py                    # Tessaract client: routes requests to adapters
├── providers/
│   ├── provider.py              # Provider base dataclass
│   ├── openai_provider.py       # OpenAIProvider (wraps openai.OpenAI)
│   └── anthropic_provider.py    # AnthropicProvider (wraps anthropic.Anthropic)
├── adapters/
│   ├── adapter.py               # Adapter base class + protocols
│   ├── openai/openai_adapter.py # Canonical <-> OpenAI Responses API translation
│   └── anthropic/anthropic_adapter.py # Canonical <-> Anthropic Messages API translation
├── tools/function.py            # FunctionTool, InputSchema, Property
└── types/
    ├── input_types.py           # UserMessage, FunctionToolResult
    ├── output_types.py          # AssistantMessage, ReasoningOutputItem, FunctionCallOutputItem, ...
    ├── request.py               # Request, ReasoningOptions
    ├── response.py              # Response, ResponseStatus, ResponseError
    └── streaming/event_types.py # Canonical stream events
```

---

## Status and roadmap

**Working today (OpenAI):**

- Synchronous and streaming requests via the Responses API
- Multi-turn conversations
- Function tools with parallel calls
- Reasoning effort and summaries
- Pass-through request options
- Token usage on `Response`, including OpenAI usage details

**Working today (Anthropic):**

- Synchronous and streaming requests via the Messages API
- Multi-turn conversations, with automatic grouping into user/assistant turns
- Function tools with parallel calls, and native tool-result error flags
- Adaptive, budgeted and between-tools thinking, effort, and thinking display (including beta `updates`)
- System prompts via `instructions=` or `SystemPrompt`, and prompt caching through `cache_control` pass-through
- Token usage on `Response`, normalized so `input_tokens` includes cached tokens

**Planned:**

- Google GenAI provider and adapter
- Finish details on `Response`
- TestProvider for CI and testing without making live API calls
- built-in utils: agent loop helper, tool schema autowriter
- Token counting

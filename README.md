# Tessaract

A provider-agnostic SDK for building AI agents natively.

Tessaract exists to make it straightforward to build agents that work across multiple providers. It is an unopinionated portal to multi-model intelligence giving you model diversity and full control over how you compose models in your applications. 

With a single SDK, you can build an agent loop that supports both Anthropic and OpenAI models with full support for native caching, reasoning and tool calling.

More concretely, Tessaract exposes a **canonical model of the LLM ecosystem**: requests, responses, content, streaming events, tools, usage, errors and reasoning. You write an agent loop once against Tessaract's types, and a provider **adapter** translates to and from each vendor's native API.


> **BUILD IN PROGRESS ⛏️🛠️🚜**
> OpenAI (via the Responses API) is the only provider implemented today. Anthropic support is planned.


## Quickstart

Set your OPENAI_API_KEY:

```bash
export OPENAI_API_KEY="sk..."
```

Make your first request:
```python
import os

from tessaract import OpenAIProvider, Tessaract

client = Tessaract(
    providers={"oai": OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])}
)

response = client.send(model="oai/gpt-5.6-luna", input="Say hello in five words.")

print(response.output_text)
```

- `providers` maps a **prefix** (`"oai"`) to a provider instance. You can pick any prefix.
- `model` is always `"<prefix>/<model-name>"`. Tessaract uses the prefix to choose the provider and sends the rest to the API.
- `input` accepts a string, or a list of strings and message objects (see below).

> If you leave out `api_key`, `OpenAIProvider` reads `OPENAI_API_KEY` from the environment. It raises `ValueError` if neither is set.

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

- **One client, many providers.** Register providers under a prefix and address models as `"<prefix>/<model>"`, e.g. `"oai/gpt-5.6-luna"`.
- **Canonical tools.** Define function tools once with `FunctionTool` / `InputSchema` / `Property`, including nested objects, arrays, enums and nullable fields. The adapter emits the provider's native schema.
- **Canonical reasoning.** Control effort and summaries with `ReasoningOptions`; read reasoning back as `ReasoningOutputItem`s.
- **Typed output.** Responses contain `AssistantMessage`, `ReasoningOutputItem`, `FunctionCallOutputItem` and `ProviderOutputItem` objects. Function-call arguments are already parsed into a `dict`.
- **Canonical streaming events.** Text, reasoning and tool-argument deltas are normalized into typed events (`text.delta`, `reasoning_summary.delta`, `response.completed`, …).
- **Lossless multi-turn history.** Every output item keeps its provider-native `raw` payload, so you can append a response's output straight back into the conversation history.
- **Escape hatches everywhere.** `request_options` passes any native request parameter through, `FunctionTool.provider_options` adds native tool fields, and `raw_response` / `raw_event` expose the original SDK objects.

---

## Installation

Tessaract requires **Python 3.11+**. 

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
            request_options={"instructions": "You are a concise, helpful assistant."},
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

The step-by-step walkthrough is in **[docs/building-an-agent.md](docs/building-an-agent.md)**, and runnable versions are in [`examples/`](examples/).

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

See **[docs/streaming.md](docs/streaming.md)** for the full event list.

---

## Documentation

| Guide | What it covers |
| --- | --- |
| [Getting started](docs/getting-started.md) | Installing, configuring providers, sending your first request, and building multi-turn history |
| [Building an agent](docs/building-an-agent.md) | A step-by-step tutorial for an agent with reasoning and tool calling, both synchronous and streaming |
| [Tool calling](docs/tool-calling.md) | `FunctionTool`, `InputSchema`, `Property`, strict mode, tool results and `provider_options` |
| [Reasoning](docs/reasoning.md) | `ReasoningOptions`, effort levels, summaries, and reading and preserving reasoning items |
| [Streaming](docs/streaming.md) | Every canonical stream event, and how to build a streaming agent loop |
| [API reference](docs/api-reference.md) | Every public class, field and method |
| [Architecture](docs/architecture.md) | Canonical types, adapters and providers, and how to add a new provider |

---

## Project layout

```
src/tessaract/
├── client.py                    # Tessaract client: routes requests to adapters
├── providers/
│   ├── provider.py              # Provider base dataclass
│   └── openai_provider.py       # OpenAIProvider (wraps openai.OpenAI)
├── adapters/
│   ├── adapter.py               # Adapter base class + protocols
│   └── openai/openai_adapter.py # Canonical <-> OpenAI Responses API translation
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

**Planned:**

- Anthropic provider and adapter
- Google GenAI provider and adapter
- Token usage and finish details on `Response`
- TestProvider for CI and testing without making live API calls
- built-in utils: agent loop helper, tool schema autowriter
- Token counting


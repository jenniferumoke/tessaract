# Getting started

This guide covers installing Tessaract, configuring the OpenAI provider, sending requests, and keeping conversation history across turns.

## Installation

Tessaract requires **Python 3.11+**. Install the `openai` extra to use the OpenAI provider:

```bash
python -m pip install "tessaract[openai]"
```

If you use uv, add it to your project with `uv add "tessaract[openai]"`.

To work from a checkout of this repository, run `uv sync --extra openai` or `python -m pip install -e ".[openai]"` from the repository root.

The extra installs the OpenAI SDK. Tessaract imports it only when you create an `OpenAIProvider`, so installing the core package alone is possible, but the OpenAI provider will ask you to install the extra before use.

## Configure a provider

Set your OpenAI API key in your shell:

```bash
export OPENAI_API_KEY="your-api-key"
```

```python
import os
from tessaract import OpenAIProvider, Tessaract

client = Tessaract(
    providers={
        "oai": OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
    }
)
```

`Tessaract(providers=...)` takes a dict that maps a **prefix** to a provider. The prefix is your choice. It's what you put before the `/` in a model name:

```python
client.send(model="oai/gpt-5.6-luna", input="Hi")
#                  ^^^ prefix   ^^^^^^^^^^^^^ model sent to OpenAI
```

If the prefix isn't registered, `send()` raises `ValueError`.

### Passing options to the OpenAI client

`OpenAIProvider` builds an `openai.OpenAI` client. The common client options are fields on the provider, and they're passed to the SDK when they're set:

```python
OpenAIProvider(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://my-proxy.example.com/v1",
    timeout=60,
    max_retries=3,
    default_headers={"X-Team": "agents"},
    default_query={"api-version": "2026-01-01"},
)
```

For any other `openai.OpenAI(...)` argument, such as `organization` or `project`, use `provider_args`. Keys in `provider_args` override the typed fields:

```python
OpenAIProvider(provider_args={"organization": "org_...", "project": "proj_..."})
```

### API key

If you don't pass `api_key`, `OpenAIProvider` reads the `OPENAI_API_KEY` environment variable. If neither is set, it raises `ValueError` when you create it, before any request is made:

```python
client = Tessaract(providers={"oai": OpenAIProvider()})   # uses $OPENAI_API_KEY
```

A common pattern is a `.env` file with [`python-dotenv`](https://pypi.org/project/python-dotenv/):

```python
from dotenv import load_dotenv
load_dotenv()
```

## Send a request

```python
response = client.send(model="oai/gpt-5.6-luna", input="Write a haiku about tesseracts.")

print(response.output_text)   # concatenated assistant text
print(response.status)        # "completed", "incomplete", ...
print(response.id)            # provider response id
```

`send()` takes these parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str` | required | `"<prefix>/<model>"` |
| `input` | `str \| list` | required | A prompt string, or the conversation history |
| `stream` | `bool` | `False` | If `True`, returns an iterator of stream events |
| `reasoning` | `ReasoningOptions \| None` | `None` | Reasoning effort and summary settings. See [Reasoning](reasoning.md) |
| `tools` | `list[FunctionTool] \| None` | `None` | Tools the model may call. See [Tool calling](tool-calling.md) |
| `request_options` | `dict \| None` | `None` | Extra native request parameters, passed through to the provider |

### `request_options`: native pass-through

Anything in `request_options` is merged into the OpenAI `responses.create(...)` call. Use it for parameters Tessaract doesn't model yet:

```python
client.send(
    model="oai/gpt-5.6-luna",
    input="Tell me about the sea.",
    request_options={
        "instructions": "Answer like a pirate.",   # system prompt
        "max_output_tokens": 500,
        "temperature": 0.7,
    },
)
```

Canonical parameters (`model`, `input`, `tools`, `reasoning`) always take precedence. If `request_options` contains one of them, the canonical value wins. The same rule applies to keys inside `request_options["extra_body"]`.

## Multi-turn conversations

`input` can be a list that mixes these types:

| Item | Meaning |
| --- | --- |
| `str` | Shorthand for `UserMessage(content=...)` |
| `UserMessage` | A user turn |
| `FunctionToolResult` | The result of a function call the model asked for |
| Any item from `response.output` | The model's earlier output, replayed from its native `raw` form |

So to continue a conversation, you append the model's output to your history:

```python
from tessaract import UserMessage

history = [UserMessage(content="My name is Ada.")]
response = client.send(model="oai/gpt-5.6-luna", input=history)
history.extend(response.output)

history.append(UserMessage(content="What's my name?"))
response = client.send(model="oai/gpt-5.6-luna", input=history)
print(response.output_text)   # "Your name is Ada."
```

Every output item keeps the provider's native object in `.raw`. When you send it back, Tessaract passes the raw object as-is, so the history stays lossless. Reasoning items, message IDs and anything else the provider needs for the next turn are preserved.

> Append **all** of `response.output`, not only the text. Reasoning models perform best, and function calling only works, when earlier reasoning items and function calls are sent back with their results.

## Inspecting output

`response.output` is a list of typed items. You can branch on `item.type`:

```python
for item in response.output:
    match item.type:
        case "assistant_message":
            print("assistant:", "".join(part.text for part in item.content))
        case "reasoning":
            print("reasoning summary:", item.text)
        case "function_call":
            print("call:", item.name, item.arguments)
        case "provider_output":
            print("native item Tessaract doesn't model:", item.provider_type)
```

The full field list is in the [API reference](api-reference.md#output-items).

## Next steps

- [Build an agent with reasoning and tool calling](building-an-agent.md)
- [Stream responses](streaming.md)

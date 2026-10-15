# Getting started

This guide covers installing Tessaract, configuring the OpenAI and Anthropic providers, sending requests, and keeping conversation history across turns.

## Installation

Tessaract requires **Python 3.11+**. Install the extra for each provider you use:

```bash
python -m pip install "tessaract[openai]"             # OpenAI
python -m pip install "tessaract[anthropic]"          # Anthropic
python -m pip install "tessaract[openai,anthropic]"   # both
```

If you use uv, add it to your project with `uv add "tessaract[openai,anthropic]"`.

To work from a checkout of this repository, run `uv sync --extra openai --extra anthropic` or `python -m pip install -e ".[openai,anthropic]"` from the repository root.

Each extra installs that vendor's SDK. Tessaract imports an SDK only when you create its provider, so installing the core package alone is possible, but `OpenAIProvider` and `AnthropicProvider` raise an `ImportError` that names the missing extra.

## Configure a provider

Set the API key for each provider you use in your shell:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
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

### Anthropic

`AnthropicProvider` works the same way. It builds an `anthropic.Anthropic` client from the same typed fields and `provider_args`, and reads `ANTHROPIC_API_KEY` when you don't pass `api_key`:

```python
from tessaract import AnthropicProvider, OpenAIProvider, Tessaract

client = Tessaract(providers={"oai": OpenAIProvider(), "ant": AnthropicProvider()})

client.send(model="ant/claude-opus-5-5", input="Hi")
```

You can register both and switch between them by changing only the model prefix. Requests, history, tools and reasoning use the same Tessaract types on both.

### Passing options to the SDK client

Both providers take the same client options. `OpenAIProvider` builds an `openai.OpenAI` client. The common client options are fields on the provider, and they're passed to the SDK when they're set:

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

`AnthropicProvider` passes the same fields and `provider_args` to `anthropic.Anthropic(...)`:

```python
AnthropicProvider(
    base_url="https://my-proxy.example.com",
    timeout=120,
    max_retries=3,
    default_headers={"X-Team": "agents"},
)
```

### API key

If you don't pass `api_key`, `OpenAIProvider` reads the `OPENAI_API_KEY` environment variable and `AnthropicProvider` reads `ANTHROPIC_API_KEY`. If neither is set, the provider raises `ValueError` when you create it, before any request is made:

```python
client = Tessaract(
    providers={
        "oai": OpenAIProvider(),      # uses $OPENAI_API_KEY
        "ant": AnthropicProvider(),   # uses $ANTHROPIC_API_KEY
    }
)
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
print(response.id)            # provider response id
print(response.usage)         # normalized token counts
```

Every response has a canonical `status` (`"completed"`, `"incomplete"`, ...). When the output stopped early, `incomplete_reason` says why, the same way on both providers:

```python
if response.status == "incomplete":
    print(response.incomplete_reason)   # "max_tokens", "context_window", "content_filter" or "paused"
```

Provider-specific fields live on the response subclass. For example, an `AnthropicResponse` also has the native `stop_reason` (`"end_turn"`, `"tool_use"`, `"max_tokens"`, ...). `response.provider` is `"openai"` or `"anthropic"`; use `isinstance` when you need the subclass's fields:

```python
from tessaract.types.response import AnthropicResponse

if isinstance(response, AnthropicResponse):
    print(response.stop_reason)
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
| `max_tokens` | `int \| None` | `None` | Maximum output tokens. Sent to OpenAI as `max_output_tokens`, and left out when `None`. Anthropic requires a value, so when it's `None` the Anthropic adapter sends a default of **16,000** (`DEFAULT_ANTHROPIC_MAX_TOKENS`). Setting it explicitly is recommended for Anthropic models, because their output limits differ. |
| `instructions` | `str \| None` | `None` | System instructions for this request. Sent to OpenAI as the native `instructions` parameter, and to Anthropic as the first `system` block. |

### Instructions

`instructions=` is the portable way to give a request system instructions:

```python
client.send(model="ant/claude-opus-5-5", input="Tell me about the sea.", instructions="Answer like a pirate.")
```

On OpenAI it's the native `instructions` parameter, so it applies to this request only and isn't carried forward with `previous_response_id`. `request_options={"instructions": ...}` still works on OpenAI. If you set both, `instructions=` wins. Anthropic has no native `instructions` option, so `request_options={"instructions": ...}` raises a `ValueError` on Anthropic that points to `instructions=`. On Anthropic, `instructions=` isn't stored anywhere, so pass it on every request.

You can also put `SystemPrompt` items in `input`. On OpenAI they're system messages in the conversation; on Anthropic they're moved to the top-level `system` parameter after `instructions`. Set `cache=True` on a `SystemPrompt` to cache the prompt up to the end of it; see [Prompt caching](#prompt-caching).

### `request_options`: native pass-through

Anything in `request_options` is merged into the native call (`responses.create(...)` for OpenAI, `messages.create(...)` for Anthropic). Use it for parameters Tessaract doesn't model yet:

```python
client.send(
    model="oai/gpt-5.6-luna",
    input="Tell me about the sea.",
    request_options={
        "temperature": 0.7,
        "prompt_cache_key": "sea-facts",
    },
)
```

For Anthropic, the same mechanism reaches `messages.create(...)` parameters such as `temperature`, `stop_sequences`, `tool_choice` and `metadata`:

```python
client.send(
    model="ant/claude-opus-5-5",
    input="Tell me about the sea.",
    max_tokens=4096,
    request_options={"stop_sequences": ["\n\n"]},
)
```

Canonical parameters (`model`, `input`, `tools`, `reasoning`, `max_tokens`, `instructions`) always take precedence. If `request_options` contains one of them, the canonical value wins. On Anthropic, `messages`, `system`, `thinking` and `output_config.effort` are canonical too, since the adapter builds them from `input`, `instructions` and `reasoning`. Other `output_config` fields, such as a structured output `format`, are kept. The same rule applies to keys inside `request_options["extra_body"]`.

### Prompt caching

Both providers cache a prompt prefix up to a *breakpoint*, so a later request that starts with the same bytes reads it from the cache instead of processing it again. Tessaract configures this in two places: `cache=CacheOptions(...)` on `send()` for the request, and `cache=True` on an input item or tool to put a breakpoint at the end of it.

```python
from tessaract import CacheOptions, SystemPrompt

client.send(
    model=MODEL,  # "oai/gpt-5.6-luna" or "ant/claude-opus-5-5"
    input=[SystemPrompt(content=LONG_DOCS, cache=True), "Summarise section 3."],
    cache=CacheOptions(mode="implicit"),
)
```

#### `CacheOptions`

| Field | OpenAI | Anthropic |
| --- | --- | --- |
| *(no `CacheOptions`)* | Caching stays on: OpenAI caches automatically (on `gpt-5.6`+, with one automatic breakpoint) | No automatic breakpoint; only `cache=True` breakpoints are cached |
| *(any `CacheOptions`)* | Nothing extra | Top-level `cache_control`, which places one automatic breakpoint on the last cacheable block |
| `mode="implicit"` | `prompt_cache_options={"mode": "implicit"}` | Same as passing any `CacheOptions` |
| `mode="explicit"` | `prompt_cache_options={"mode": "explicit"}`: no automatic breakpoint | No top-level `cache_control` |
| `retention="short"` / `"long"` | Ignored | `ttl="5m"` / `"1h"` on the automatic breakpoint and on every breakpoint without its own `ttl` |
| `key` | `prompt_cache_key` | Ignored |

The defaults differ when you pass nothing: OpenAI caches automatically, Anthropic doesn't. Pass `CacheOptions()` to get automatic caching on both.

`"implicit"` suits agents and chats, where each request extends the previous one: the automatic breakpoint follows the end of the conversation, so the whole history is read from the cache on the next turn. `"explicit"` suits requests that share one large prefix and end in one-off content, such as many questions about the same document. There an automatic breakpoint would write the one-off ending to the cache on every request (Anthropic bills cache writes above the normal input price) without it ever being read back. With `"explicit"` and no `cache=True` breakpoints, the request isn't cached at all.

Typed values take precedence over the same keys in `request_options`. On OpenAI, other `prompt_cache_options` fields you pass in `request_options`, such as `ttl`, are kept.

`prompt_cache_options` is only sent when `mode` is set, because OpenAI accepts it on `gpt-5.6` and later only. OpenAI's cache lifetime controls also depend on the model: `prompt_cache_retention` is deprecated, `gpt-5.5` rejects `"in_memory"`, and `prompt_cache_options.ttl` currently accepts only its default `"30m"`. Since no single value works across models, `retention` is not sent to OpenAI; pass model-specific fields in `request_options`.

On Anthropic, entries with a longer TTL must come before entries with a shorter one. That's why `retention` also applies to breakpoints without their own `ttl`: a 5-minute breakpoint on the system prompt followed by a 1-hour automatic breakpoint would be rejected.

#### Breakpoints: `cache=True`

`cache=True` marks the end of an item as a breakpoint:

```python
from tessaract import FunctionTool, FunctionToolResult, SystemPrompt, UserMessage

SystemPrompt(content=LONG_INSTRUCTIONS, cache=True)
UserMessage(content=LONG_DOCUMENT, cache=True)
FunctionToolResult(call_id=call.call_id, result=output, cache=True)
FunctionTool(name="search", description="...", cache=True)  # Anthropic only
```

| Item | OpenAI | Anthropic |
| --- | --- | --- |
| `SystemPrompt`, `UserMessage` | `prompt_cache_breakpoint` on the last content part; string content becomes one `input_text` part | `cache_control` on the last content block; string content becomes one `text` block |
| `FunctionToolResult` | `prompt_cache_breakpoint` on the last output part; string or JSON output becomes one `input_text` part | `cache_control` on the `tool_result` block |
| `FunctionTool` | Ignored. OpenAI has no breakpoints on tool definitions; tools come before the input, so any later breakpoint covers them | `cache_control` on the tool definition. Put it on the last tool to cache all of them |

A breakpoint that's already in the native content (`cache_control` or `prompt_cache_breakpoint`) is kept as it is. Each provider writes at most four breakpoints per request, and on both the automatic breakpoint counts as one of the four. OpenAI documents breakpoints alongside `prompt_cache_options`, so expect them to need `gpt-5.6` or later too.

Prefixes shorter than the provider's minimum aren't cached, and there is no error. Cache reads and writes are reported in `response.usage.cached_input_tokens` and `response.usage.cache_write_input_tokens`.

#### Native breakpoints

You can still write breakpoints in each provider's own format, for example to give one breakpoint its own Anthropic `ttl`:

```python
SystemPrompt(content=[{"type": "text", "text": LONG_INSTRUCTIONS, "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
FunctionTool(name="search", description="...", provider_options={"cache_control": {"type": "ephemeral"}})
FunctionToolResult(call_id=call.call_id, result=output, provider_options={"cache_control": {"type": "ephemeral"}})
```

## Multi-turn conversations

`input` can be a list that mixes these types:

| Item | Meaning |
| --- | --- |
| `str` | Shorthand for `UserMessage(content=...)` |
| `UserMessage` | A user turn |
| `SystemPrompt` | System instructions |
| `FunctionToolResult` | The result of a function call the model asked for |
| Any item from `response.output` | The model's earlier output, replayed from its native `raw` form |

So to continue a conversation, you append the model's output to your history:

```python
from tessaract import SystemPrompt, UserMessage

history = [SystemPrompt(content="Answer concisely."), UserMessage(content="My name is Ada.")]
response = client.send(model="oai/gpt-5.6-luna", input=history)
history.extend(response.output)

history.append(UserMessage(content="What's my name?"))
response = client.send(model="oai/gpt-5.6-luna", input=history)
print(response.output_text)   # "Your name is Ada."
```

Every output item keeps the provider's native object in `.raw`. When you send it back, Tessaract passes the raw object as-is, so the history stays lossless. For Anthropic, the adapter also groups the replayed blocks into the alternating user and assistant turns the Messages API expects. History is tied to the provider that produced it: replay output items only to the same provider. Reasoning items, message IDs and anything else the provider needs for the next turn are preserved.

> Append **all** of `response.output`, not only the text. Reasoning models perform best, and function calling only works, when earlier reasoning items and function calls are sent back with their results.

### A simple chat loop

With `OPENAI_API_KEY` set as above, save this example as `chat.py` and run `python chat.py`. To chat with Claude instead, register `AnthropicProvider()` under `"ant"` and use `model="ant/claude-opus-5-5"`:

```python
from tessaract import OpenAIProvider, Tessaract, UserMessage

client = Tessaract(providers={"oai": OpenAIProvider()})
history = []

print("Chat with Tessaract. Type /exit to quit.")

while True:
    try:
        message = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if message.lower() == "/exit":
        break
    if not message:
        continue

    history.append(UserMessage(content=message))
    response = client.send(model="oai/gpt-6-luna", input=history)
    print(f"Assistant: {response.output_text}")
    history.extend(response.output)
```

The loop sends the full conversation on each turn. It stores every item in `response.output`, so a follow-up question has the context from earlier replies. No tools are involved.

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

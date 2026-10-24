# Tessaract

<img class="tessaract-wordmark" src="assets/images/tessaract-logo.webp" srcset="assets/images/tessaract-logo.webp 600w, assets/images/tessaract-logo-1200.webp 1200w" sizes="(max-width: 632px) calc(100vw - 32px), 600px" alt="Tessaract" width="600" height="200" decoding="async" fetchpriority="high">

Tessaract is a Python SDK for building AI agents across providers. It provides common types for requests, responses, tools, reasoning, and streaming, while provider adapters handle each vendor's API.

OpenAI's Responses API and Anthropic's Messages API are supported today. Other provider adapters are planned.

## Quickstart

Install Tessaract with the extra for each provider you use:

```bash
pip install "tessaract[openai,anthropic]"
```

Set the API key for each provider you use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Then send your first request:

```python
from tessaract import AnthropicProvider, OpenAIProvider, Tessaract

client = Tessaract(providers={"oai": OpenAIProvider(), "ant": AnthropicProvider()})

openai_response = client.send(model="oai/gpt-6-luna", input="Say hello in five words.")

anthropic_response = client.send(model="ant/claude-opus-5-5", input="Say hello in five words.")

print(f'openai response: {openai_response.output_text}')
print(f'anthropic response: {anthropic_response.output_text}')
```


## Start here

1. [Install Tessaract and send your first request](getting-started.md).
2. [Build an agent with reasoning and tool calling](building-an-agent.md).
3. Use the [API reference](api-reference.md) when you need field and method details.

## Guides

- [Tool calling](tool-calling.md): define tools, handle calls, and return results.
- [Server-side tools](server-tools.md): research, data analysis and computer use agents, and workflows that combine providers.
- [Reasoning](reasoning.md): request reasoning and preserve it across turns.
- [Streaming](streaming.md): consume typed streaming events.
- [Architecture](architecture.md): understand the canonical types and provider adapters.

The [source code and examples](https://github.com/jenniferumoke/tessaract) are on GitHub.

# Examples

Runnable scripts from the [`examples/`](https://github.com/jenniferumoke/tessaract/tree/main/examples) directory. Both are walked through step by step in [Building an agent](building-an-agent.md).

Each script reads the model from `TESSARACT_MODEL`. They use OpenAI by default; to run them on Anthropic:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
TESSARACT_MODEL=ant/claude-opus-5-5 uv run --extra anthropic python examples/agent.py
```

## Reasoning and tool-calling agent

[`examples/agent.py`](https://github.com/jenniferumoke/tessaract/blob/main/examples/agent.py) is a synchronous agent that reasons, calls Python tools, and keeps a multi-turn conversation.

```python
--8<-- "examples/agent.py"
```

## Streaming agent

[`examples/streaming_agent.py`](https://github.com/jenniferumoke/tessaract/blob/main/examples/streaming_agent.py) reuses the tools from `agent.py` and streams reasoning summaries and answer text as they're generated.

```python
--8<-- "examples/streaming_agent.py"
```

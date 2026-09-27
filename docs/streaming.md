# Streaming

Pass `stream=True` to `client.send()` to get an iterator of **canonical stream events** instead of a `Response`:

```python
for event in client.send(model="oai/gpt-5.6-luna", input="Tell me a story.", stream=True):
    if event.type == "text.delta":
        print(event.delta, end="", flush=True)
```

Every event has a `type` string that you can branch on, and a `raw_event` field that holds the untouched OpenAI SDK event.

## Event reference

| `type` | Class | Emitted when | Key fields |
| --- | --- | --- | --- |
| `response_started` | `ResponseStartedEvent` | The provider creates the response | `message` |
| `reasoning.started` | `ReasoningStartedEvent` | A reasoning summary part begins | `item_id` |
| `reasoning_summary.delta` | `ReasoningSummaryDeltaEvent` | A chunk of reasoning summary text arrives | `delta`, `item_id`, `output_index` |
| `reasoning_text.delta` | `ReasoningTextDeltaEvent` | A chunk of raw reasoning text arrives, on models that expose it | `delta`, `item_id`, `output_index` |
| `text.delta` | `TextDeltaEvent` | A chunk of assistant text arrives | `delta`, `output_index`, `content_index`, `provider` |
| `tool_call.started` | `ToolCallStartedEvent` | The model starts a function call, before its arguments stream in | `call_id`, `name`, `output_index` |
| `tool_arguments.delta` | `FunctionCallArgumentDeltaEvent` | A chunk of a function call's JSON arguments arrives | `delta`, `output_index` |
| `output_item.done` | `OutputItemCompletedEvent` | An output item is complete | `item` (a fully typed [output item](api-reference.md#output-items)) |
| `response.completed` | `ResponseCompletedEvent` | The response is finished | `response` (a full `Response`) |
| `response.failed` | `ResponseFailedEvent` | The response failed, or the provider sent an error event | `message`, `error` (`ResponseError` with `message` and `code`), `response` (when available) |
| *(native type)* | `CustomProviderEvent` | Any provider event without a canonical equivalent, e.g. `response.in_progress` or `response.output_text.done` | `type` (the native event name), `raw_event` |

> A stream ends with either `response.completed` or `response.failed`. Transport and HTTP errors are still raised as OpenAI SDK exceptions, so wrap the loop in `try`/`except openai.APIError` if you need to handle them.

## Use `response.completed` for the final state

You don't need to rebuild messages or tool calls from deltas. The `response.completed` event carries a full `Response`, with the same typed `output` list as a non-streaming call:

```python
completed = None
for event in stream:
    ...
    if event.type == "response.completed":
        completed = event.response

history.extend(completed.output)
print(completed.output_text)
```

This means a streaming agent loop is the same as the non-streaming one: use the deltas for display, and `completed.output` for state.

## A streaming agent loop

```python
def stream_turn(history, user_text):
    history.append(UserMessage(content=user_text))

    while True:
        completed = None

        for event in client.send(
            model="oai/gpt-5.6-luna",
            input=history,
            tools=tools,
            reasoning=ReasoningOptions(effort="medium", summary="auto"),
            stream=True,
        ):
            match event.type:
                case "reasoning.started":
                    print("\n[thinking] ", end="", flush=True)
                case "reasoning_summary.delta" | "text.delta":
                    print(event.delta, end="", flush=True)
                case "tool_call.started":
                    print(f"\n[tool] calling {event.name}...", flush=True)
                case "output_item.done" if event.item.type == "function_call":
                    print(f"[tool] {event.item.name}({event.item.arguments})")
                case "response.failed":
                    raise RuntimeError(f"Response failed: {event.message}")
                case "response.completed":
                    completed = event.response

        if completed is None:
            raise RuntimeError("Stream ended without a response.completed event")

        history.extend(completed.output)

        calls = [i for i in completed.output if i.type == "function_call"]
        if not calls:
            print()
            return completed

        for call in calls:
            result = FUNCTIONS[call.name](**call.arguments)
            history.append(FunctionToolResult(call_id=call.call_id, result=result))
```

The complete runnable version is [`examples/streaming_agent.py`](https://github.com/jenniferumoke/tessaract/blob/main/examples/streaming_agent.py).

## Notes

- The stream is a generator backed by the OpenAI SDK's `responses.stream(...)` context manager. The HTTP connection closes when you finish iterating, or when the generator is garbage-collected if you `break` early.
- Consume the whole stream before you use `completed`. If you stop early, you don't get a `response.completed` event.
- `CustomProviderEvent` passes through every native event that Tessaract doesn't model. Ignore it, or check `event.type` / `event.raw_event` for provider-specific handling.

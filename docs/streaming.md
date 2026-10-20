# Streaming

Pass `stream=True` to `client.send()` to get an iterator of **canonical stream events** instead of a `Response`:

```python
for event in client.send(model="oai/gpt-5.6-luna", input="Tell me a story.", stream=True):
    if event.type == "text.delta":
        print(event.delta, end="", flush=True)
```

Every event has a `type` string that you can branch on, and a `raw_event` field that holds the untouched provider SDK event. The same events are emitted for OpenAI and Anthropic.

## Event reference

| `type` | Class | Emitted when | Key fields |
| --- | --- | --- | --- |
| `response_started` | `ResponseStartedEvent` | The provider creates the response | `message` |
| `reasoning.started` | `ReasoningStartedEvent` | A reasoning summary part begins | `item_id` (OpenAI), `output_index` (Anthropic) |
| `reasoning_summary.delta` | `ReasoningSummaryDeltaEvent` | A chunk of reasoning summary text arrives | `delta`, `item_id`, `output_index` |
| `reasoning_text.delta` | `ReasoningTextDeltaEvent` | A chunk of raw reasoning text arrives, on models that expose it | `delta`, `item_id`, `output_index` |
| `text.delta` | `TextDeltaEvent` | A chunk of assistant text arrives | `delta`, `output_index`, `content_index`, `provider` |
| `tool_call.started` | `ToolCallStartedEvent` | The model starts a call your code runs (a function, computer or shell call), before its arguments stream in | `call_id`, `name`, `call_type` (the output item type: `"function_call"`, `"computer_call"` or `"shell_call"`), `output_index` |
| `server_tool_call.started` | `ServerToolCallStartedEvent` | The model starts a call to a tool the provider runs, such as web search | `call_id`, `name`, `output_index` |
| `tool_arguments.delta` | `FunctionCallArgumentDeltaEvent` | A chunk of a function call's JSON arguments arrives | `delta`, `output_index` |
| `output_item.done` | `OutputItemCompletedEvent` | An output item is complete | `item` (a fully typed [output item](api-reference.md#output-items)) |
| `response.completed` | `ResponseCompletedEvent` | The response is finished | `response` (a full `Response`) |
| `response.failed` | `ResponseFailedEvent` | The response failed, or the provider sent an error event | `message`, `error` (`ResponseError` with `message` and `code`), `response` (when available) |
| `custom_provider_event` | `CustomProviderEvent` | Any provider event without a canonical equivalent, e.g. `response.in_progress` or `response.output_text.done` | `provider_type` (the native event name), `raw_event` |

> A stream ends with either `response.completed` or `response.failed`. Transport and HTTP errors before the stream starts are still raised as SDK exceptions, so wrap the loop in `try`/`except openai.APIError` (or `anthropic.APIError`) if you need to handle them.

### How Anthropic events map

| Anthropic event | Tessaract event |
| --- | --- |
| `message_start` | `response_started` |
| `content_block_start` for a `thinking` block | `reasoning.started` |
| `content_block_start` for a `tool_use` block | `tool_call.started`; `call_type` is `"computer_call"` for blocks with `toolset_name: "computer"`, and `"shell_call"` for `bash` when the request has a `ShellTool` |
| `content_block_start` for a `server_tool_use` block | `server_tool_call.started` |
| `thinking_delta` | `reasoning_summary.delta` (Anthropic thinking is summarized) |
| `text_delta` | `text.delta` |
| `input_json_delta` | `tool_arguments.delta` for `tool_use` blocks; `CustomProviderEvent` for `server_tool_use` blocks |
| `content_block_stop` | `output_item.done`, with the finished block as an output item. A web search sends one `output_item.done` when its result block stops, holding the whole call |
| `message_stop` | `response.completed` |
| an `error` event mid-stream | `response.failed` |
| `signature_delta`, `citations_delta`, `message_delta`, other `content_block_start`s | `CustomProviderEvent` |

Each Anthropic text block produces its own `output_item.done`. The `Response` in `response.completed` merges neighbouring text blocks into one `AssistantMessage`, like a non-streaming call.

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

- The stream is a generator backed by the OpenAI SDK's `responses.stream(...)` or the Anthropic SDK's `messages.stream(...)` context manager. The HTTP connection closes when you finish iterating, or when the generator is garbage-collected if you `break` early.
- Consume the whole stream before you use `completed`. If you stop early, you don't get a `response.completed` event.
- `CustomProviderEvent` passes through every native event that Tessaract doesn't model. Ignore it, or check `event.provider_type` / `event.raw_event` for provider-specific handling:

  ```python
  if event.type == "custom_provider_event" and event.provider_type == "response.in_progress":
      ...
  ```

- Every event's `type` is a literal, so checking `event.type` (with `if` or `match`) narrows the event to its class in Pylance, Pyright and mypy. After `if event.type == "text.delta":`, `event.delta` type-checks without a cast.

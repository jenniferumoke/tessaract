# Reasoning

Reasoning models think before they answer. Tessaract lets you control that reasoning with `ReasoningOptions` and read it back as `ReasoningOutputItem`s, in both non-streaming and streaming responses.

```python
from tessaract import ReasoningOptions
```

## Requesting reasoning

```python
response = client.send(
    model="oai/gpt-5.6-luna",
    input="How many r's are in 'strawberry'? Explain.",
    reasoning=ReasoningOptions(effort="high", summary="detailed"),
)
```

If you leave out `reasoning`, Tessaract doesn't send a `reasoning` parameter, and the model uses its default.

### `ReasoningOptions`

| Field | Values | Description |
| --- | --- | --- |
| `effort` | `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"extra_high"`, `"max"` | How much the model thinks. Higher values are slower and use more tokens, and handle harder problems better. |
| `summary` | `"auto"`, `"concise"`, `"detailed"`, `"omitted"`, `"updates"` | Asks for a readable summary of the reasoning. `"omitted"` and `"updates"` are Anthropic-only. |
| `mode` | `"standard"`, `"pro"`, `"enabled"`, `"adaptive"`, `"between_tools"`, `"disabled"` | The reasoning mode. `"standard"`/`"pro"` are OpenAI modes; the others are Anthropic thinking types. |
| `budget` | `int` | Anthropic-only: the thinking token budget for `mode="enabled"`. |
| `native_options` | `dict` | Extra native fields merged into the provider's reasoning object (OpenAI `reasoning`, Anthropic `thinking`). The typed fields win. |

Every field is optional. Only the fields you set are sent. A value that a provider can't use raises a `ValueError` instead of being dropped silently. For code that runs on both providers, stick to `effort` and `summary="auto"` or `"concise"`.

### How the fields map to OpenAI

| Canonical | OpenAI `reasoning` parameter |
| --- | --- |
| `effort="extra_high"` | `effort="xhigh"` |
| any other `effort` | sent unchanged |
| `summary` | `summary` |
| `mode` | `mode` |
| `native_options` | merged into `reasoning` |
| `budget`, `summary="omitted"`/`"updates"`, Anthropic modes | `ValueError` |

Support for each effort level and mode depends on the model. The API returns an error if a model doesn't support a value.

### How the fields map to Anthropic

Anthropic splits reasoning across two parameters: `thinking` (whether and how the model thinks) and `output_config.effort` (how much effort it spends).

| Canonical | Anthropic |
| --- | --- |
| `reasoning=None` | neither parameter is sent, so the model default applies |
| `mode="adaptive"` | `thinking={"type": "adaptive"}` |
| `mode="enabled"` + `budget=N` | `thinking={"type": "enabled", "budget_tokens": N}`. `N` must be at least 1024 and less than `max_tokens` |
| `mode="between_tools"` / `"disabled"` | `thinking={"type": ...}`. `summary` and `budget` can't be used with these |
| `mode=None` | `"enabled"` if `budget` is set, otherwise `"adaptive"` |
| `effort="none"` | `thinking={"type": "disabled"}` |
| `effort` `"low"`, `"medium"`, `"high"`, `"max"` | `output_config.effort`, unchanged |
| `effort="extra_high"` | `output_config.effort="xhigh"` |
| `summary` `"auto"`, `"concise"`, `"detailed"` | `thinking.display="summarized"` (Anthropic has one summary level) |
| `summary="omitted"` | `thinking.display="omitted"` |
| `summary="updates"` | `thinking.display="updates"`, sent through the beta Messages API |
| `native_options` | merged into `thinking`, e.g. `{"block_binding": {"prefix_mismatch_behavior": "drop_block"}}` (beta) |
| `effort="minimal"`, `mode="standard"`/`"pro"` | `ValueError` |

`display="updates"` and `block_binding` exist only on Anthropic's beta Messages API. When a request uses either one, the adapter sends it to `client.beta.messages` and adds the matching beta header (`thinking-display-updates-2026-08-18`, `thinking-binding-controls-2026-08-01`) to any `betas` you pass in `request_options`.

An `output_config` in `request_options`, such as one with a structured output `format`, is kept; only its `effort` is replaced by the typed one.

So the portable example above, `ReasoningOptions(effort="high", summary="detailed")`, becomes adaptive thinking with summarized display and `effort="high"` on Anthropic.

### Choosing an effort

| Use case | Suggested effort |
| --- | --- |
| Chat, simple tool routing, extraction | `"none"`, `"minimal"` or `"low"` |
| A general agent that plans multi-step tool use | `"medium"` |
| Hard math, code, or long-horizon planning | `"high"`, `"extra_high"` or `"max"` |

## Reading reasoning output

Reasoning comes back as `ReasoningOutputItem`s in `response.output`, usually before the message or function calls that follow from it:

```python
for item in response.output:
    if item.type == "reasoning":
        print("id:      ", item.id)
        print("summary: ", item.text)     # summary text (requires `summary=`)
        print("content: ", item.content)  # raw reasoning text, if the model exposes it
```

| Field | Description |
| --- | --- |
| `id` | The provider's reasoning item ID |
| `text` | All the reasoning **summary** parts joined together. It's an empty string if you didn't ask for a summary. |
| `content` | All the raw reasoning **content** parts joined together. It's often empty, because many OpenAI models don't expose raw reasoning. |
| `raw` | The native OpenAI `ResponseReasoningItem`, or the Anthropic `ThinkingBlock` |

On Anthropic, `text` is the (summarized) thinking text, `content` is `None`, and `id` is `None`, because thinking blocks have no ID. The block's signature stays in `raw`, which is what's replayed. `redacted_thinking` blocks come back as `ProviderOutputItem`s and are replayed too.

> `response.output_text` includes only assistant message text. Reasoning is never mixed into it.

## Keeping reasoning across turns

Keep the reasoning items in your history. Append **all** of `response.output`:

```python
history.extend(response.output)
```

This matters most in tool-calling agents. Anthropic rejects a tool-use turn whose thinking blocks are missing or modified when thinking is on. When a model reasons, calls a tool and then gets the result, the earlier reasoning item lets it continue its thought instead of starting over. Tessaract replays each item from its native `raw` form, so nothing is lost.

### Stateless / ZDR usage

By default OpenAI stores responses, and reasoning items are looked up by ID. If you turn storage off, ask OpenAI to return encrypted reasoning so it can travel in your history:

```python
client.send(
    model="oai/gpt-5.6-luna",
    input=history,
    reasoning=ReasoningOptions(effort="medium", summary="auto"),
    request_options={
        "store": False,
        "include": ["reasoning.encrypted_content"],
    },
)
```

## Streaming reasoning

With `stream=True`, reasoning arrives as events:

| Event `type` | When | Useful fields |
| --- | --- | --- |
| `reasoning.started` | A new reasoning summary part begins | `item_id` |
| `reasoning_summary.delta` | A chunk of the reasoning summary | `delta`, `item_id`, `output_index` |
| `reasoning_text.delta` | A chunk of raw reasoning text, on models that expose it | `delta`, `item_id`, `output_index` |
| `output_item.done` | The complete reasoning item | `item` (a `ReasoningOutputItem`) |

```python
for event in client.send(model=..., input=..., reasoning=ReasoningOptions(effort="medium", summary="auto"), stream=True):
    if event.type == "reasoning.started":
        print("\n[thinking] ", end="")
    elif event.type == "reasoning_summary.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "text.delta":
        print(event.delta, end="", flush=True)
```

`reasoning.started` fires once per summary part, so a long reasoning summary can start more than once.

On Anthropic, `reasoning.started` fires when a thinking block starts (with `output_index` set and `item_id=None`), and the thinking text arrives as `reasoning_summary.delta` events.

See [Streaming](streaming.md) for the full event list.

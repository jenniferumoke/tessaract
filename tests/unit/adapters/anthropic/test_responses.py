import pytest

pytest.importorskip("anthropic")

from tessaract import FunctionTool, FunctionToolResult, ReasoningOptions
from tessaract.types.output_types import (
    AssistantMessage,
    FunctionCallOutputItem,
    ProviderOutputItem,
    ReasoningOutputItem,
)
from tessaract.types.response import AnthropicResponse, StopReason


def _message(content, stop_reason="end_turn", usage=None):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-20260101",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }

THINKING = {"type": "thinking", "thinking": "Paris first.", "signature": "sig_1"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}


def test_sync_response_is_normalized(anthropic_client, anthropic_api):
    anthropic_api.message = _message(
        [THINKING, TOOL_USE],
        stop_reason="tool_use",
        usage={
            "input_tokens": 10, "output_tokens": 50,
            "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20,
            "output_tokens_details": {"thinking_tokens": 30},
        },
    )

    response = anthropic_client.send(model="ant/claude-test", input="Weather in Paris?")

    assert isinstance(response, AnthropicResponse)
    assert response.provider == "anthropic"
    assert response.model == "claude-test-20260101"
    assert response.stop_reason is StopReason.TOOL_USE

    reasoning, call = response.output
    assert isinstance(reasoning, ReasoningOutputItem)
    assert reasoning.id is None
    assert reasoning.text == "Paris first."
    assert reasoning.raw.signature == "sig_1"

    assert isinstance(call, FunctionCallOutputItem)
    assert (call.call_id, call.name, call.arguments) == ("toolu_1", "get_weather", {"city": "Paris"})

    usage = response.usage
    assert usage.input_tokens == 130
    assert usage.output_tokens == 50
    assert usage.total_tokens == 180
    assert usage.cached_input_tokens == 100
    assert usage.cache_write_input_tokens == 20
    assert usage.reasoning_tokens == 30
    assert response.provider_usage.input_tokens == 10

def test_neighbouring_text_blocks_become_one_message(anthropic_client, anthropic_api):
    anthropic_api.message = _message([
        {"type": "text", "text": "Paris is "},
        {"type": "text", "text": "19C", "citations": [{
            "type": "web_search_result_location", "cited_text": "19C",
            "encrypted_index": "e", "title": "Weather", "url": "https://example.com",
        }]},
        {"type": "text", "text": "."},
    ])

    response = anthropic_client.send(model="ant/claude-test", input="Weather?")

    [message] = response.output
    assert isinstance(message, AssistantMessage)
    assert [part.text for part in message.content] == ["Paris is ", "19C", "."]
    assert len(message.raw) == 3
    assert message.content[1].annotations[0].source == "https://example.com"
    assert response.output_text == "Paris is 19C."

def test_unknown_and_redacted_blocks_pass_through(anthropic_client, anthropic_api):
    anthropic_api.message = _message([
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "x"}},
    ])

    response = anthropic_client.send(model="ant/claude-test", input="hi")

    assert [type(item) for item in response.output] == [ProviderOutputItem, ProviderOutputItem]
    assert [item.provider_type for item in response.output] == ["redacted_thinking", "server_tool_use"]

def test_unknown_stop_reason_is_kept_as_a_string(anthropic_adapter):
    assert anthropic_adapter._normalize_stop_reason("something_new") == "something_new"

def test_tool_loop_round_trip_sends_valid_turns(anthropic_client, anthropic_api):
    anthropic_api.message = _message([THINKING, TOOL_USE], stop_reason="tool_use")
    tools = [FunctionTool(name="get_weather", description="Weather", strict=False)]
    reasoning = ReasoningOptions(budget=2048, summary="auto")

    history: list = ["Weather in Paris?"]
    response = anthropic_client.send(model="ant/claude-test", input=history, tools=tools, reasoning=reasoning)
    history.extend(response.output)
    history.append(FunctionToolResult(call_id=response.output[1].call_id, result="19C"))

    anthropic_api.message = _message([{"type": "text", "text": "19C in Paris."}])
    response = anthropic_client.send(
        model="ant/claude-test", input=history, tools=tools, reasoning=reasoning,
        instructions="Be brief.", max_tokens=4096,
    )

    body = anthropic_api.last_body
    assert body["system"] == [{"type": "text", "text": "Be brief."}]
    assert body["max_tokens"] == 4096
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048, "display": "summarized"}
    assert [turn["role"] for turn in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"] == [THINKING, TOOL_USE]
    assert body["messages"][2]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "19C"}]
    assert response.output_text == "19C in Paris."

def test_beta_features_are_sent_to_the_beta_api(anthropic_client, anthropic_api):
    anthropic_api.message = _message([{"type": "text", "text": "ok"}])

    anthropic_client.send(
        model="ant/claude-test", input="hi",
        reasoning=ReasoningOptions(mode="adaptive", summary="updates"),
    )

    request = anthropic_api.requests[-1]
    assert "beta=true" in str(request.url)
    assert "thinking-display-updates-2026-08-18" in request.headers["anthropic-beta"]
    assert anthropic_api.last_body["thinking"] == {"type": "adaptive", "display": "updates"}


# --- streaming ----------------------------------------------------------------

def _stream_events():
    start = _message([], stop_reason=None, usage={"input_tokens": 10, "output_tokens": 1})
    return [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Paris first."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig_1"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Checking."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 40}},
        {"type": "message_stop"},
    ]

def test_stream_events_are_normalized(anthropic_client, anthropic_api):
    anthropic_api.sse_events = _stream_events()

    events = list(anthropic_client.send(model="ant/claude-test", input="Weather?", stream=True))
    types = [event.type for event in events]

    assert types == [
        "response_started",
        "reasoning.started",
        "reasoning_summary.delta",
        "content_block_delta",  # signature_delta, as a CustomProviderEvent
        "output_item.done",
        "content_block_start",  # text block start, as a CustomProviderEvent
        "text.delta",
        "output_item.done",
        "tool_call.started",
        "tool_arguments.delta",
        "tool_arguments.delta",
        "output_item.done",
        "message_delta",
        "response.completed",
    ]
    assert anthropic_api.last_body["stream"] is True

    by_type = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event)

    assert by_type["reasoning_summary.delta"][0].delta == "Paris first."
    assert by_type["text.delta"][0].delta == "Checking."
    assert by_type["text.delta"][0].provider == "anthropic"
    assert by_type["tool_call.started"][0].call_id == "toolu_1"

    reasoning, text, call = (event.item for event in by_type["output_item.done"])
    assert reasoning.raw.signature == "sig_1"
    assert text.content[0].text == "Checking."
    assert call.arguments == {"city": "Paris"}

    [completed] = by_type["response.completed"]
    response = completed.response
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.usage.output_tokens == 40
    assert [item.type for item in response.output] == ["reasoning", "assistant_message", "function_call"]

def test_stream_error_event_becomes_response_failed(anthropic_client, anthropic_api):
    anthropic_api.sse_events = [
        _stream_events()[0],
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]

    events = list(anthropic_client.send(model="ant/claude-test", input="hi", stream=True))

    assert [event.type for event in events] == ["response_started", "response.failed"]
    assert events[-1].error.code == "overloaded_error"
    assert "Overloaded" in events[-1].message

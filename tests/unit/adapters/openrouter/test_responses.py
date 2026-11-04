import pytest

pytest.importorskip("openrouter")

from tessaract import FunctionToolResult
from tessaract.types.output_types import (
    AssistantMessage,
    FunctionCallOutputItem,
    ProviderOutputItem,
    ReasoningOutputItem,
)
from tessaract.types.response import IncompleteReason, OpenRouterResponse, ResponseStatus
from tests.fixtures import openrouter as payloads

MODEL = "or/openai/gpt-5-mini"


def _send(client, input="What's the temperature in Paris?"):
    return client.send(model=MODEL, input=input)


def test_tool_call_response_is_normalized(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.response([payloads.OPENAI_REASONING, payloads.WEATHER_CALL])]

    response = _send(openrouter_client)

    assert isinstance(response, OpenRouterResponse)
    assert response.provider == "openrouter"
    assert response.id == "gen-1"
    assert response.model == "openai/gpt-5-mini"
    assert response.status is ResponseStatus.COMPLETED

    reasoning, call = response.output
    assert isinstance(reasoning, ReasoningOutputItem)
    assert reasoning.id == "rs_openai"
    assert reasoning.text == "Need the weather tool."
    assert reasoning.content == ""

    assert isinstance(call, FunctionCallOutputItem)
    assert (call.call_id, call.name, call.arguments) == ("call_1", "get_weather", {"city": "Paris"})


def test_usage_and_cost(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.response([payloads.message("19C")])]

    response = _send(openrouter_client)

    assert response.usage is not None
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 40
    assert response.usage.total_tokens == 160
    assert response.usage.cached_input_tokens == 20
    assert response.usage.cache_write_input_tokens is None
    assert response.usage.reasoning_tokens == 25
    assert response.cost == 0.00042
    assert response.output_text == "19C"


def test_a_response_without_usage(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.response([payloads.message("hi")], usage=None)]

    response = _send(openrouter_client)

    assert response.usage is None
    assert response.cost is None


def test_text_annotations(openrouter_client, openrouter_api):
    citation = {"type": "url_citation", "url": "https://example.com", "title": "Example", "start_index": 0, "end_index": 3}
    openrouter_api.responses = [payloads.response([payloads.message("19C", [citation])])]

    [message] = _send(openrouter_client).output

    assert isinstance(message, AssistantMessage)
    [part] = message.content  # type: ignore[misc]
    [annotation] = part.annotations
    assert annotation.provider == "openrouter"
    assert annotation.type == "citation"
    assert annotation.provider_type == "url_citation"
    assert annotation.source == "https://example.com"
    assert annotation.title == "Example"


@pytest.mark.parametrize(
    ("status", "incomplete_details", "expected_status", "expected_reason"),
    [
        ("incomplete", {"reason": "max_output_tokens"}, ResponseStatus.INCOMPLETE, IncompleteReason.MAX_TOKENS),
        ("incomplete", {"reason": "content_filter"}, ResponseStatus.INCOMPLETE, IncompleteReason.CONTENT_FILTER),
        ("incomplete", None, ResponseStatus.INCOMPLETE, None),
        ("in_progress", None, ResponseStatus.IN_PROGRESS, None),
    ],
)
def test_status_and_incomplete_reason(openrouter_client, openrouter_api, status, incomplete_details, expected_status, expected_reason):
    openrouter_api.responses = [
        payloads.response([payloads.message("partial")], status=status, incomplete_details=incomplete_details)
    ]

    response = _send(openrouter_client)

    assert response.status is expected_status
    assert response.incomplete_reason is expected_reason


def test_failed_response_carries_its_error(openrouter_client, openrouter_api):
    openrouter_api.responses = [
        payloads.response([], status="failed", error={"code": "server_error", "message": "Upstream failed"})
    ]

    response = _send(openrouter_client)

    assert response.status is ResponseStatus.FAILED
    assert response.error is not None
    assert (response.error.message, response.error.code) == ("Upstream failed", "server_error")


def test_refusals_and_unknown_items_stay_native(openrouter_client, openrouter_api):
    refusal = {
        "type": "message", "id": "msg_2", "role": "assistant", "status": "completed",
        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
    }
    unknown = {"type": "openrouter:something_new", "id": "x_1", "detail": 1}
    openrouter_api.responses = [payloads.response([refusal, unknown])]

    refused, new = _send(openrouter_client).output

    assert isinstance(refused, ProviderOutputItem)
    assert refused.provider_type == "message"
    assert isinstance(new, ProviderOutputItem)
    assert new.provider_type == "openrouter:something_new"


@pytest.mark.parametrize(
    "reasoning",
    [payloads.OPENAI_REASONING, payloads.ANTHROPIC_REASONING, payloads.OPEN_MODEL_REASONING],
    ids=["openai_encrypted", "anthropic_signed", "open_model_plain"],
)
def test_reasoning_and_tool_calls_replay_unchanged(openrouter_client, openrouter_api, reasoning):
    '''A tool round trip: the second request carries the reasoning and the call exactly as they came back.'''
    openrouter_api.responses = [
        payloads.response([reasoning, payloads.WEATHER_CALL]),
        payloads.response([payloads.message("It's 19C in Paris.")]),
    ]
    history: list = ["What's the temperature in Paris?"]

    first = _send(openrouter_client, history)
    history.extend(first.output)
    history.extend(
        FunctionToolResult(call_id=item.call_id, result={"temperature": "19C"})
        for item in first.output if isinstance(item, FunctionCallOutputItem)
    )
    second = _send(openrouter_client, history)

    assert second.output_text == "It's 19C in Paris."
    assert openrouter_api.last_body["input"] == [
        {"role": "user", "content": "What's the temperature in Paris?"},
        reasoning,
        payloads.WEATHER_CALL,
        {"type": "function_call_output", "call_id": "call_1", "output": '{"temperature": "19C"}'},
    ]


def test_anthropic_reasoning_text_is_normalized(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.response([payloads.ANTHROPIC_REASONING, payloads.WEATHER_CALL])]

    reasoning = _send(openrouter_client).output[0]

    assert isinstance(reasoning, ReasoningOutputItem)
    assert reasoning.content == "The user wants the weather in Paris."
    assert reasoning.text == ""


def test_assistant_messages_replay_unchanged(openrouter_client, openrouter_api):
    message = payloads.message("Hello!")
    openrouter_api.responses = [payloads.response([message])]

    first = _send(openrouter_client, ["hi"])
    _send(openrouter_client, ["hi", *first.output, "how are you?"])

    assert openrouter_api.last_body["input"] == [
        {"role": "user", "content": "hi"},
        message,
        {"role": "user", "content": "how are you?"},
    ]


def _stream(client, input="What's the temperature in Paris?"):
    return list(client.send(model=MODEL, input=input, stream=True))


def _done_items(*items):
    return [
        event
        for index, item in enumerate(items)
        for event in (
            {"type": "response.output_item.added", "output_index": index, "item": item},
            {"type": "response.output_item.done", "output_index": index, "item": item},
        )
    ]


def test_stream_of_text(openrouter_client, openrouter_api):
    message = payloads.message("Hello!")
    final = payloads.response([message])
    openrouter_api.responses = [payloads.stream(
        {"type": "response.created", "response": payloads.response([], status="in_progress", usage=None)},
        {"type": "response.output_item.added", "output_index": 0, "item": {**message, "content": []}},
        *[
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "item_id": "msg_1", "delta": delta, "logprobs": []}
            for delta in ("Hel", "lo!")
        ],
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": final},
    )]

    events = _stream(openrouter_client)

    assert openrouter_api.last_body["stream"] is True
    assert [event.type for event in events] == [
        "response_started", "custom_provider_event", "text.delta", "text.delta", "output_item.done", "response.completed",
    ]
    assert events[1].provider_type == "response.output_item.added"
    assert events[2].provider == "openrouter"
    assert events[2].item_id == "msg_1"
    assert "".join(event.delta for event in events if event.type == "text.delta") == "Hello!"
    assert isinstance(events[4].item, AssistantMessage)

    response = events[-1].response
    assert isinstance(response, OpenRouterResponse)
    assert response.output_text == "Hello!"
    assert response.cost == 0.00042


@pytest.mark.parametrize(
    ("delta_type", "canonical_type"),
    [
        ("response.reasoning_summary_text.delta", "reasoning_summary.delta"),
        ("response.reasoning_text.delta", "reasoning_text.delta"),
    ],
)
def test_stream_of_reasoning(openrouter_client, openrouter_api, delta_type, canonical_type):
    delta = {"type": delta_type, "output_index": 0, "item_id": "rs_open", "delta": "Hmm"}
    delta.update({"summary_index": 0} if "summary" in delta_type else {"content_index": 0})
    openrouter_api.responses = [payloads.stream(
        {"type": "response.output_item.added", "output_index": 0, "item": {**payloads.OPEN_MODEL_REASONING, "content": []}},
        delta,
        {"type": "response.output_item.done", "output_index": 0, "item": payloads.OPEN_MODEL_REASONING},
    )]

    started, text, done = _stream(openrouter_client)

    assert (started.type, started.item_id, started.output_index) == ("reasoning.started", "rs_open", 0)
    assert (text.type, text.delta, text.item_id) == (canonical_type, "Hmm", "rs_open")
    assert isinstance(done.item, ReasoningOutputItem)
    assert done.item.content == "I should call get_weather."


def test_stream_of_a_tool_call(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.stream(
        {"type": "response.output_item.added", "output_index": 0, "item": {**payloads.WEATHER_CALL, "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_1", "delta": '{"city":"Paris"}'},
        {"type": "response.output_item.done", "output_index": 0, "item": payloads.WEATHER_CALL},
    )]

    started, arguments, done = _stream(openrouter_client)

    assert (started.type, started.call_id, started.name) == ("tool_call.started", "call_1", "get_weather")
    assert (arguments.type, arguments.delta, arguments.item_id) == ("tool_arguments.delta", '{"city":"Paris"}', "fc_1")
    assert isinstance(done.item, FunctionCallOutputItem)
    assert done.item.arguments == {"city": "Paris"}


@pytest.mark.parametrize(
    ("item", "name"),
    [
        ({"type": "web_search_call", "id": "ws_1", "status": "in_progress"}, "web_search"),
        ({"type": "openrouter:datetime", "id": "dt_1", "status": "in_progress", "datetime": "2026-01-01T00:00:00Z", "timezone": "UTC"}, "datetime"),
    ],
    ids=["hosted", "openrouter"],
)
def test_stream_of_a_server_tool_call(openrouter_client, openrouter_api, item, name):
    openrouter_api.responses = [payloads.stream(
        {"type": "response.output_item.added", "output_index": 0, "item": item},
    )]

    [started] = _stream(openrouter_client)

    assert (started.type, started.call_id, started.name) == ("server_tool_call.started", item["id"], name)


def test_stream_output_replays_like_a_response(openrouter_client, openrouter_api):
    reasoning = payloads.ANTHROPIC_REASONING
    openrouter_api.responses = [
        payloads.stream(*_done_items(reasoning, payloads.WEATHER_CALL), {"type": "response.completed", "response": payloads.response([reasoning, payloads.WEATHER_CALL])}),
    ]
    history: list = ["What's the temperature in Paris?"]

    completed = _stream(openrouter_client, history)[-1]
    history.extend(completed.response.output)
    _stream(openrouter_client, history)

    assert openrouter_api.last_body["input"] == [
        {"role": "user", "content": "What's the temperature in Paris?"},
        reasoning,
        payloads.WEATHER_CALL,
    ]


def test_incomplete_stream_ends_as_completed(openrouter_client, openrouter_api):
    response = payloads.response(
        [payloads.message("Hel")], status="incomplete", incomplete_details={"reason": "max_output_tokens"},
    )
    openrouter_api.responses = [payloads.stream({"type": "response.incomplete", "response": response})]

    [event] = _stream(openrouter_client)

    assert event.type == "response.completed"
    assert event.response.status is ResponseStatus.INCOMPLETE
    assert event.response.incomplete_reason is IncompleteReason.MAX_TOKENS


def test_failed_stream_carries_its_error(openrouter_client, openrouter_api):
    response = payloads.response([], status="failed", error={"code": "server_error", "message": "Upstream failed"})
    openrouter_api.responses = [payloads.stream({"type": "response.failed", "response": response})]

    [event] = _stream(openrouter_client)

    assert event.type == "response.failed"
    assert event.message == "Upstream failed"
    assert event.error.code == "server_error"
    assert event.response.status is ResponseStatus.FAILED


def test_error_event_fails_the_stream(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.stream(
        {"type": "error", "code": "rate_limit_exceeded", "message": "Slow down", "param": None},
    )]

    [event] = _stream(openrouter_client)

    assert event.type == "response.failed"
    assert (event.message, event.error.code) == ("Slow down", "rate_limit_exceeded")
    assert event.response is None


@pytest.mark.parametrize(
    ("event", "message", "code"),
    [
        ({"type": "error", "message": "Upstream failed", "code": 502}, "Upstream failed", "502"),
        ({"type": "error", "error": {"message": "Nested", "code": "bad"}}, "Nested", "bad"),
        ({"type": "response.failed", "response": {"error": {"message": "Partial"}}}, "Partial", None),
        ({"type": "response.completed", "response": {}}, "The OpenRouter SDK couldn't read a response.completed event.", None),
    ],
    ids=["error", "nested_error", "failed", "completed"],
)
def test_unreadable_terminal_events_fail_the_stream(openrouter_client, openrouter_api, event, message, code):
    openrouter_api.responses = [[event]]

    [failed] = _stream(openrouter_client)

    assert failed.type == "response.failed"
    assert (failed.message, failed.error.code) == (message, code)


def test_unknown_events_and_items_pass_through(openrouter_client, openrouter_api):
    openrouter_api.responses = [payloads.stream(
        {"type": "response.in_progress", "response": payloads.response([], status="in_progress", usage=None)},
        {"type": "response.brand_new", "payload": 1},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "brand_new_item"}},
        {"type": "response.output_item.done", "output_index": 0, "item": {"type": "brand_new_item"}},
    )]

    in_progress, brand_new, added, done = _stream(openrouter_client)

    assert in_progress.provider_type == "response.in_progress"
    assert brand_new.provider_type == "response.brand_new"
    assert brand_new.raw_event.raw["payload"] == 1
    assert added.provider_type == "response.output_item.added"
    assert isinstance(done.item, ProviderOutputItem)
    assert done.item.provider_type == "brand_new_item"


def test_stopping_early_closes_the_connection(openrouter_client, openrouter_api, monkeypatch):
    openrouter_api.responses = [payloads.stream(
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "item_id": "msg_1", "delta": "Hi", "logprobs": []},
        {"type": "response.completed", "response": payloads.response([payloads.message("Hi")])},
    )]
    from openrouter.utils.eventstreaming import EventStream

    closed = []
    close = EventStream.close
    monkeypatch.setattr(EventStream, "close", lambda self: (closed.append(True), close(self))[1])

    stream = openrouter_client.send(model=MODEL, input="hi", stream=True)
    next(stream)
    stream.close()

    assert closed == [True]

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

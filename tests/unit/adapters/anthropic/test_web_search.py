import pytest

pytest.importorskip("anthropic")

from tessaract import FunctionTool, ProviderTool, UserLocation, WebSearchTool
from tessaract.adapters.anthropic.anthropic_adapter import DEFAULT_WEB_SEARCH_TOOL_TYPE
from tessaract.types.output_types import AssistantMessage, WebSearchCallOutputItem
from tessaract.types.request import Request
from tessaract.types.response import IncompleteReason, ResponseStatus


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

SEARCH_CALL = {
    "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
    "input": {"query": "python 3.15 release"},
}
SEARCH_RESULT = {
    "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
    "content": [
        {"type": "web_search_result", "url": "https://docs.python.org/3.15/whatsnew", "title": "What's New",
         "encrypted_content": "enc_1", "page_age": "2 days ago"},
        {"type": "web_search_result", "url": "https://peps.python.org/pep-0790", "title": "PEP 790",
         "encrypted_content": "enc_2"},
    ],
}
CITED_TEXT = {
    "type": "text", "text": "Python 3.15 is out.",
    "citations": [{
        "type": "web_search_result_location", "cited_text": "Python 3.15",
        "encrypted_index": "ei_1", "title": "What's New", "url": "https://docs.python.org/3.15/whatsnew",
    }],
}


# request mapping

def test_web_search_tool_maps_to_native_web_search(anthropic_adapter):
    tool = WebSearchTool(
        max_uses=3,
        blocked_domains=["spam.com"],
        user_location=UserLocation(country="NL", timezone="Europe/Amsterdam"),
    )

    assert anthropic_adapter.map_web_search_tool(tool) == {
        "type": DEFAULT_WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        "max_uses": 3,
        "blocked_domains": ["spam.com"],
        "user_location": {"type": "approximate", "country": "NL", "timezone": "Europe/Amsterdam"},
    }

def test_bare_web_search_tool(anthropic_adapter):
    assert anthropic_adapter.map_web_search_tool(WebSearchTool()) == {
        "type": "web_search_20250305", "name": "web_search",
    }

def test_allowed_domains_are_sent(anthropic_adapter):
    native = anthropic_adapter.map_web_search_tool(WebSearchTool(allowed_domains=["python.org"]))

    assert native["allowed_domains"] == ["python.org"]

def test_provider_options_add_native_fields_and_pin_the_version(anthropic_adapter):
    tool = WebSearchTool(
        max_uses=2,
        provider_options={"type": "web_search_20260209", "max_uses": 99, "name": "other", "allowed_callers": ["direct"]},
    )

    assert anthropic_adapter.map_web_search_tool(tool) == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 2,
        "allowed_callers": ["direct"],
    }

def test_cached_web_search_tool_gets_a_breakpoint(anthropic_adapter):
    native = anthropic_adapter.map_web_search_tool(WebSearchTool(cache=True))

    assert native["cache_control"] == {"type": "ephemeral"}

def test_native_cache_control_is_kept(anthropic_adapter):
    native = anthropic_adapter.map_web_search_tool(
        WebSearchTool(cache=True, provider_options={"cache_control": {"type": "ephemeral", "ttl": "1h"}})
    )

    assert native["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

def test_tools_keep_their_order(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(Request(
        model="claude-test", input=[{"role": "user", "content": "hi"}],
        tools=[
            WebSearchTool(),
            FunctionTool(name="get_weather", description="Weather"),
            ProviderTool(provider="anthropic", definition={"type": "web_fetch_20250910", "name": "web_fetch"}),
        ],
    ))

    assert [tool["name"] for tool in kwargs["tools"]] == ["web_search", "get_weather", "web_fetch"]

def test_provider_tool_is_sent_unchanged(anthropic_adapter):
    definition = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3}

    assert anthropic_adapter.map_tools([ProviderTool(provider="anthropic", definition=definition)]) == [definition]

def test_provider_tool_for_another_provider_raises(anthropic_adapter):
    with pytest.raises(ValueError, match="'openai'"):
        anthropic_adapter.map_tools([ProviderTool(provider="openai", definition={"type": "file_search"})])

def test_web_search_tool_is_sent_in_the_request(anthropic_client, anthropic_api):
    anthropic_api.message = _message([{"type": "text", "text": "ok"}])

    anthropic_client.send(model="ant/claude-test", input="hi", tools=[WebSearchTool(max_uses=1)])

    assert anthropic_api.last_body["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}]


# responses

def test_search_call_and_result_become_one_item(anthropic_client, anthropic_api):
    anthropic_api.message = _message(
        [SEARCH_CALL, SEARCH_RESULT, CITED_TEXT],
        usage={"input_tokens": 10, "output_tokens": 5, "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0}},
    )

    response = anthropic_client.send(model="ant/claude-test", input="What's new in 3.15?", tools=[WebSearchTool()])

    search, message = response.output
    assert isinstance(search, WebSearchCallOutputItem)
    assert search.provider == "anthropic"
    assert (search.call_id, search.status, search.action) == ("srvtoolu_1", "completed", "search")
    assert search.queries == ["python 3.15 release"]
    assert [(s.url, s.title, s.page_age) for s in search.sources] == [
        ("https://docs.python.org/3.15/whatsnew", "What's New", "2 days ago"),
        ("https://peps.python.org/pep-0790", "PEP 790", None),
    ]
    assert search.error is None
    assert [block.type for block in search.raw] == ["server_tool_use", "web_search_tool_result"]

    assert isinstance(message, AssistantMessage)
    assert message.content[0].annotations[0].source == "https://docs.python.org/3.15/whatsnew"

    assert response.usage.server_tool_requests == {"web_search": 1}

def test_search_error_result_fails_the_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([
        SEARCH_CALL,
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
         "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
    ])

    [search] = anthropic_client.send(model="ant/claude-test", input="hi").output

    assert (search.status, search.error, search.sources) == ("failed", "max_uses_exceeded", [])

def test_paused_turn_keeps_the_unfinished_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([SEARCH_CALL], stop_reason="pause_turn")

    response = anthropic_client.send(model="ant/claude-test", input="hi")

    [search] = response.output
    assert (search.status, search.queries, search.sources) == ("in_progress", ["python 3.15 release"], [])
    assert [block.type for block in search.raw] == ["server_tool_use"]
    assert response.incomplete_reason is IncompleteReason.PAUSED

def test_parallel_searches_are_not_reordered(anthropic_client, anthropic_api):
    second_call = {**SEARCH_CALL, "id": "srvtoolu_2", "input": {"query": "pep 790"}}
    second_result = {**SEARCH_RESULT, "tool_use_id": "srvtoolu_2"}
    anthropic_api.message = _message([SEARCH_CALL, second_call, SEARCH_RESULT, second_result])

    output = anthropic_client.send(model="ant/claude-test", input="hi").output

    # only neighbouring blocks are merged, so replaying the output keeps the native block order
    assert [(item.call_id, item.status, item.queries) for item in output] == [
        ("srvtoolu_1", "in_progress", ["python 3.15 release"]),
        ("srvtoolu_2", "in_progress", ["pep 790"]),
        ("srvtoolu_1", "completed", ["python 3.15 release"]),
        ("srvtoolu_2", "completed", ["pep 790"]),
    ]
    assert [[block.type for block in item.raw] for item in output] == [
        ["server_tool_use"], ["server_tool_use"], ["web_search_tool_result"], ["web_search_tool_result"],
    ]

def test_no_server_tool_use_leaves_server_tool_requests_unset(anthropic_client, anthropic_api):
    anthropic_api.message = _message([{"type": "text", "text": "hi"}])

    assert anthropic_client.send(model="ant/claude-test", input="hi").usage.server_tool_requests is None

def test_search_is_replayed_unchanged_in_the_next_request(anthropic_client, anthropic_api):
    anthropic_api.message = _message([SEARCH_CALL, SEARCH_RESULT, CITED_TEXT])
    history: list = ["What's new in 3.15?"]

    response = anthropic_client.send(model="ant/claude-test", input=history, tools=[WebSearchTool()])
    history.extend(response.output)
    history.append("Thanks")

    anthropic_api.message = _message([{"type": "text", "text": "You're welcome."}])
    anthropic_client.send(model="ant/claude-test", input=history, tools=[WebSearchTool()])

    _, assistant, follow_up = anthropic_api.last_body["messages"]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["server_tool_use", "web_search_tool_result", "text"]
    assert assistant["content"][0] == SEARCH_CALL
    assert assistant["content"][1] == SEARCH_RESULT
    assert follow_up == {"role": "user", "content": [{"type": "text", "text": "Thanks"}]}


# streaming

def _stream_events():
    return [
        {"type": "message_start", "message": _message([], stop_reason=None)},
        {"type": "content_block_start", "index": 0, "content_block": {**SEARCH_CALL, "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"query\": "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"python 3.15 release\"}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": SEARCH_RESULT},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Python 3.15 is out."}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 20, "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0}}},
        {"type": "message_stop"},
    ]

def test_stream_events_are_normalized(anthropic_client, anthropic_api):
    anthropic_api.sse_events = _stream_events()

    events = list(anthropic_client.send(model="ant/claude-test", input="hi", tools=[WebSearchTool()], stream=True))
    types = [
        event.provider_type if event.type == "custom_provider_event" else event.type
        for event in events
    ]

    assert types == [
        "response_started",
        "server_tool_call.started",
        "content_block_delta",  # server tool input, never a tool_arguments.delta
        "content_block_delta",
        "content_block_start",  # result block start, as a CustomProviderEvent
        "output_item.done",     # the search call, sent when its result arrives
        "content_block_start",  # text block start, as a CustomProviderEvent
        "text.delta",
        "output_item.done",
        "message_delta",
        "response.completed",
    ]

    started = events[1]
    assert (started.call_id, started.name, started.output_index) == ("srvtoolu_1", "web_search", 0)

    search = events[5].item
    assert isinstance(search, WebSearchCallOutputItem)
    assert search.queries == ["python 3.15 release"]
    assert search.status == "completed"
    assert len(search.sources) == 2

    response = events[-1].response
    assert response.status is ResponseStatus.COMPLETED
    assert [item.type for item in response.output] == ["web_search_call", "assistant_message"]
    assert response.output[0].queries == ["python 3.15 release"]
    assert response.usage.server_tool_requests == {"web_search": 1}

def test_function_tool_deltas_are_unaffected_by_server_tools(anthropic_client, anthropic_api):
    events = _stream_events()[:7] + [
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]
    anthropic_api.sse_events = events

    stream = list(anthropic_client.send(model="ant/claude-test", input="hi", stream=True))

    deltas = [event for event in stream if event.type == "tool_arguments.delta"]
    assert [(event.output_index, event.delta) for event in deltas] == [(2, "{}")]

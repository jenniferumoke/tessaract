from types import SimpleNamespace

import pytest
from openai.types.responses import Response as OpenAINativeResponse
from openai.types.responses import ResponseFunctionWebSearch

from tessaract import (
    FunctionTool,
    ProviderTool,
    UnsupportedToolOptionError,
    UserLocation,
    WebSearchTool,
)
from tessaract.types.output_types import WebSearchCallOutputItem
from tessaract.types.request import Request


def _web_search_call(action, status="completed", id="ws_1"):
    return ResponseFunctionWebSearch.model_validate(
        {"id": id, "type": "web_search_call", "status": status, "action": action}
    )

def _response(output, usage=None):
    from openai.types.responses import ResponseUsage

    return OpenAINativeResponse.model_construct(
        id="resp_1",
        model="gpt-test",
        status="completed",
        incomplete_details=None,
        output=output,
        error=None,
        usage=ResponseUsage.model_validate(usage or {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }),
    )


# request mapping

def test_web_search_tool_maps_to_native_web_search(openai_adapter):
    tool = WebSearchTool(
        allowed_domains=["python.org"],
        user_location=UserLocation(country="NL", city="Amsterdam"),
    )

    assert openai_adapter.map_web_search_tool(tool) == {
        "type": "web_search",
        "filters": {"allowed_domains": ["python.org"]},
        "user_location": {"type": "approximate", "country": "NL", "city": "Amsterdam"},
    }

def test_bare_web_search_tool_sends_only_the_type(openai_adapter):
    assert openai_adapter.map_web_search_tool(WebSearchTool()) == {"type": "web_search"}

@pytest.mark.parametrize(
    ("tool", "message"),
    [
        (WebSearchTool(max_uses=3), "max_uses"),
        (WebSearchTool(blocked_domains=["x.com"]), "blocked_domains"),
    ],
)
def test_options_openai_cannot_express_raise(openai_adapter, tool, message):
    with pytest.raises(UnsupportedToolOptionError, match=message):
        openai_adapter.map_web_search_tool(tool)

def test_unsupported_tool_option_error_is_a_value_error():
    assert issubclass(UnsupportedToolOptionError, ValueError)

def test_provider_options_add_native_fields_and_pin_the_type(openai_adapter):
    tool = WebSearchTool(
        allowed_domains=["python.org"],
        provider_options={
            "type": "web_search_2025_08_26",
            "search_context_size": "high",
            "filters": {"allowed_domains": ["ignored.com"], "future_filter": True},
        },
    )

    assert openai_adapter.map_web_search_tool(tool) == {
        "type": "web_search_2025_08_26",
        "search_context_size": "high",
        "filters": {"allowed_domains": ["python.org"], "future_filter": True},
    }

def test_web_search_cache_is_ignored(openai_adapter):
    assert openai_adapter.map_web_search_tool(WebSearchTool(cache=True)) == {"type": "web_search"}

def test_tools_keep_their_order(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(
        model="gpt-test", input=[],
        tools=[
            FunctionTool(name="a", description="a"),
            WebSearchTool(),
            ProviderTool(provider="openai", definition={"type": "image_generation"}),
        ],
    ))

    assert [tool["type"] for tool in kwargs["tools"]] == ["function", "web_search", "image_generation"]

def test_web_search_asks_for_sources(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(model="gpt-test", input=[], tools=[WebSearchTool()]))

    assert kwargs["include"] == ["web_search_call.action.sources"]

def test_sources_include_is_merged_with_native_include(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(
        model="gpt-test", input=[], tools=[WebSearchTool()],
        provider_options={"include": ["reasoning.encrypted_content", "web_search_call.action.sources"]},
    ))

    assert kwargs["include"] == ["reasoning.encrypted_content", "web_search_call.action.sources"]

def test_no_web_search_sends_no_include(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(
        model="gpt-test", input=[], tools=[FunctionTool(name="a", description="a")],
    ))

    assert "include" not in kwargs

def test_provider_tool_is_sent_unchanged(openai_adapter):
    definition = {"type": "file_search", "vector_store_ids": ["vs_1"]}

    assert openai_adapter.map_tools([ProviderTool(provider="openai", definition=definition)]) == [definition]

def test_provider_tool_for_another_provider_raises(openai_adapter):
    tool = ProviderTool(provider="anthropic", definition={"type": "web_fetch_20250910", "name": "web_fetch"})

    with pytest.raises(ValueError, match="'anthropic'"):
        openai_adapter.map_tools([tool])

def test_native_tools_in_request_options_are_kept_without_canonical_tools(openai_adapter):
    native = [{"type": "web_search"}]

    kwargs = openai_adapter._build_request_kwargs(Request(model="gpt-test", input=[], provider_options={"tools": native}))

    assert kwargs["tools"] == native

def test_canonical_tools_win_over_native_tools(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(
        model="gpt-test", input=[], tools=[WebSearchTool()], provider_options={"tools": [{"type": "x"}]},
    ))

    assert kwargs["tools"] == [{"type": "web_search"}]


# responses

def test_search_call_is_normalized(openai_adapter):
    native = _web_search_call({
        "type": "search",
        "query": "python 3.15",
        "queries": ["python 3.15", "python 3.15 release notes"],
        "sources": [{"type": "url", "url": "https://docs.python.org/3.15/whatsnew"}],
    })

    item = openai_adapter._normalize_output_item(native)

    assert isinstance(item, WebSearchCallOutputItem)
    assert item.provider == "openai"
    assert (item.call_id, item.status, item.action) == ("ws_1", "completed", "search")
    assert item.queries == ["python 3.15", "python 3.15 release notes"]
    assert [source.url for source in item.sources] == ["https://docs.python.org/3.15/whatsnew"]
    assert item.sources[0].title is None
    assert item.error is None
    assert item.raw is native

def test_single_query_without_sources_is_normalized(openai_adapter):
    item = openai_adapter._normalize_output_item(_web_search_call({"type": "search", "query": "q", "sources": None}))

    assert item.queries == ["q"]
    assert item.sources == []

def test_open_page_call_is_normalized(openai_adapter):
    item = openai_adapter._normalize_output_item(_web_search_call({"type": "open_page", "url": "https://a.com"}))

    assert (item.action, item.url, item.queries) == ("open_page", "https://a.com", [])

def test_find_in_page_call_is_normalized(openai_adapter):
    item = openai_adapter._normalize_output_item(
        _web_search_call({"type": "find_in_page", "url": "https://a.com", "pattern": "release"})
    )

    assert (item.action, item.url, item.pattern) == ("find_in_page", "https://a.com", "release")

def test_failed_call_keeps_its_status(openai_adapter):
    item = openai_adapter._normalize_output_item(_web_search_call({"type": "search", "query": "q"}, status="failed"))

    assert item.status == "failed"

def test_web_search_calls_are_counted_in_usage(openai_adapter):
    response = openai_adapter._normalize_response(_response([
        _web_search_call({"type": "search", "query": "a"}, id="ws_1"),
        _web_search_call({"type": "open_page", "url": "https://a.com"}, id="ws_2"),
    ]))

    assert [item.type for item in response.output] == ["web_search_call", "web_search_call"]
    assert response.usage.server_tool_requests == {"web_search": 2}

def test_no_server_tools_leaves_server_tool_requests_unset(openai_adapter):
    assert openai_adapter._normalize_response(_response([])).usage.server_tool_requests is None

def test_web_search_call_is_replayed_unchanged(openai_adapter):
    native = _web_search_call({"type": "search", "query": "q"})

    item = openai_adapter._normalize_output_item(native)

    assert openai_adapter.map_output_item(item) is native


# streaming

def test_web_search_call_added_starts_a_server_tool_call(openai_adapter):
    event = SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(type="web_search_call", id="ws_1"),
    )

    [normalized] = openai_adapter._normalize_stream_event(event)

    assert normalized.type == "server_tool_call.started"
    assert (normalized.call_id, normalized.name, normalized.output_index) == ("ws_1", "web_search", 0)
    assert normalized.raw_event is event

def test_web_search_call_done_carries_the_canonical_item(openai_adapter):
    native = _web_search_call({"type": "search", "query": "q"})
    event = SimpleNamespace(type="response.output_item.done", output_index=0, item=native)

    [normalized] = openai_adapter._normalize_stream_event(event)

    assert normalized.type == "output_item.done"
    assert isinstance(normalized.item, WebSearchCallOutputItem)

def test_native_web_search_progress_events_pass_through(openai_adapter):
    event = SimpleNamespace(type="response.web_search_call.searching", output_index=0, item_id="ws_1")

    [normalized] = openai_adapter._normalize_stream_event(event)

    assert normalized.type == "custom_provider_event"
    assert normalized.provider_type == "response.web_search_call.searching"

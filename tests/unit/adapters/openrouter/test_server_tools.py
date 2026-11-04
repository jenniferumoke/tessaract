import pytest

pytest.importorskip("openrouter")

from tessaract import (
    ContainerEnvironment,
    ShellTool,
    UnsupportedToolOptionError,
    UserLocation,
    WebSearchTool,
)
from tessaract.types.output_types import (
    ProviderOutputItem,
    ShellCallOutputItem,
    ShellCommandOutput,
    WebSearchCallOutputItem,
    WebSource,
)
from tests.fixtures import openrouter as payloads

MODEL = "or/openai/gpt-5-mini"
CONTAINER = ShellTool(environment=ContainerEnvironment())

OUTCOME = {"stdout": "3.13\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
SHELL_ACTION = {"commands": ["python -V"], "timeout_ms": 1000, "max_output_length": None}


def _web_search(**overrides):
    return {
        "type": "openrouter:web_search",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "python", "sources": [{"type": "url", "url": "https://python.org"}]},
        **overrides,
    }


def _openrouter_shell(**overrides):
    return {
        "type": "openrouter:shell",
        "id": "sh_1",
        "call_id": "call_sh",
        "status": "completed",
        "action": SHELL_ACTION,
        "container_id": "sess_1",
        "output": [OUTCOME],
        **overrides,
    }


def _shell_call(call_id="call_sh"):
    return {"type": "shell_call", "id": f"item_{call_id}", "call_id": call_id, "status": "completed", "action": SHELL_ACTION}


def _shell_output(call_id="call_sh"):
    return {
        "type": "shell_call_output",
        "id": f"out_{call_id}",
        "call_id": call_id,
        "status": "completed",
        "container_id": "cntr_1",
        "output": [OUTCOME],
    }


def _send(client, openrouter_api, output, tools=None, input="hi"):
    openrouter_api.responses = [payloads.response(output)]
    return client.send(model=MODEL, input=input, tools=tools or [])


def _sent_tool(client, openrouter_api, tool):
    _send(client, openrouter_api, [payloads.message("ok")], tools=[tool])
    return openrouter_api.last_body["tools"][0]


# request mapping

def test_web_search_tool_with_no_options(openrouter_client, openrouter_api):
    assert _sent_tool(openrouter_client, openrouter_api, WebSearchTool()) == {"type": "openrouter:web_search"}


def test_web_search_tool_options(openrouter_client, openrouter_api):
    tool = WebSearchTool(
        max_uses=2,
        blocked_domains=["example.com"],
        user_location=UserLocation(country="NL", city="Amsterdam"),
        provider_options={"engine": "exa", "max_results": 3, "max_uses": 9},
    )

    assert _sent_tool(openrouter_client, openrouter_api, tool)["parameters"] == {
        "engine": "exa",
        "max_results": 3,
        "max_uses": 2,
        "excluded_domains": ["example.com"],
        "user_location": {"type": "approximate", "city": "Amsterdam", "country": "NL"},
    }


def test_web_search_allowed_domains(openrouter_client, openrouter_api):
    tool = WebSearchTool(allowed_domains=["python.org"])

    assert _sent_tool(openrouter_client, openrouter_api, tool)["parameters"] == {"allowed_domains": ["python.org"]}


def test_web_search_type_cannot_be_pinned(openrouter_client):
    with pytest.raises(UnsupportedToolOptionError, match="ProviderTool"):
        openrouter_client.send(model=MODEL, input="hi", tools=[WebSearchTool(provider_options={"type": "web_search"})])


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (ContainerEnvironment(), {"type": "container_auto"}),
        (ContainerEnvironment(container_id="sess_1"), {"type": "container_reference", "container_id": "sess_1"}),
        (
            ContainerEnvironment(file_ids=["or_file_1"], allowed_domains=["pypi.org"]),
            {
                "type": "container_auto",
                "file_ids": ["or_file_1"],
                "network_policy": {"type": "allowlist", "allowed_domains": ["pypi.org"]},
            },
        ),
        (ContainerEnvironment(network_disabled=True), {"type": "container_auto", "network_policy": {"type": "disabled"}}),
    ],
    ids=["new", "reused", "files_and_allowlist", "network_disabled"],
)
def test_container_environment(openrouter_client, openrouter_api, environment, expected):
    tool = _sent_tool(openrouter_client, openrouter_api, ShellTool(environment=environment))

    assert tool == {"type": "openrouter:shell", "parameters": {"environment": expected}}


def test_shell_tool_native_options(openrouter_client, openrouter_api):
    tool = ShellTool(environment=ContainerEnvironment(), provider_options={"engine": "openrouter"})

    assert _sent_tool(openrouter_client, openrouter_api, tool)["parameters"] == {
        "engine": "openrouter",
        "environment": {"type": "container_auto"},
    }


def test_memory_limit_is_not_supported(openrouter_client):
    tool = ShellTool(environment=ContainerEnvironment(memory_limit="4g"))

    with pytest.raises(UnsupportedToolOptionError, match="memory_limit"):
        openrouter_client.send(model=MODEL, input="hi", tools=[tool])


@pytest.mark.parametrize(
    "tool",
    [WebSearchTool(cache=True), ShellTool(environment=ContainerEnvironment(), cache=True)],
    ids=["web_search", "shell"],
)
def test_caching_server_tools_is_not_supported_yet(openrouter_client, tool):
    with pytest.raises(NotImplementedError, match="Prompt caching"):
        openrouter_client.send(model=MODEL, input="hi", tools=[tool])


# responses

@pytest.mark.parametrize(
    "item",
    [
        _web_search(),
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "python", "sources": [{"type": "url", "url": "https://python.org"}]},
        },
    ],
    ids=["openrouter", "upstream"],
)
def test_web_search_call_is_normalized(openrouter_client, openrouter_api, item):
    response = _send(openrouter_client, openrouter_api, [item, payloads.message("Done")], [WebSearchTool()])

    call = response.output[0]
    assert isinstance(call, WebSearchCallOutputItem)
    assert call.provider == "openrouter"
    assert (call.call_id, call.status, call.action) == ("ws_1", "completed", "search")
    assert call.queries == ["python"]
    assert call.sources == [WebSource(url="https://python.org")]
    assert response.usage.server_tool_requests == {"web_search": 1}


def test_failed_web_search_keeps_its_error(openrouter_client, openrouter_api):
    item = _web_search(status="failed", error="No engine available", action=None)

    call = _send(openrouter_client, openrouter_api, [item]).output[0]

    assert (call.status, call.error, call.queries) == ("failed", "No engine available", [])


def test_web_search_with_an_unknown_status_stays_native(openrouter_client, openrouter_api):
    item = _send(openrouter_client, openrouter_api, [_web_search(status="brand_new")]).output[0]

    assert isinstance(item, ProviderOutputItem)
    assert item.provider_type == "openrouter:web_search"


def test_openrouter_shell_item_is_normalized(openrouter_client, openrouter_api):
    files = [{"type": "container_file_citation", "container_id": "sess_1", "file_id": "f_1", "filename": "a.txt", "start_index": 0, "end_index": 1}]

    response = _send(openrouter_client, openrouter_api, [_openrouter_shell(files=files)], [CONTAINER])

    [call] = response.output
    assert isinstance(call, ShellCallOutputItem)
    assert call.executed_by == "provider"
    assert call.provider == "openrouter"
    assert (call.call_id, call.status, call.timeout_ms) == ("call_sh", "completed", 1000)
    assert call.commands == ["python -V"]
    assert call.output == [ShellCommandOutput(stdout="3.13\n", exit_code=0)]
    assert call.file_ids == ["f_1"]
    assert response.container_id == "sess_1"
    assert response.usage.server_tool_requests == {"shell": 1}


def test_failed_openrouter_shell_item(openrouter_client, openrouter_api):
    item = _openrouter_shell(status="failed", error="Container limit reached", output=None, action=None)

    [call] = _send(openrouter_client, openrouter_api, [item]).output

    assert (call.status, call.error, call.commands, call.output) == ("incomplete", "Container limit reached", [], [])


def test_command_timeout(openrouter_client, openrouter_api):
    item = _openrouter_shell(output=[{"stdout": "", "stderr": "", "outcome": {"type": "timeout"}}])

    [call] = _send(openrouter_client, openrouter_api, [item]).output

    assert call.output[0].timed_out is True


def test_shell_call_and_its_output_merge(openrouter_client, openrouter_api):
    response = _send(openrouter_client, openrouter_api, [_shell_call(), _shell_output()], [CONTAINER])

    [call] = response.output
    assert call.executed_by == "provider"
    assert call.commands == ["python -V"]
    assert call.output[0].stdout == "3.13\n"
    assert response.container_id == "cntr_1"


def test_interleaved_shell_outputs_keep_their_order(openrouter_client, openrouter_api):
    output = [_shell_call("a"), _shell_call("b"), _shell_output("a"), _shell_output("b")]

    response = _send(openrouter_client, openrouter_api, output, [CONTAINER])

    assert [(item.call_id, bool(item.output)) for item in response.output] == [
        ("a", False), ("b", False), ("a", True), ("b", True),
    ]
    assert response.output[2].commands == ["python -V"]


def test_shell_call_without_output_is_still_running(openrouter_client, openrouter_api):
    [call] = _send(openrouter_client, openrouter_api, [_shell_call()], [CONTAINER]).output

    assert (call.executed_by, call.output) == ("provider", [])


@pytest.mark.parametrize(
    "output",
    [[_web_search()], [_openrouter_shell()], [_shell_call(), _shell_output()]],
    ids=["web_search", "openrouter_shell", "shell_call"],
)
def test_server_tool_items_replay_unchanged(openrouter_client, openrouter_api, output):
    first = _send(openrouter_client, openrouter_api, output)

    openrouter_client.send(model=MODEL, input=["hi", *first.output, "and now?"])

    sent = openrouter_api.last_body["input"][1:-1]
    assert len(sent) == len(output)
    for sent_item, native in zip(sent, output):
        for key, value in native.items():
            if key == "action":
                # the SDK leaves out unset optional action fields
                assert sent_item[key].items() >= {k: v for k, v in value.items() if v is not None}.items()
            else:
                assert sent_item[key] == value


# streaming

def _stream(client, openrouter_api, *events):
    openrouter_api.responses = [payloads.stream(*events)]
    return list(client.send(model=MODEL, input="hi", stream=True))


def test_stream_of_a_web_search(openrouter_client, openrouter_api):
    started, done = _stream(
        openrouter_client, openrouter_api,
        {"type": "response.output_item.added", "output_index": 0, "item": _web_search(status="in_progress", action=None)},
        {"type": "response.output_item.done", "output_index": 0, "item": _web_search()},
    )

    assert (started.type, started.name, started.call_id) == ("server_tool_call.started", "web_search", "ws_1")
    assert isinstance(done.item, WebSearchCallOutputItem)


def test_stream_of_an_openrouter_shell_call(openrouter_client, openrouter_api):
    started, done = _stream(
        openrouter_client, openrouter_api,
        {"type": "response.output_item.added", "output_index": 0, "item": _openrouter_shell(status="in_progress", output=None)},
        {"type": "response.output_item.done", "output_index": 0, "item": _openrouter_shell()},
    )

    assert (started.type, started.name, started.call_id) == ("server_tool_call.started", "shell", "call_sh")
    assert done.item.output[0].stdout == "3.13\n"


def test_stream_of_a_shell_call_finishes_with_its_output(openrouter_client, openrouter_api):
    started, done = _stream(
        openrouter_client, openrouter_api,
        {"type": "response.output_item.added", "output_index": 0, "item": _shell_call()},
        {"type": "response.output_item.done", "output_index": 0, "item": _shell_call()},
        {"type": "response.output_item.done", "output_index": 1, "item": _shell_output()},
    )

    assert (started.type, started.name) == ("server_tool_call.started", "shell")
    assert done.item.commands == ["python -V"]
    assert done.item.output[0].stdout == "3.13\n"


def test_stream_ends_shell_calls_that_got_no_output(openrouter_client, openrouter_api):
    completed = payloads.response([_shell_call()])

    done, end = _stream(
        openrouter_client, openrouter_api,
        {"type": "response.output_item.done", "output_index": 0, "item": _shell_call()},
        {"type": "response.completed", "response": completed},
    )

    assert done.type == "output_item.done"
    assert (done.item.call_id, done.item.output) == ("call_sh", [])
    assert end.type == "response.completed"

import pytest

pytest.importorskip("anthropic")

from tessaract import (
    ContainerEnvironment,
    ShellCommandOutput,
    ShellTool,
    UnsupportedToolOptionError,
    UserMessage,
)
from tessaract.adapters.anthropic.anthropic_adapter import (
    DEFAULT_CODE_EXECUTION_TOOL_TYPE,
)
from tessaract.types.output_types import ProviderOutputItem, ShellCallOutputItem
from tessaract.types.request import Request

HOSTED = ShellTool(environment=ContainerEnvironment())


def _message(content, stop_reason="end_turn", container=None):
    message = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test-20260101",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    if container is not None:
        message["container"] = container
    return message

CONTAINER = {"id": "container_1", "expires_at": "2026-10-22T12:00:00Z"}

BASH_CALL = {
    "type": "server_tool_use", "id": "srvtoolu_1", "name": "bash_code_execution",
    "input": {"command": "python -c 'print(6 * 7)'"},
}
BASH_RESULT = {
    "type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_1",
    "content": {
        "type": "bash_code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0,
        "content": [{"type": "bash_code_execution_output", "file_id": "file_out_1"}],
    },
}


def _request(*tools, input=None):
    return Request(model="claude-test", input=input or [{"role": "user", "content": "hi"}], tools=list(tools))


# tool definition and request

def test_container_shell_tool_maps_to_code_execution(anthropic_adapter):
    assert anthropic_adapter.map_shell_tool(HOSTED) == {"type": DEFAULT_CODE_EXECUTION_TOOL_TYPE, "name": "code_execution"}
    assert DEFAULT_CODE_EXECUTION_TOOL_TYPE == "code_execution_20250825"

def test_code_execution_version_can_be_pinned_and_cached(anthropic_adapter):
    tool = ShellTool(environment=ContainerEnvironment(), cache=True, provider_options={"type": "code_execution_20260120"})

    assert anthropic_adapter.map_shell_tool(tool) == {
        "type": "code_execution_20260120", "name": "code_execution", "cache_control": {"type": "ephemeral"},
    }

@pytest.mark.parametrize(
    ("container", "message"),
    [
        (ContainerEnvironment(memory_limit="4g"), "memory_limit"),
        (ContainerEnvironment(allowed_domains=["pypi.org"]), "allowed_domains"),
    ],
)
def test_options_anthropic_containers_cannot_express_raise(anthropic_adapter, container, message):
    with pytest.raises(UnsupportedToolOptionError, match=message):
        anthropic_adapter.map_shell_tool(ShellTool(environment=container))

def test_network_disabled_is_how_anthropic_containers_run(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(_request(ShellTool(environment=ContainerEnvironment(network_disabled=True))))

    assert kwargs["tools"] == [{"type": "code_execution_20250825", "name": "code_execution"}]
    assert "container" not in kwargs

def test_container_id_is_sent_as_the_container_parameter(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(_request(ShellTool(environment=ContainerEnvironment(container_id="container_1"))))

    assert kwargs["container"] == "container_1"

def test_container_provider_options_make_a_container_object(anthropic_adapter):
    skills = [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]
    tool = ShellTool(environment=ContainerEnvironment(container_id="container_1", provider_options={"skills": skills}))

    assert anthropic_adapter._build_request_kwargs(_request(tool))["container"] == {"skills": skills, "id": "container_1"}

def test_file_ids_are_uploaded_at_the_start_of_the_first_user_turn(anthropic_adapter):
    tool = ShellTool(environment=ContainerEnvironment(file_ids=["file_a", "file_b"]))
    already = {"type": "container_upload", "file_id": "file_b"}
    input = [
        {"role": "user", "content": [already, {"type": "text", "text": "Analyze the data"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Sure."}]},
        {"role": "user", "content": "And plot it"},
    ]

    messages = anthropic_adapter._build_request_kwargs(_request(tool, input=input))["messages"]

    assert messages[0]["content"] == [
        {"type": "container_upload", "file_id": "file_a"}, already, {"type": "text", "text": "Analyze the data"},
    ]
    assert messages[2]["content"] == [{"type": "text", "text": "And plot it"}]

def test_different_containers_in_one_request_raise(anthropic_adapter):
    with pytest.raises(ValueError, match="one container per request"):
        anthropic_adapter._build_request_kwargs(_request(
            ShellTool(environment=ContainerEnvironment(container_id="a")),
            ShellTool(environment=ContainerEnvironment(container_id="b")),
        ))

def test_local_and_container_shells_can_be_combined(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(_request(ShellTool(), HOSTED))

    assert [tool["name"] for tool in kwargs["tools"]] == ["bash", "code_execution"]


# responses

def test_bash_call_and_result_become_one_provider_shell_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([BASH_CALL, BASH_RESULT, {"type": "text", "text": "It's 42."}], container=CONTAINER)

    response = anthropic_client.send(model="ant/claude-test", input="Compute 6 * 7", tools=[HOSTED])

    call, message = response.output
    assert isinstance(call, ShellCallOutputItem)
    assert (call.executed_by, call.provider, call.call_id, call.status) == ("provider", "anthropic", "srvtoolu_1", "completed")
    assert call.commands == ["python -c 'print(6 * 7)'"]
    assert call.output == [ShellCommandOutput(stdout="42\n")]
    assert call.file_ids == ["file_out_1"]
    assert call.error is None
    assert [block.type for block in call.raw] == ["server_tool_use", "bash_code_execution_tool_result"]
    assert message.type == "assistant_message"
    assert response.container_id == "container_1"

def test_failed_command_keeps_its_exit_code(anthropic_client, anthropic_api):
    result = {**BASH_RESULT, "content": {**BASH_RESULT["content"], "stdout": "", "stderr": "boom", "return_code": 1, "content": []}}
    anthropic_api.message = _message([BASH_CALL, result])

    [call] = anthropic_client.send(model="ant/claude-test", input="hi").output

    assert call.output == [ShellCommandOutput(stderr="boom", exit_code=1)]
    assert call.status == "completed"

@pytest.mark.parametrize(
    ("error_code", "output"),
    [("execution_time_exceeded", [ShellCommandOutput(timed_out=True)]), ("unavailable", [])],
)
def test_error_results(anthropic_client, anthropic_api, error_code, output):
    result = {**BASH_RESULT, "content": {"type": "bash_code_execution_tool_result_error", "error_code": error_code}}
    anthropic_api.message = _message([BASH_CALL, result])

    [call] = anthropic_client.send(model="ant/claude-test", input="hi").output

    assert (call.status, call.error, call.output) == ("incomplete", error_code, output)

def test_paused_turn_keeps_the_running_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([BASH_CALL], stop_reason="pause_turn")

    [call] = anthropic_client.send(model="ant/claude-test", input="hi").output

    assert (call.executed_by, call.status, call.output) == ("provider", "in_progress", [])

def test_text_editor_blocks_pass_through(anthropic_client, anthropic_api):
    anthropic_api.message = _message([
        {"type": "server_tool_use", "id": "srvtoolu_2", "name": "text_editor_code_execution",
         "input": {"command": "view", "path": "data.csv"}},
    ])

    [item] = anthropic_client.send(model="ant/claude-test", input="hi").output

    assert isinstance(item, ProviderOutputItem)

def test_interleaved_calls_keep_native_order(anthropic_client, anthropic_api):
    second_call = {**BASH_CALL, "id": "srvtoolu_2", "input": {"command": "ls"}}
    second_result = {**BASH_RESULT, "tool_use_id": "srvtoolu_2"}
    anthropic_api.message = _message([BASH_CALL, second_call, BASH_RESULT, second_result])

    output = anthropic_client.send(model="ant/claude-test", input="hi").output

    assert [(item.call_id, item.commands, len(item.raw)) for item in output] == [
        ("srvtoolu_1", ["python -c 'print(6 * 7)'"], 1),
        ("srvtoolu_2", ["ls"], 1),
        ("srvtoolu_1", ["python -c 'print(6 * 7)'"], 1),
        ("srvtoolu_2", ["ls"], 1),
    ]

def test_container_is_reused_and_call_replayed_on_the_next_turn(anthropic_client, anthropic_api):
    anthropic_api.message = _message([BASH_CALL, BASH_RESULT], container=CONTAINER)
    history: list = [UserMessage(content="Compute 6 * 7")]

    response = anthropic_client.send(model="ant/claude-test", input=history, tools=[HOSTED])
    history.extend(response.output)
    history.append("Now times 2")

    anthropic_api.message = _message([{"type": "text", "text": "84"}])
    tools = [ShellTool(environment=ContainerEnvironment(container_id=response.container_id))]
    anthropic_client.send(model="ant/claude-test", input=history, tools=tools)

    body = anthropic_api.last_body
    assert body["container"] == "container_1"
    _, assistant, _ = body["messages"]
    assert assistant["content"] == [BASH_CALL, BASH_RESULT]


# streaming

def test_stream_reports_a_provider_shell_call(anthropic_client, anthropic_api):
    anthropic_api.sse_events = [
        {"type": "message_start", "message": _message([], stop_reason=None)},
        {"type": "content_block_start", "index": 0, "content_block": {**BASH_CALL, "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": BASH_RESULT},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None, "container": CONTAINER},
         "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]

    events = list(anthropic_client.send(model="ant/claude-test", input="hi", tools=[HOSTED], stream=True))

    assert "tool_arguments.delta" not in [event.type for event in events]
    [started] = [event for event in events if event.type == "server_tool_call.started"]
    assert (started.call_id, started.name) == ("srvtoolu_1", "shell")

    [done] = [event.item for event in events if event.type == "output_item.done"]
    assert (done.executed_by, done.commands, done.output) == ("provider", ["ls"], [ShellCommandOutput(stdout="42\n")])

    [completed] = [event for event in events if event.type == "response.completed"]
    assert completed.response.container_id == "container_1"

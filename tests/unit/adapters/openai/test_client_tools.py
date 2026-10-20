import base64
from types import SimpleNamespace

import pytest
from openai.types.responses import (
    ResponseComputerToolCall,
    ResponseFunctionShellToolCall,
)

from tessaract import (
    ComputerToolResult,
    ComputerUseTool,
    ProviderToolResult,
    ShellCommandOutput,
    ShellTool,
    ShellToolResult,
    UnsupportedToolOptionError,
)
from tessaract.types import computer_actions as actions
from tessaract.types.output_types import (
    ComputerCallOutputItem,
    ProviderOutputItem,
    SafetyCheck,
    ShellCallOutputItem,
)
from tessaract.types.request import Request

PNG = b"\x89PNG\r\n\x1a\nfake"


def _computer_call(**fields):
    return ResponseComputerToolCall.model_validate({
        "id": "cu_1", "call_id": "call_1", "type": "computer_call", "status": "completed",
        "pending_safety_checks": [], **fields,
    })

def _shell_call(**fields):
    return ResponseFunctionShellToolCall.model_validate({
        "id": "sh_1", "call_id": "call_1", "type": "shell_call", "status": "completed",
        "action": {"commands": ["ls", "pwd"], "timeout_ms": 5000, "max_output_length": 2000}, **fields,
    })


# tool definitions

def test_computer_use_tool_maps_to_computer(openai_adapter):
    assert openai_adapter.map_computer_use_tool(ComputerUseTool()) == {"type": "computer"}

def test_computer_use_tool_cache_is_ignored(openai_adapter):
    assert openai_adapter.map_computer_use_tool(ComputerUseTool(cache=True)) == {"type": "computer"}

def test_shell_tool_maps_to_local_shell(openai_adapter):
    assert openai_adapter.map_shell_tool(ShellTool()) == {"type": "shell", "environment": {"type": "local"}}

def test_shell_tool_keeps_a_local_environment_with_skills(openai_adapter):
    environment = {"type": "local", "skills": [{"name": "x", "description": "x", "path": "/skills/x"}]}

    native = openai_adapter.map_shell_tool(ShellTool(provider_options={"environment": environment}))

    assert native == {"type": "shell", "environment": environment}

def test_shell_tool_rejects_a_hosted_environment(openai_adapter):
    with pytest.raises(ValueError, match="ProviderTool"):
        openai_adapter.map_shell_tool(ShellTool(provider_options={"environment": {"type": "container_auto"}}))

def test_client_tools_are_sent_in_order(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(model="gpt-test", input=[], tools=[ShellTool(), ComputerUseTool()]))

    assert [tool["type"] for tool in kwargs["tools"]] == ["shell", "computer"]


# computer calls

def test_batched_computer_actions_are_normalized(openai_adapter):
    native = _computer_call(actions=[
        {"type": "click", "button": "wheel", "x": 10, "y": 20, "keys": ["SHIFT"]},
        {"type": "double_click", "x": 1, "y": 2},
        {"type": "drag", "path": [{"x": 0, "y": 0}, {"x": 5, "y": 5}]},
        {"type": "keypress", "keys": ["CTRL", "S"]},
        {"type": "move", "x": 3, "y": 4},
        {"type": "screenshot"},
        {"type": "scroll", "x": 100, "y": 200, "scroll_x": 0, "scroll_y": 300},
        {"type": "type", "text": "hello"},
        {"type": "wait"},
    ])

    item = openai_adapter._normalize_output_item(native)

    assert isinstance(item, ComputerCallOutputItem)
    assert (item.provider, item.call_id, item.status) == ("openai", "call_1", "completed")
    assert item.raw is native
    assert item.actions == [
        actions.Click(x=10, y=20, button="middle", modifiers=["SHIFT"]),
        actions.Click(x=1, y=2, clicks=2),
        actions.Drag(path=[actions.Point(x=0, y=0), actions.Point(x=5, y=5)]),
        actions.KeyPress(keys=["CTRL", "S"]),
        actions.MoveMouse(x=3, y=4),
        actions.Screenshot(),
        actions.Scroll(x=100, y=200, dx=0, dy=300, unit="pixels"),
        actions.TypeText(text="hello"),
        actions.Wait(),
    ]

def test_single_action_becomes_a_list(openai_adapter):
    item = openai_adapter._normalize_output_item(_computer_call(action={"type": "screenshot"}))

    assert item.actions == [actions.Screenshot()]

def test_pending_safety_checks_are_normalized(openai_adapter):
    item = openai_adapter._normalize_output_item(_computer_call(
        action={"type": "screenshot"},
        pending_safety_checks=[{"id": "sc_1", "code": "malicious_instructions", "message": "Careful"}],
    ))

    assert item.safety_checks == [SafetyCheck(id="sc_1", code="malicious_instructions", message="Careful")]

def test_unknown_action_keeps_its_payload(openai_adapter):
    action = SimpleNamespace(type="teleport", model_dump=lambda mode: {"type": "teleport", "to": "mars"})

    normalized = openai_adapter._normalize_computer_action(action)

    assert normalized == actions.ProviderComputerAction(provider_type="teleport", payload={"type": "teleport", "to": "mars"})

def test_computer_call_is_replayed_unchanged(openai_adapter):
    native = _computer_call(action={"type": "screenshot"})

    assert openai_adapter.map_output_item(openai_adapter._normalize_output_item(native)) is native


# computer results

def test_computer_result_sends_the_screenshot_as_a_data_url(openai_adapter):
    native = openai_adapter.map_computer_tool_result(ComputerToolResult(
        call_id="call_1", screenshot=PNG, output="ignored on OpenAI",
        acknowledged_safety_checks=[SafetyCheck(id="sc_1", code="malicious_instructions")],
    ))

    assert native == {
        "type": "computer_call_output",
        "call_id": "call_1",
        "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64," + base64.b64encode(PNG).decode()},
        "acknowledged_safety_checks": [{"id": "sc_1", "code": "malicious_instructions"}],
    }

def test_computer_result_needs_a_screenshot(openai_adapter):
    with pytest.raises(ValueError, match="screenshot"):
        openai_adapter.map_computer_tool_result(ComputerToolResult(call_id="call_1", output="OK"))

def test_computer_error_result_raises(openai_adapter):
    with pytest.raises(UnsupportedToolOptionError, match="no error flag"):
        openai_adapter.map_computer_tool_result(ComputerToolResult(call_id="call_1", screenshot=PNG, is_error=True))

def test_computer_result_provider_options_cannot_override_canonical_fields(openai_adapter):
    native = openai_adapter.map_computer_tool_result(ComputerToolResult(
        call_id="call_1", screenshot=PNG, provider_options={"call_id": "sneaky", "status": "completed"},
    ))

    assert (native["call_id"], native["status"]) == ("call_1", "completed")


# shell calls and results

def test_shell_call_is_normalized(openai_adapter):
    native = _shell_call(environment={"type": "local"})

    item = openai_adapter._normalize_output_item(native)

    assert isinstance(item, ShellCallOutputItem)
    assert item.provider == "openai"
    assert (item.call_id, item.commands, item.timeout_ms, item.max_output_length) == ("call_1", ["ls", "pwd"], 5000, 2000)
    assert item.restart is False
    assert item.raw is native

def test_shell_call_without_environment_is_a_local_call(openai_adapter):
    assert isinstance(openai_adapter._normalize_output_item(_shell_call()), ShellCallOutputItem)

def test_hosted_shell_call_passes_through(openai_adapter):
    item = openai_adapter._normalize_output_item(_shell_call(environment={"type": "container_reference", "container_id": "cntr_1"}))

    assert isinstance(item, ProviderOutputItem)
    assert item.provider_type == "shell_call"

def test_shell_result_has_one_output_per_command(openai_adapter):
    native = openai_adapter.map_shell_tool_result(ShellToolResult(
        call_id="call_1",
        output=[
            ShellCommandOutput(stdout="a.py\n"),
            ShellCommandOutput(stderr="boom", exit_code=2),
            ShellCommandOutput(stdout="partial", timed_out=True),
        ],
        max_output_length=2000,
    ))

    assert native == {
        "type": "shell_call_output",
        "call_id": "call_1",
        "output": [
            {"stdout": "a.py\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
            {"stdout": "", "stderr": "boom", "outcome": {"type": "exit", "exit_code": 2}},
            {"stdout": "partial", "stderr": "", "outcome": {"type": "timeout"}},
        ],
        "max_output_length": 2000,
    }

def test_shell_error_result_raises(openai_adapter):
    with pytest.raises(UnsupportedToolOptionError, match="no error flag"):
        openai_adapter.map_shell_tool_result(ShellToolResult(call_id="call_1", is_error=True))


# provider tool results

def test_provider_tool_result_is_sent_unchanged(openai_adapter):
    item = {"type": "apply_patch_call_output", "call_id": "call_1", "status": "completed"}

    assert openai_adapter.map_provider_tool_result(ProviderToolResult(provider="openai", item=item)) == item

def test_provider_tool_result_for_another_provider_raises(openai_adapter):
    with pytest.raises(ValueError, match="'anthropic'"):
        openai_adapter.map_provider_tool_result(ProviderToolResult(provider="anthropic", item={"type": "tool_result"}))

def test_results_are_accepted_as_send_input(client):
    request = client._build_request_model(
        model="gpt-test", provider="oai",
        input=[
            ComputerToolResult(call_id="c1", screenshot=PNG),
            ShellToolResult(call_id="c2", output=[ShellCommandOutput(stdout="ok")]),
        ],
        reasoning=None, max_tokens=None, tools=[], request_options={}, stream=False,
    )

    assert [item["type"] for item in request.input] == ["computer_call_output", "shell_call_output"]


# streaming

@pytest.mark.parametrize(("item_type", "name"), [("computer_call", "computer"), ("shell_call", "shell")])
def test_client_tool_call_added_starts_a_tool_call(openai_adapter, item_type, name):
    event = SimpleNamespace(
        type="response.output_item.added", output_index=1,
        item=SimpleNamespace(type=item_type, id="x_1", call_id="call_1", environment=None),
    )

    [normalized] = openai_adapter._normalize_stream_event(event)

    assert normalized.type == "tool_call.started"
    assert (normalized.call_id, normalized.name, normalized.call_type, normalized.output_index) == ("call_1", name, item_type, 1)

def test_function_call_started_has_function_call_type(openai_adapter):
    event = SimpleNamespace(
        type="response.output_item.added", output_index=0,
        item=SimpleNamespace(type="function_call", call_id="call_1", name="get_weather"),
    )

    [normalized] = openai_adapter._normalize_stream_event(event)

    assert normalized.call_type == "function_call"

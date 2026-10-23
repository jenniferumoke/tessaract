import base64

import pytest

pytest.importorskip("anthropic")

from tessaract import (
    ComputerToolResult,
    ComputerUseTool,
    FunctionTool,
    ProviderToolResult,
    ShellCommandOutput,
    ShellTool,
    ShellToolResult,
)
from tessaract.adapters.anthropic.anthropic_adapter import (
    DEFAULT_BASH_TOOL_TYPE,
    DEFAULT_COMPUTER_TOOLSET_TYPE,
)
from tessaract.types import computer_actions as actions
from tessaract.types.output_types import (
    ComputerCallOutputItem,
    FunctionCallOutputItem,
    ShellCallOutputItem,
)

PNG = b"\x89PNG\r\n\x1a\nfake"


def _message(content, stop_reason="tool_use"):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test-20260101",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

def _member(name, input, id="toolu_1"):
    return {"type": "tool_use", "id": id, "name": name, "toolset_name": "computer", "input": input}

BASH = {"type": "tool_use", "id": "toolu_b", "name": "bash", "input": {"command": "ls -la"}}


# tool definitions

def test_computer_use_tool_maps_to_the_toolset(anthropic_adapter):
    assert anthropic_adapter.map_computer_use_tool(ComputerUseTool()) == {"type": DEFAULT_COMPUTER_TOOLSET_TYPE}

def test_computer_use_tool_keeps_native_configs_and_cache(anthropic_adapter):
    native = anthropic_adapter.map_computer_use_tool(
        ComputerUseTool(cache=True, provider_options={"configs": {"zoom": {"enabled": False}}})
    )

    assert native == {
        "type": "computer_toolset_20260801",
        "configs": {"zoom": {"enabled": False}},
        "cache_control": {"type": "ephemeral"},
    }

def test_earlier_computer_tool_versions_are_rejected(anthropic_adapter):
    with pytest.raises(ValueError, match="ProviderTool"):
        anthropic_adapter.map_computer_use_tool(ComputerUseTool(provider_options={"type": "computer_20251124"}))

def test_shell_tool_maps_to_bash(anthropic_adapter):
    assert anthropic_adapter.map_shell_tool(ShellTool()) == {"type": DEFAULT_BASH_TOOL_TYPE, "name": "bash"}

def test_cached_shell_tool_gets_a_breakpoint(anthropic_adapter):
    assert anthropic_adapter.map_shell_tool(ShellTool(cache=True))["cache_control"] == {"type": "ephemeral"}


# computer calls

@pytest.mark.parametrize(
    ("member", "input", "expected"),
    [
        ("screenshot", {}, actions.Screenshot()),
        ("left_click", {"coordinate": [512, 742]}, actions.Click(x=512, y=742)),
        ("left_click", {"coordinate": [5, 6], "text": "ctrl+shift"}, actions.Click(x=5, y=6, modifiers=["ctrl", "shift"])),
        ("left_click", {}, actions.Click()),
        ("right_click", {"coordinate": [1, 2]}, actions.Click(x=1, y=2, button="right")),
        ("middle_click", {"coordinate": [1, 2]}, actions.Click(x=1, y=2, button="middle")),
        ("double_click", {"coordinate": [1, 2]}, actions.Click(x=1, y=2, clicks=2)),
        ("triple_click", {"coordinate": [1, 2]}, actions.Click(x=1, y=2, clicks=3)),
        ("left_click_drag", {"start_coordinate": [200, 300], "coordinate": [600, 300]},
         actions.Drag(path=[actions.Point(x=200, y=300), actions.Point(x=600, y=300)])),
        ("mouse_move", {"coordinate": [7, 8]}, actions.MoveMouse(x=7, y=8)),
        ("left_mouse_down", {}, actions.MouseDown()),
        ("left_mouse_up", {}, actions.MouseUp()),
        ("cursor_position", {}, actions.CursorPosition()),
        ("scroll", {"scroll_direction": "down", "scroll_amount": 3, "coordinate": [10, 20]},
         actions.Scroll(x=10, y=20, dy=3, unit="clicks")),
        ("scroll", {"scroll_direction": "left", "scroll_amount": 2}, actions.Scroll(dx=-2, unit="clicks")),
        ("type", {"text": "hello"}, actions.TypeText(text="hello")),
        ("key", {"text": "ctrl+s"}, actions.KeyPress(keys=["ctrl", "s"])),
        ("key", {"text": "Return", "repeat": 3}, actions.KeyPress(keys=["Return"], repeat=3)),
        ("key", {"text": "+"}, actions.KeyPress(keys=["+"])),
        ("hold_key", {"text": "shift", "duration": 2}, actions.HoldKey(keys=["shift"], duration=2)),
        ("wait", {"duration": 1.5}, actions.Wait(duration=1.5)),
        ("zoom", {"region": [0, 0, 100, 50]}, actions.Zoom(region=(0, 0, 100, 50))),
        ("future_member", {"a": 1}, actions.ProviderComputerAction(provider_type="future_member", payload={"a": 1})),
        ("scroll", {"scroll_direction": "sideways", "scroll_amount": 1},
         actions.ProviderComputerAction(provider_type="scroll", payload={"scroll_direction": "sideways", "scroll_amount": 1})),
        ("mouse_move", {}, actions.ProviderComputerAction(provider_type="mouse_move", payload={})),
    ],
)
def test_toolset_members_are_normalized(anthropic_client, anthropic_api, member, input, expected):
    anthropic_api.message = _message([_member(member, input)])

    [item] = anthropic_client.send(model="ant/claude-test", input="hi", tools=[ComputerUseTool()]).output

    assert isinstance(item, ComputerCallOutputItem)
    assert (item.provider, item.call_id, item.safety_checks) == ("anthropic", "toolu_1", [])
    assert item.actions == [expected]

def test_a_function_named_like_a_member_stays_a_function_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([{"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}])

    [item] = anthropic_client.send(model="ant/claude-test", input="hi", tools=[ComputerUseTool()]).output

    assert isinstance(item, FunctionCallOutputItem)


# computer results

def test_screenshot_result_is_an_image_with_the_toolset_name(anthropic_adapter):
    native = anthropic_adapter.map_computer_tool_result(ComputerToolResult(call_id="toolu_1", screenshot=PNG))

    assert native == {"role": "user", "content": [{
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "toolset_name": "computer",
        "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}}],
    }]}

def test_result_without_screenshot_or_output_says_ok(anthropic_adapter):
    [block] = anthropic_adapter.map_computer_tool_result(ComputerToolResult(call_id="toolu_1"))["content"]

    assert block["content"] == [{"type": "text", "text": "OK"}]
    assert block["toolset_name"] == "computer"

def test_error_result_carries_the_flag_and_text(anthropic_adapter):
    [block] = anthropic_adapter.map_computer_tool_result(ComputerToolResult(
        call_id="toolu_2", is_error=True, output="Not executed: an earlier computer action in this turn failed.",
    ))["content"]

    assert block["is_error"] is True
    assert block["content"] == [{"type": "text", "text": "Not executed: an earlier computer action in this turn failed."}]

def test_screenshot_and_output_are_both_sent(anthropic_adapter):
    [block] = anthropic_adapter.map_computer_tool_result(
        ComputerToolResult(call_id="toolu_1", screenshot=PNG, output="clicked", cache=True)
    )["content"]

    assert [part["type"] for part in block["content"]] == ["image", "text"]
    assert block["cache_control"] == {"type": "ephemeral"}


# shell calls and results

def test_bash_call_is_a_shell_call_when_shell_tool_is_declared(anthropic_client, anthropic_api):
    anthropic_api.message = _message([BASH])

    [item] = anthropic_client.send(model="ant/claude-test", input="hi", tools=[ShellTool()]).output

    assert isinstance(item, ShellCallOutputItem)
    assert (item.provider, item.call_id, item.commands, item.restart) == ("anthropic", "toolu_b", ["ls -la"], False)
    assert item.timeout_ms is None

def test_bash_restart_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([{**BASH, "input": {"restart": True}}])

    [item] = anthropic_client.send(model="ant/claude-test", input="hi", tools=[ShellTool()]).output

    assert (item.commands, item.restart) == ([], True)

def test_a_function_named_bash_stays_a_function_call(anthropic_client, anthropic_api):
    anthropic_api.message = _message([BASH])

    [item] = anthropic_client.send(
        model="ant/claude-test", input="hi", tools=[FunctionTool(name="bash", description="My own bash")],
    ).output

    assert isinstance(item, FunctionCallOutputItem)
    assert item.arguments == {"command": "ls -la"}

def test_shell_result_text(anthropic_adapter):
    [block] = anthropic_adapter.map_shell_tool_result(ShellToolResult(
        call_id="toolu_b", output=[ShellCommandOutput(stdout="a.py", stderr="warning", exit_code=1)],
    ))["content"]

    assert block == {"type": "tool_result", "tool_use_id": "toolu_b", "content": "a.py\nwarning\nExit code: 1"}

@pytest.mark.parametrize(
    ("output", "text"),
    [
        ([], "tool has been restarted."),
        ([ShellCommandOutput()], "(no output)"),
        ([ShellCommandOutput(stdout="partial", timed_out=True)], "partial\nCommand timed out."),
        ([ShellCommandOutput(exit_code=127)], "Exit code: 127"),
    ],
)
def test_shell_result_edge_cases(anthropic_adapter, output, text):
    [block] = anthropic_adapter.map_shell_tool_result(ShellToolResult(call_id="toolu_b", output=output))["content"]

    assert block["content"] == text

def test_shell_error_result_sets_the_flag(anthropic_adapter):
    [block] = anthropic_adapter.map_shell_tool_result(ShellToolResult(
        call_id="toolu_b", output=[ShellCommandOutput(stderr="no shell")], is_error=True,
    ))["content"]

    assert block["is_error"] is True


# provider tool results

def test_provider_tool_result_block_goes_in_a_user_turn(anthropic_adapter):
    block = {"type": "tool_result", "tool_use_id": "toolu_e", "content": "edited"}

    assert anthropic_adapter.map_provider_tool_result(ProviderToolResult(provider="anthropic", item=block)) == {
        "role": "user", "content": [block],
    }

def test_provider_tool_result_message_is_sent_as_is(anthropic_adapter):
    message = {"role": "user", "content": [{"type": "text", "text": "hi"}]}

    assert anthropic_adapter.map_provider_tool_result(ProviderToolResult(provider="anthropic", item=message)) == message


# the agent loop

def test_batch_round_trip(anthropic_client, anthropic_api):
    click = _member("left_click", {"coordinate": [512, 742]}, id="toolu_1")
    shot = _member("screenshot", {}, id="toolu_2")
    anthropic_api.message = _message([{"type": "text", "text": "Opening the browser."}, click, shot, BASH])
    tools = [ComputerUseTool(), ShellTool()]
    history: list = ["Find a cat picture"]

    response = anthropic_client.send(model="ant/claude-test", input=history, tools=tools)
    history.extend(response.output)

    for item in response.output:
        if item.type == "computer_call":
            needs_image = any(action.type in ("screenshot", "zoom") for action in item.actions)
            history.append(ComputerToolResult(call_id=item.call_id, screenshot=PNG if needs_image else None))
        elif item.type == "shell_call":
            history.append(ShellToolResult(call_id=item.call_id, output=[ShellCommandOutput(stdout="cat.png")]))

    anthropic_api.message = _message([{"type": "text", "text": "Done."}], stop_reason="end_turn")
    anthropic_client.send(model="ant/claude-test", input=history, tools=tools)

    _, assistant, results = anthropic_api.last_body["messages"]
    assert [block.get("name", block["type"]) for block in assistant["content"]] == ["text", "left_click", "screenshot", "bash"]
    assert assistant["content"][1] == click
    assert [block["tool_use_id"] for block in results["content"]] == ["toolu_1", "toolu_2", "toolu_b"]
    assert [block.get("toolset_name") for block in results["content"]] == ["computer", "computer", None]
    assert results["content"][1]["content"][0]["type"] == "image"
    assert anthropic_api.last_body["tools"] == [
        {"type": "computer_toolset_20260801"},
        {"type": "bash_20250124", "name": "bash"},
    ]


# streaming

def test_stream_reports_client_tool_calls(anthropic_client, anthropic_api):
    anthropic_api.sse_events = [
        {"type": "message_start", "message": _message([], stop_reason=None)},
        {"type": "content_block_start", "index": 0, "content_block": {**_member("left_click", {}), "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"coordinate\": [1, 2]}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {**BASH, "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]

    events = list(anthropic_client.send(
        model="ant/claude-test", input="hi", tools=[ComputerUseTool(), ShellTool()], stream=True,
    ))

    started = [event for event in events if event.type == "tool_call.started"]
    assert [(event.call_id, event.name, event.call_type) for event in started] == [
        ("toolu_1", "computer", "computer_call"),
        ("toolu_b", "shell", "shell_call"),
    ]

    done = [event.item for event in events if event.type == "output_item.done"]
    assert done[0].actions == [actions.Click(x=1, y=2)]
    assert done[1].commands == ["ls"]

    [completed] = [event for event in events if event.type == "response.completed"]
    assert [item.type for item in completed.response.output] == ["computer_call", "shell_call"]

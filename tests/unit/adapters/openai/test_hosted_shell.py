from types import SimpleNamespace

import pytest
from openai.types.responses import (
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
)
from pydantic import ValidationError

from tessaract import ContainerEnvironment, ShellCommandOutput, ShellTool
from tessaract.types.output_types import ShellCallOutputItem
from tessaract.types.request import Request

CONTAINER = {"type": "container_reference", "container_id": "cntr_1"}


def _shell_call(call_id="call_1", commands=("ls", "python -V")):
    return ResponseFunctionShellToolCall.model_validate({
        "id": f"sh_{call_id}", "call_id": call_id, "type": "shell_call", "status": "completed",
        "action": {"commands": list(commands), "timeout_ms": 10_000, "max_output_length": 4000},
        "environment": CONTAINER,
    })

def _shell_output(call_id="call_1"):
    return ResponseFunctionShellToolCallOutput.model_validate({
        "id": f"sho_{call_id}", "call_id": call_id, "type": "shell_call_output", "status": "completed",
        "max_output_length": 4000,
        "output": [
            {"stdout": "data.csv\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
            {"stdout": "", "stderr": "", "outcome": {"type": "timeout"}},
        ],
    })


# the environment

def test_container_environment_defaults_to_a_new_container(openai_adapter):
    native = openai_adapter.map_shell_tool(ShellTool(environment=ContainerEnvironment()))

    assert native == {"type": "shell", "environment": {"type": "container_auto"}}

def test_new_container_options_are_mapped(openai_adapter):
    tool = ShellTool(environment=ContainerEnvironment(
        file_ids=["file_1"], memory_limit="4g", allowed_domains=["pypi.org"],
    ))

    assert openai_adapter.map_shell_tool(tool)["environment"] == {
        "type": "container_auto",
        "file_ids": ["file_1"],
        "memory_limit": "4g",
        "network_policy": {"type": "allowlist", "allowed_domains": ["pypi.org"]},
    }

def test_network_can_be_disabled(openai_adapter):
    native = openai_adapter.map_shell_tool(ShellTool(environment=ContainerEnvironment(network_disabled=True)))

    assert native["environment"]["network_policy"] == {"type": "disabled"}

def test_container_id_reuses_a_container(openai_adapter):
    native = openai_adapter.map_shell_tool(ShellTool(environment=ContainerEnvironment(container_id="cntr_1")))

    assert native["environment"] == CONTAINER

def test_native_environment_fields_are_kept(openai_adapter):
    secret = {"domain": "api.example.com", "name": "TOKEN", "value": "s3cret"}
    skills = [{"type": "skill_reference", "skill_id": "skill_1"}]
    tool = ShellTool(environment=ContainerEnvironment(
        allowed_domains=["api.example.com"],
        provider_options={"skills": skills, "network_policy": {"domain_secrets": [secret], "allowed_domains": ["x"]}},
    ))

    environment = openai_adapter.map_shell_tool(tool)["environment"]

    assert environment["skills"] == skills
    assert environment["network_policy"] == {
        "type": "allowlist", "allowed_domains": ["api.example.com"], "domain_secrets": [secret],
    }

def test_tool_provider_options_environment_fields_are_merged(openai_adapter):
    tool = ShellTool(environment=ContainerEnvironment(), provider_options={"environment": {"skills": [], "type": "local"}})

    # the typed environment's type wins
    assert openai_adapter.map_shell_tool(tool)["environment"] == {"type": "container_auto", "skills": []}

@pytest.mark.parametrize(
    "fields",
    [
        {"allowed_domains": ["a.com"], "network_disabled": True},
        {"container_id": "cntr_1", "memory_limit": "4g"},
        {"container_id": "cntr_1", "network_disabled": True},
    ],
)
def test_invalid_container_combinations_are_rejected(fields):
    with pytest.raises(ValidationError):
        ContainerEnvironment(**fields)

def test_container_shell_tool_needs_no_sources_include(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(
        model="gpt-test", input=[], tools=[ShellTool(environment=ContainerEnvironment())],
    ))

    assert kwargs["tools"] == [{"type": "shell", "environment": {"type": "container_auto"}}]
    assert "include" not in kwargs


# responses

def test_provider_shell_call_and_output_become_one_item(openai_adapter):
    call, output = _shell_call(), _shell_output()

    [item] = openai_adapter._normalize_output([call, output])

    assert isinstance(item, ShellCallOutputItem)
    assert (item.executed_by, item.provider, item.call_id, item.status) == ("provider", "openai", "call_1", "completed")
    assert item.commands == ["ls", "python -V"]
    assert item.output == [
        ShellCommandOutput(stdout="data.csv\n"),
        ShellCommandOutput(timed_out=True),
    ]
    assert (item.timeout_ms, item.max_output_length) == (10_000, 4000)
    assert item.raw == [call, output]

def test_provider_shell_call_is_replayed_as_both_items(openai_adapter, client):
    call, output = _shell_call(), _shell_output()
    [item] = openai_adapter._normalize_output([call, output])

    assert openai_adapter.map_output_item(item) == [call, output]

    request = client._build_request_model(
        model="gpt-test", provider="oai", input=["hi", item],
        reasoning=None, max_tokens=None, tools=[], request_options={}, stream=False,
    )
    assert request.input == [{"role": "user", "content": "hi"}, call, output]

def test_running_provider_shell_call_has_no_output_yet(openai_adapter):
    [item] = openai_adapter._normalize_output([_shell_call()])

    assert (item.executed_by, item.output) == ("provider", [])
    assert item.raw == [item.raw[0]]

def test_interleaved_provider_shell_calls_keep_native_order(openai_adapter):
    first, second = _shell_call("call_1"), _shell_call("call_2", commands=["pwd"])

    output = openai_adapter._normalize_output([first, second, _shell_output("call_1"), _shell_output("call_2")])

    assert [(item.call_id, len(item.raw), item.commands) for item in output] == [
        ("call_1", 1, ["ls", "python -V"]),
        ("call_2", 1, ["pwd"]),
        ("call_1", 1, ["ls", "python -V"]),
        ("call_2", 1, ["pwd"]),
    ]
    assert [len(item.output) for item in output] == [0, 0, 2, 2]

def test_local_shell_call_is_still_client_executed(openai_adapter):
    local = ResponseFunctionShellToolCall.model_validate({
        "id": "sh_1", "call_id": "call_1", "type": "shell_call", "status": "completed",
        "action": {"commands": ["ls"]}, "environment": {"type": "local"},
    })

    [item] = openai_adapter._normalize_output([local])

    assert item.executed_by == "client"
    assert item.raw is local


# streaming

def test_stream_reports_a_provider_shell_call(openai_adapter):
    call, output = _shell_call(), _shell_output()
    pending: dict = {}
    events = [
        SimpleNamespace(type="response.output_item.added", output_index=0, item=call),
        SimpleNamespace(type="response.output_item.done", output_index=0, item=call),
        SimpleNamespace(type="response.output_item.added", output_index=1, item=output),
        SimpleNamespace(type="response.output_item.done", output_index=1, item=output),
    ]

    normalized = [e for event in events for e in openai_adapter._normalize_stream_event(event, pending)]

    assert [event.type for event in normalized] == ["server_tool_call.started", "custom_provider_event", "output_item.done"]
    started = normalized[0]
    assert (started.call_id, started.name) == ("call_1", "shell")

    item = normalized[-1].item
    assert (item.executed_by, item.commands, len(item.output)) == ("provider", ["ls", "python -V"], 2)
    assert pending == {}

def test_response_names_the_container(openai_adapter):
    from openai.types.responses import Response as OpenAINativeResponse

    response = openai_adapter._normalize_response(OpenAINativeResponse.model_construct(
        id="resp_1", model="gpt-test", status="completed", incomplete_details=None, error=None, usage=None,
        output=[_shell_call(), _shell_output()],
    ))

    assert response.container_id == "cntr_1"

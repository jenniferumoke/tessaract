import json

import pytest

pytest.importorskip("anthropic")

from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

from tessaract import FunctionTool, FunctionToolResult, InputSchema, Property, ReasoningOptions
from tessaract.adapters.anthropic.anthropic_adapter import (
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    THINKING_BINDING_CONTROLS_BETA,
    THINKING_DISPLAY_UPDATES_BETA,
)
from tessaract.types.output_types import FunctionCallOutputItem, ReasoningOutputItem
from tessaract.types.request import Request


# --- reasoning ----------------------------------------------------------------

def test_no_reasoning_sends_nothing(anthropic_adapter):
    assert anthropic_adapter.map_reasoning_params(None) == {}

@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            ReasoningOptions(effort="medium", summary="auto"),
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "medium"}},
        ),
        (
            ReasoningOptions(effort="extra_high"),
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
        ),
        (
            ReasoningOptions(budget=4096, summary="omitted"),
            {"thinking": {"type": "enabled", "budget_tokens": 4096, "display": "omitted"}},
        ),
        (
            ReasoningOptions(mode="enabled", budget=2048, effort="max"),
            {"thinking": {"type": "enabled", "budget_tokens": 2048}, "output_config": {"effort": "max"}},
        ),
        (
            ReasoningOptions(mode="adaptive", summary="detailed"),
            {"thinking": {"type": "adaptive", "display": "summarized"}},
        ),
        (
            ReasoningOptions(mode="between_tools", effort="low"),
            {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
        ),
        (ReasoningOptions(mode="disabled"), {"thinking": {"type": "disabled"}}),
        (ReasoningOptions(effort="none"), {"thinking": {"type": "disabled"}}),
        (
            ReasoningOptions(mode="adaptive", summary="updates"),
            {"thinking": {"type": "adaptive", "display": "updates"}},
        ),
    ],
)
def test_reasoning_options_map_to_thinking_and_effort(anthropic_adapter, options, expected):
    assert anthropic_adapter.map_reasoning_params(options) == expected

def test_native_options_are_merged_into_thinking_and_typed_fields_win(anthropic_adapter):
    native = anthropic_adapter.map_reasoning_params(
        ReasoningOptions(
            mode="adaptive",
            native_options={"block_binding": {"prefix_mismatch_behavior": "drop_block"}, "type": "enabled"},
        )
    )

    assert native["thinking"] == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }

@pytest.mark.parametrize(
    ("options", "match"),
    [
        (ReasoningOptions(effort="minimal"), "does not support effort"),
        (ReasoningOptions(mode="pro"), "does not support mode"),
        (ReasoningOptions(mode="standard"), "does not support mode"),
        (ReasoningOptions(mode="enabled"), "requires a thinking budget"),
        (ReasoningOptions(budget=512), "at least 1024"),
        (ReasoningOptions(budget=DEFAULT_ANTHROPIC_MAX_TOKENS), "less than max_tokens"),
        (ReasoningOptions(mode="adaptive", budget=2048), "only used with mode=\"enabled\""),
        (ReasoningOptions(mode="between_tools", summary="auto"), "cannot be used with mode"),
        (ReasoningOptions(mode="disabled", budget=2048), "cannot be used with mode"),
        (ReasoningOptions(effort="none", mode="adaptive"), "turns thinking off"),
    ],
)
def test_invalid_reasoning_options_raise(anthropic_adapter, options, match):
    with pytest.raises(ValueError, match=match):
        anthropic_adapter.map_reasoning_params(options)

def test_budget_is_checked_against_explicit_max_tokens(anthropic_adapter):
    native = anthropic_adapter.map_reasoning_params(ReasoningOptions(budget=20_000), max_tokens=32_000)

    assert native["thinking"]["budget_tokens"] == 20_000


# --- tools --------------------------------------------------------------------

def test_tool_schema_is_strict_with_closed_objects(anthropic_adapter):
    tool = FunctionTool(
        name="get_weather",
        description="Weather for a city",
        input_schema=InputSchema(
            properties={"city": Property(type="string")},
            required=["city"]
        ),
    )

    [native] = anthropic_adapter.map_function_schema([tool])

    assert native == {
        "type": "custom",
        "name": "get_weather",
        "description": "Weather for a city",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False
        }
    }

def test_tool_without_arguments_gets_an_empty_input_schema(anthropic_adapter):
    [native] = anthropic_adapter.map_function_schema(
        [FunctionTool(name="get_time", description="Current time", strict=False)]
    )

    assert native["input_schema"] == {"type": "object", "properties": {}}
    assert native["strict"] is False

def test_provider_options_cannot_override_input_schema(anthropic_adapter):
    [native] = anthropic_adapter.map_function_schema([
        FunctionTool(
            name="t", description="d",
            provider_options={"input_schema": {"type": "string"}, "cache_control": {"type": "ephemeral"}},
        )
    ])

    assert native["input_schema"] == {"type": "object", "properties": {}}
    assert native["cache_control"] == {"type": "ephemeral"}


# --- tool results -------------------------------------------------------------

def test_string_tool_result_is_passed_through(anthropic_adapter):
    native = anthropic_adapter.map_tool_result(FunctionToolResult(call_id="toolu_1", result="19C"))

    assert native == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "19C"}],
    }

def test_dict_tool_result_is_json_encoded(anthropic_adapter):
    native = anthropic_adapter.map_tool_result(FunctionToolResult(call_id="toolu_1", result={"temp": 19}))

    assert json.loads(native["content"][0]["content"]) == {"temp": 19}

def test_error_tool_result_uses_native_error_flag(anthropic_adapter):
    native = anthropic_adapter.map_tool_result(
        FunctionToolResult(call_id="toolu_1", result="city not found", is_error=True)
    )

    [block] = native["content"]
    assert block["content"] == "city not found"
    assert block["is_error"] is True

def test_content_block_tool_result_is_not_json_encoded(anthropic_adapter):
    parts = [{"type": "text", "text": "19C"}]

    native = anthropic_adapter.map_tool_result(FunctionToolResult(call_id="toolu_1", result=parts))

    assert native["content"][0]["content"] == parts

def test_tool_result_provider_options_are_merged_but_cannot_override(anthropic_adapter):
    native = anthropic_adapter.map_tool_result(
        FunctionToolResult(
            call_id="toolu_1", result="ok",
            provider_options={"cache_control": {"type": "ephemeral"}, "tool_use_id": "sneaky"},
        )
    )

    [block] = native["content"]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["tool_use_id"] == "toolu_1"


# --- request building ---------------------------------------------------------

def _request(adapter, input_items, **kwargs):
    from tessaract import Tessaract

    client = Tessaract(providers={"ant": adapter._provider})
    return client._build_request_model(
        model="claude-test", provider="ant", input=input_items,
        reasoning=kwargs.pop("reasoning", None), max_tokens=kwargs.pop("max_tokens", None),
        tools=kwargs.pop("tools", []), request_options=kwargs.pop("request_options", {}),
        stream=False, instructions=kwargs.pop("instructions", None),
    )

def test_max_tokens_defaults_when_not_set(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(Request(model="claude-test", input=[]))

    assert kwargs["max_tokens"] == DEFAULT_ANTHROPIC_MAX_TOKENS == 16_000

def test_explicit_max_tokens_is_sent(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(Request(model="claude-test", input=[], max_tokens=1234))

    assert kwargs["max_tokens"] == 1234

def test_empty_tools_are_left_out(anthropic_adapter):
    kwargs = anthropic_adapter._build_request_kwargs(Request(model="claude-test", input=[]))

    assert "tools" not in kwargs

def test_system_prompts_and_instructions_move_to_system(anthropic_adapter):
    from tessaract import SystemPrompt

    cached = {"type": "text", "text": "Long context", "cache_control": {"type": "ephemeral"}}
    request = _request(
        anthropic_adapter,
        [SystemPrompt(content="Be brief."), SystemPrompt(content=[cached]), "hi"],
        instructions="You are helpful.",
    )

    kwargs = anthropic_adapter._build_request_kwargs(request)

    assert kwargs["system"] == [
        {"type": "text", "text": "You are helpful."},
        {"type": "text", "text": "Be brief."},
        cached,
    ]
    assert kwargs["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

def test_instructions_request_option_is_rejected_with_a_pointer(anthropic_adapter):
    request = Request(model="claude-test", input=[], provider_options={"instructions": "x"})

    with pytest.raises(ValueError, match="instructions="):
        anthropic_adapter._build_request_kwargs(request)

def test_tool_loop_history_is_merged_into_alternating_turns(anthropic_adapter):
    thinking = ThinkingBlock(type="thinking", thinking="Need weather for both.", signature="sig_1")
    call_1 = ToolUseBlock(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris"})
    call_2 = ToolUseBlock(type="tool_use", id="toolu_2", name="get_weather", input={"city": "Lagos"})

    history = [
        "Weather in Paris and Lagos?",
        ReasoningOutputItem(raw=thinking, text=thinking.thinking),
        FunctionCallOutputItem(raw=call_1, call_id="toolu_1", name="get_weather", arguments={"city": "Paris"}),
        FunctionCallOutputItem(raw=call_2, call_id="toolu_2", name="get_weather", arguments={"city": "Lagos"}),
        FunctionToolResult(call_id="toolu_1", result="19C"),
        FunctionToolResult(call_id="toolu_2", result="31C"),
    ]

    kwargs = anthropic_adapter._build_request_kwargs(_request(anthropic_adapter, history))

    assert kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Paris and Lagos?"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Need weather for both.", "signature": "sig_1"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "Lagos"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "19C"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "31C"},
        ]},
    ]

def test_tool_results_come_before_text_in_a_merged_user_turn(anthropic_adapter):
    history = [FunctionToolResult(call_id="toolu_1", result="19C")]
    request = _request(anthropic_adapter, history)
    request.input.insert(0, {"role": "user", "content": "And tomorrow?"})

    kwargs = anthropic_adapter._build_request_kwargs(request)

    [turn] = kwargs["messages"]
    assert [block["type"] for block in turn["content"]] == ["tool_result", "text"]

def test_assistant_text_replays_as_text_blocks(anthropic_adapter):
    from tessaract.types.output_types import AssistantMessage, TextOutputItem

    block = TextBlock(type="text", text="Hello")
    message = AssistantMessage(raw=[block], content=[TextOutputItem(raw=block, text="Hello")])

    kwargs = anthropic_adapter._build_request_kwargs(_request(anthropic_adapter, ["hi", message]))

    assert kwargs["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}

def test_merging_does_not_mutate_the_callers_input(anthropic_adapter):
    content = [{"type": "text", "text": "a"}]
    request = Request(model="claude-test", input=[
        {"role": "user", "content": content},
        {"role": "user", "content": "b"},
    ])

    anthropic_adapter._build_request_kwargs(request)

    assert content == [{"type": "text", "text": "a"}]

def test_reasoning_is_sent_and_output_config_keeps_native_fields(anthropic_adapter):
    request = Request(
        model="claude-test", input=[],
        reasoning=ReasoningOptions(effort="high"),
        provider_options={"output_config": {"effort": "low", "format": {"type": "json_schema"}}},
    )

    kwargs = anthropic_adapter._build_request_kwargs(request)

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high", "format": {"type": "json_schema"}}

def test_output_config_from_request_options_passes_through_without_effort(anthropic_adapter):
    request = Request(model="claude-test", input=[], provider_options={"output_config": {"format": {"type": "json_schema"}}})

    kwargs = anthropic_adapter._build_request_kwargs(request)

    assert kwargs["output_config"] == {"format": {"type": "json_schema"}}

def test_canonical_request_params_win_over_provider_options(anthropic_adapter):
    request = Request(
        model="claude-test", input=[],
        provider_options={"model": "other", "temperature": 0.2, "cache_control": {"type": "ephemeral"}}
    )

    kwargs = anthropic_adapter._build_request_kwargs(request)

    assert kwargs["model"] == "claude-test"
    assert kwargs["temperature"] == 0.2
    assert kwargs["cache_control"] == {"type": "ephemeral"}


# --- beta routing -------------------------------------------------------------

def test_ga_features_use_the_messages_api(anthropic_adapter):
    api, kwargs = anthropic_adapter._messages_api({"thinking": {"type": "adaptive", "display": "summarized"}})

    assert api is anthropic_adapter._client.messages
    assert "betas" not in kwargs

def test_beta_thinking_features_use_the_beta_api_with_headers(anthropic_adapter):
    api, kwargs = anthropic_adapter._messages_api({
        "thinking": {"type": "adaptive", "display": "updates", "block_binding": {"prefix_mismatch_behavior": "error"}},
        "betas": ["context-1m-2025-08-07", THINKING_DISPLAY_UPDATES_BETA],
    })

    assert api is anthropic_adapter._client.beta.messages
    assert kwargs["betas"] == ["context-1m-2025-08-07", THINKING_DISPLAY_UPDATES_BETA, THINKING_BINDING_CONTROLS_BETA]

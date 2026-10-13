import json

import pytest
from openai import Omit

from tessaract import FunctionTool, FunctionToolResult, InputSchema, Property, ReasoningOptions
from tessaract.types.request import Request


def test_no_reasoning_is_omitted(openai_adapter):
    assert isinstance(openai_adapter.map_reasoning_params(None), Omit)

def test_extra_high_reasoning_param_is_converted_to_xhigh(openai_adapter):
    native = openai_adapter.map_reasoning_params(ReasoningOptions("extra_high"))

    assert native == {"effort": "xhigh"}

def test_only_set_reasoning_fields_are_sent(openai_adapter):
    native = openai_adapter.map_reasoning_params(ReasoningOptions(summary="auto"))

    assert native == {"summary": "auto"}

def test_string_tool_result_is_passed_through(openai_adapter):
    native = openai_adapter.map_tool_result(FunctionToolResult(call_id="c1", result="19C"))
    assert native == {"type": "function_call_output", "call_id": "c1", "output": "19C"}

def test_dict_tool_result_is_json_encoded(openai_adapter):
    native = openai_adapter.map_tool_result(FunctionToolResult(call_id="c1", result={"temp": 19}))
    assert json.loads(native["output"]) == {"temp": 19}

def test_error_tool_result_is_wrapped_in_error_key(openai_adapter):
    native = openai_adapter.map_tool_result(
        FunctionToolResult(call_id="c1", result="city not found", is_error=True)
    )
    assert json.loads(native["output"]) == {"error": "city not found"}

def test_tool_schema_defaults_to_strict_and_closes_objects(openai_adapter):
    tool = FunctionTool(
        name="get_weather",
        description="Weather for a city",
        input_schema=InputSchema(
            properties={"city": Property(type="string")},
            required=["city"]
        ),
    )

    [native] = openai_adapter.map_function_schema([tool])

    assert native == {
        "type": "function",
        "name": "get_weather",
        "description": "Weather for a city",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False
        }
    }

def test_provider_options_cannot_override_canonical_tool_keys(openai_adapter):
    tool = FunctionTool(
        name="real_name", description="d",
        provider_options={"name": "sneaky", "cache_control": "x"},
    )
    [native] = openai_adapter.map_function_schema([tool])
    assert native["name"] == "real_name"
    assert native["cache_control"] == "x"

def test_canonical_request_params_win_over_provider_options(openai_adapter):
    request = Request(
        model="gpt-test", input=[],
        provider_options={"model": "other", "temperature": 0.2}
    )

    kwargs = openai_adapter._build_request_kwargs(request)
    assert kwargs["model"] == "gpt-test"
    assert kwargs["temperature"] == 0.2

def test_content_part_tool_result_is_not_json_encoded(openai_adapter):
    parts = [{"type": "input_text", "text": "19C"}]

    native = openai_adapter.map_tool_result(FunctionToolResult(call_id="c1", result=parts))
    assert native["output"] == parts

def test_max_tokens_is_sent_as_max_output_tokens(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(model="gpt-test", input=[], max_tokens=500))
    assert kwargs["max_output_tokens"] == 500

def test_unset_max_tokens_and_instructions_are_not_sent(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(model="gpt-test", input=[]))
    assert "max_output_tokens" not in kwargs
    assert "instructions" not in kwargs

def test_instructions_are_sent_as_native_instructions(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(Request(model="gpt-test", input=[], instructions="Be brief."))
    assert kwargs["instructions"] == "Be brief."

def test_instructions_request_option_still_passes_through(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(
        Request(model="gpt-test", input=[], provider_options={"instructions": "legacy"})
    )
    assert kwargs["instructions"] == "legacy"

def test_canonical_instructions_win_over_request_option(openai_adapter):
    kwargs = openai_adapter._build_request_kwargs(
        Request(model="gpt-test", input=[], instructions="typed", provider_options={"instructions": "legacy"})
    )
    assert kwargs["instructions"] == "typed"

def test_reasoning_native_options_are_merged_and_typed_fields_win(openai_adapter):
    native = openai_adapter.map_reasoning_params(
        ReasoningOptions(effort="low", native_options={"effort": "high", "generate_summary": "auto"})
    )
    assert native == {"effort": "low", "generate_summary": "auto"}

def test_tool_result_provider_options_are_merged_but_cannot_override(openai_adapter):
    native = openai_adapter.map_tool_result(
        FunctionToolResult(call_id="c1", result="ok", provider_options={"status": "completed", "call_id": "sneaky"})
    )
    assert native == {"type": "function_call_output", "call_id": "c1", "output": "ok", "status": "completed"}

def test_unknown_output_items_pass_through(openai_adapter):
    from types import SimpleNamespace

    from tessaract.types.output_types import ProviderOutputItem

    item = SimpleNamespace(type="web_search_call")
    result = openai_adapter._normalize_output_item(item)

    assert isinstance(result, ProviderOutputItem)
    assert result.provider_type == "web_search_call"
    assert result.raw is item

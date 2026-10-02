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
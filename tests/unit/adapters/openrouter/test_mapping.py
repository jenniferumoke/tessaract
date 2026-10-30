import pytest

pytest.importorskip("openrouter")

from tessaract import (
    CacheOptions,
    ComputerUseTool,
    FunctionTool,
    FunctionToolResult,
    InputSchema,
    Property,
    ProviderTool,
    ProviderToolResult,
    ReasoningOptions,
    ShellTool,
    SystemPrompt,
    UserMessage,
    WebSearchTool,
)
from tests.fixtures import openrouter as payloads

MODEL = "or/openai/gpt-5-mini"

WEATHER = FunctionTool(
    name="get_weather",
    description="Get the current temperature for a city.",
    input_schema=InputSchema(
        properties={
            "city": Property(type="string", description="City name."),
            "units": Property(type="string", enum=["c", "f"]),
        },
        required=["city", "units"],
    ),
)


@pytest.fixture(autouse=True)
def _serve_a_message(openrouter_api):
    openrouter_api.responses = [payloads.response([payloads.message("ok")])]


def test_model_keeps_the_openrouter_vendor_prefix(openrouter_client, openrouter_api):
    openrouter_client.send(model="or/anthropic/claude-sonnet-5.5", input="hi")

    assert openrouter_api.last_body["model"] == "anthropic/claude-sonnet-5.5"


def test_messages_and_instructions(openrouter_client, openrouter_api):
    openrouter_client.send(
        model=MODEL,
        input=[SystemPrompt(content="Be brief."), "hi", UserMessage(content=[{"type": "input_text", "text": "there"}])],
        instructions="Answer in French.",
        max_tokens=256,
    )

    body = openrouter_api.last_body
    assert body["input"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": [{"type": "input_text", "text": "there"}]},
    ]
    assert body["instructions"] == "Answer in French."
    assert body["max_output_tokens"] == 256
    # OpenRouter keeps no state, so the SDK always sends store: false
    assert body["store"] is False


def test_encrypted_reasoning_is_always_requested(openrouter_client, openrouter_api):
    openrouter_client.send(model=MODEL, input="hi", request_options={"include": ["file_search_call.results"]})

    assert openrouter_api.last_body["include"] == ["file_search_call.results", "reasoning.encrypted_content"]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (ReasoningOptions(effort="low", summary="auto"), {"effort": "low", "summary": "auto"}),
        (ReasoningOptions(effort="extra_high"), {"effort": "xhigh"}),
        (ReasoningOptions(effort="max", mode="pro"), {"effort": "max", "mode": "pro"}),
        (ReasoningOptions(mode="enabled", budget=2048), {"enabled": True, "max_tokens": 2048}),
        (ReasoningOptions(mode="disabled"), {"enabled": False}),
        # native_options add fields; the typed fields win
        (
            ReasoningOptions(effort="high", native_options={"effort": "low", "context": "all_turns"}),
            {"effort": "high", "context": "all_turns"},
        ),
    ],
)
def test_reasoning_options(openrouter_client, openrouter_api, options, expected):
    openrouter_client.send(model=MODEL, input="hi", reasoning=options)

    assert openrouter_api.last_body["reasoning"] == expected


def test_reasoning_is_left_out_when_not_set(openrouter_client, openrouter_api):
    openrouter_client.send(model=MODEL, input="hi")

    assert "reasoning" not in openrouter_api.last_body


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (ReasoningOptions(mode="adaptive"), "mode"),
        (ReasoningOptions(mode="between_tools"), "mode"),
        (ReasoningOptions(summary="omitted"), "summary"),
    ],
)
def test_unsupported_reasoning_options_raise(openrouter_client, options, match):
    with pytest.raises(ValueError, match=match):
        openrouter_client.send(model=MODEL, input="hi", reasoning=options)


def test_function_tools_are_strict_by_default(openrouter_client, openrouter_api):
    openrouter_client.send(model=MODEL, input="hi", tools=[WEATHER])

    assert openrouter_api.last_body["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current temperature for a city.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                    "units": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city", "units"],
                "additionalProperties": False,
            },
        }
    ]


@pytest.mark.parametrize(
    ("result", "is_error", "expected_output"),
    [
        ("19C", False, "19C"),
        ({"temperature": 19}, False, '{"temperature": 19}'),
        ([{"type": "input_text", "text": "19C"}], False, [{"type": "input_text", "text": "19C"}]),
        ("city not found", True, '{"error": "city not found"}'),
    ],
)
def test_function_tool_results(openrouter_client, openrouter_api, result, is_error, expected_output):
    openrouter_client.send(
        model=MODEL, input=[FunctionToolResult(call_id="call_1", result=result, is_error=is_error)]
    )

    assert openrouter_api.last_body["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": expected_output}
    ]


def test_provider_tools_and_results_are_sent_unchanged(openrouter_client, openrouter_api):
    datetime_tool = ProviderTool(provider="openrouter", definition={"type": "openrouter:datetime"})
    result = ProviderToolResult(
        provider="openrouter", item={"type": "function_call_output", "call_id": "call_9", "output": "done"}
    )

    openrouter_client.send(model=MODEL, input=[result], tools=[datetime_tool])

    body = openrouter_api.last_body
    assert body["tools"] == [{"type": "openrouter:datetime"}]
    assert body["input"] == [{"type": "function_call_output", "call_id": "call_9", "output": "done"}]


def test_provider_tools_for_another_provider_raise(openrouter_client):
    tool = ProviderTool(provider="openai", definition={"type": "image_generation"})

    with pytest.raises(ValueError, match="'openai'"):
        openrouter_client.send(model=MODEL, input="hi", tools=[tool])


def test_request_options_are_sent_and_canonical_parameters_win(openrouter_client, openrouter_api):
    openrouter_client.send(
        model=MODEL,
        input="hi",
        max_tokens=100,
        request_options={
            "provider": {"sort": "price"},
            "models": ["anthropic/claude-haiku-4.5"],
            "max_output_tokens": 5,
        },
    )

    body = openrouter_api.last_body
    assert body["provider"] == {"sort": "price"}
    assert body["models"] == ["anthropic/claude-haiku-4.5"]
    assert body["max_output_tokens"] == 100


@pytest.mark.parametrize(
    "tool",
    [WebSearchTool(), ComputerUseTool(display_width=1024, display_height=768), ShellTool()],
    ids=["web_search", "computer", "shell"],
)
def test_canonical_server_and_client_tools_are_not_supported_yet(openrouter_client, tool):
    with pytest.raises(NotImplementedError, match="OpenRouter adapter"):
        openrouter_client.send(model=MODEL, input="hi", tools=[tool])


def test_streaming_is_not_supported_yet(openrouter_client):
    with pytest.raises(NotImplementedError, match="Streaming"):
        openrouter_client.send(model=MODEL, input="hi", stream=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache": CacheOptions(mode="implicit")},
        {"input": [UserMessage(content="hi", cache=True)]},
        {"input": [FunctionToolResult(call_id="call_1", result="ok", cache=True)]},
    ],
    ids=["cache_options", "cached_message", "cached_tool_result"],
)
def test_prompt_caching_is_not_supported_yet(openrouter_client, kwargs):
    with pytest.raises(NotImplementedError, match="caching"):
        openrouter_client.send(**{"model": MODEL, "input": "hi", **kwargs})

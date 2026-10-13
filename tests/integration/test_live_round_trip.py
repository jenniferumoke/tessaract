"""Live round trips against the real APIs. Skipped by default; run with:

    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... uv run --all-extras pytest -m live

Override the models with TESSARACT_LIVE_ANTHROPIC_MODEL / TESSARACT_LIVE_OPENAI_MODEL.
"""

import os

import pytest

from tessaract import (
    FunctionTool,
    FunctionToolResult,
    InputSchema,
    Property,
    ReasoningOptions,
    Tessaract,
)

pytestmark = pytest.mark.live

PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ant/" + os.environ.get("TESSARACT_LIVE_ANTHROPIC_MODEL", "claude-sonnet-5-5")),
    "openai": ("OPENAI_API_KEY", "oai/" + os.environ.get("TESSARACT_LIVE_OPENAI_MODEL", "gpt-5.6-luna")),
}

WEATHER = FunctionTool(
    name="get_weather",
    description="Get the current temperature for a city.",
    input_schema=InputSchema(
        properties={"city": Property(type="string", description="City name, e.g. Paris.")},
        required=["city"],
        additionalProperties=False,
    ),
    strict=True,
)

REASONING = ReasoningOptions(effort="low", summary="auto")
INSTRUCTIONS = "Always use the get_weather tool for weather questions. Answer in one short sentence."
QUESTION = "What's the temperature in Paris right now?"


@pytest.fixture(params=list(PROVIDERS))
def live(request):
    env_var, model = PROVIDERS[request.param]
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} is not set")

    if request.param == "anthropic":
        pytest.importorskip("anthropic")
        from tessaract import AnthropicProvider
        client = Tessaract(providers={"ant": AnthropicProvider()})
    else:
        pytest.importorskip("openai")
        from tessaract import OpenAIProvider
        client = Tessaract(providers={"oai": OpenAIProvider()})

    return client, model


def _send(client, model, history, **kwargs):
    return client.send(
        model=model, input=history, tools=[WEATHER], reasoning=REASONING,
        instructions=INSTRUCTIONS, max_tokens=4096, **kwargs,
    )

def _answer_calls(history, calls):
    for call in calls:
        assert call.name == "get_weather"
        assert call.arguments["city"].lower() == "paris"
        history.append(FunctionToolResult(call_id=call.call_id, result={"city": "Paris", "temperature": "19C"}))


def test_reasoning_tool_call_round_trip(live):
    client, model = live
    history: list = [QUESTION]

    response = _send(client, model, history)
    calls = [item for item in response.output if item.type == "function_call"]
    assert calls, f"expected a function call, got {response.output}"
    assert response.usage is not None and response.usage.input_tokens > 0

    history.extend(response.output)
    _answer_calls(history, calls)

    response = _send(client, model, history)
    assert "19" in response.output_text


def test_streaming_reasoning_tool_call_round_trip(live):
    client, model = live
    history: list = [QUESTION]

    events = list(_send(client, model, history, stream=True))
    types = {event.type for event in events}
    assert {"response_started", "tool_call.started", "output_item.done", "response.completed"} <= types

    [completed] = [event for event in events if event.type == "response.completed"]
    calls = [item for item in completed.response.output if item.type == "function_call"]
    assert calls

    history.extend(completed.response.output)
    _answer_calls(history, calls)

    events = list(_send(client, model, history, stream=True))
    text = "".join(event.delta for event in events if event.type == "text.delta")
    assert "19" in text

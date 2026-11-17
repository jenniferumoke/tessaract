"""Live OpenRouter round trips. Skipped by default; run with:

    OPENROUTER_API_KEY=... uv run --all-extras pytest -m live tests/integration/test_live_openrouter.py

Reasoning comes back differently depending on the model OpenRouter routes to, so each test runs
against an OpenAI model (encrypted reasoning), an Anthropic model (signed thinking) and an open-weights
model (plain reasoning text). Override the models with TESSARACT_LIVE_OPENROUTER_OPENAI_MODEL,
TESSARACT_LIVE_OPENROUTER_ANTHROPIC_MODEL and TESSARACT_LIVE_OPENROUTER_OPEN_MODEL.
"""

import os

import pytest

from tessaract import (
    ContainerEnvironment,
    FunctionTool,
    FunctionToolResult,
    InputSchema,
    Property,
    ReasoningOptions,
    ShellTool,
    Tessaract,
    WebSearchTool,
)

pytestmark = pytest.mark.live

MODELS = {
    "openai": os.environ.get("TESSARACT_LIVE_OPENROUTER_OPENAI_MODEL", "openai/gpt-5-mini"),
    "anthropic": os.environ.get("TESSARACT_LIVE_OPENROUTER_ANTHROPIC_MODEL", "anthropic/claude-haiku-4.5"),
    "open": os.environ.get("TESSARACT_LIVE_OPENROUTER_OPEN_MODEL", "qwen/qwen3-235b-a22b-thinking-2507"),
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

INSTRUCTIONS = "Always use the get_weather tool for weather questions. Answer in one short sentence."
QUESTION = "What's the temperature in Paris right now?"


@pytest.fixture(params=list(MODELS))
def live(request):
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")

    pytest.importorskip("openrouter")
    from tessaract import OpenRouterProvider

    client = Tessaract(providers={"or": OpenRouterProvider(app_title="Tessaract live tests")})
    return client, "or/" + MODELS[request.param], request.param


def _reasoning(family: str) -> ReasoningOptions:
    # Anthropic models take a budget; the others take an effort
    if family == "anthropic":
        return ReasoningOptions(mode="enabled", budget=1024)
    return ReasoningOptions(effort="low", summary="auto")


def _send(client, model, family, history, stream=False):
    return client.send(
        model=model, input=history, tools=[WEATHER], reasoning=_reasoning(family),
        instructions=INSTRUCTIONS, max_tokens=4096, stream=stream,
    )


def test_reasoning_tool_call_round_trip(live):
    client, model, family = live
    history: list = [QUESTION]

    response = _send(client, model, family, history)

    calls = [item for item in response.output if item.type == "function_call"]
    assert calls, f"expected a function call, got {response.output}"
    assert response.usage is not None and response.usage.input_tokens > 0
    assert response.cost is not None

    reasoning = [item for item in response.output if item.type == "reasoning"]
    assert reasoning, f"expected reasoning from {model}, got {response.output}"
    if family == "openai":
        assert any(getattr(item.raw, "encrypted_content", None) for item in reasoning)
    elif family == "anthropic":
        assert any(getattr(item.raw, "signature", None) for item in reasoning)

    history.extend(response.output)
    for call in calls:
        assert call.name == "get_weather"
        assert call.arguments["city"].lower() == "paris"
        history.append(FunctionToolResult(call_id=call.call_id, result={"city": "Paris", "temperature": "19C"}))

    # the replayed reasoning has to be accepted by the model's upstream provider
    response = _send(client, model, family, history)
    assert "19" in response.output_text


def test_streaming_reasoning_tool_call_round_trip(live):
    client, model, family = live
    history: list = [QUESTION]

    events = list(_send(client, model, family, history, stream=True))
    types = {event.type for event in events}
    assert {"response_started", "reasoning.started", "tool_call.started", "tool_arguments.delta", "output_item.done"} <= types
    assert events[-1].type == "response.completed", f"the stream ended with {events[-1]}"

    response = events[-1].response
    assert response.usage is not None and response.usage.input_tokens > 0
    calls = [item for item in response.output if item.type == "function_call"]
    assert calls, f"expected a function call, got {response.output}"

    history.extend(response.output)
    for call in calls:
        history.append(FunctionToolResult(call_id=call.call_id, result={"city": "Paris", "temperature": "19C"}))

    events = list(_send(client, model, family, history, stream=True))
    assert events[-1].type == "response.completed", f"the stream ended with {events[-1]}"
    assert "19" in "".join(event.delta for event in events if event.type == "text.delta")


def _server_tool_turn(client, model, tool, question, stream):
    """Runs a question with a server tool, then a follow-up that replays its output."""
    history: list = [question]

    def send():
        result = client.send(model=model, input=history, tools=[tool], max_tokens=4096, stream=stream)
        if not stream:
            return result, []
        events = list(result)
        assert events[-1].type == "response.completed", f"the stream ended with {events[-1]}"
        return events[-1].response, events

    response, events = send()
    history.extend(response.output)
    history.append("Now say only the word done.")
    follow_up, _ = send()
    assert follow_up.output_text

    return response, events


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_web_search_round_trip(live, stream):
    client, model, _ = live

    response, events = _server_tool_turn(
        client, model, WebSearchTool(max_uses=1), "Search the web: what is the latest Python release?", stream,
    )

    assert any(item.type == "web_search_call" for item in response.output), response.output
    assert response.usage is not None and response.usage.server_tool_requests["web_search"] >= 1
    if stream:
        assert any(event.type == "server_tool_call.started" and event.name == "web_search" for event in events)


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_container_shell_round_trip(live, stream):
    client, model, _ = live

    response, events = _server_tool_turn(
        client, model, ShellTool(environment=ContainerEnvironment()), "Use the shell to run: echo tessaract", stream,
    )

    calls = [item for item in response.output if item.type == "shell_call"]
    assert calls, response.output
    assert calls[0].executed_by == "provider"
    assert any("tessaract" in output.stdout for call in calls for output in call.output)
    assert response.container_id
    if stream:
        assert any(event.type == "server_tool_call.started" and event.name == "shell" for event in events)

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
    FunctionTool,
    FunctionToolResult,
    InputSchema,
    Property,
    ReasoningOptions,
    Tessaract,
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


def _send(client, model, family, history):
    return client.send(
        model=model, input=history, tools=[WEATHER], reasoning=_reasoning(family),
        instructions=INSTRUCTIONS, max_tokens=4096,
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

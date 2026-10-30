'''OpenRouter Responses API payloads, shaped like what the API returns.

    Reasoning differs by the model OpenRouter routes to: OpenAI models return encrypted_content,
    Anthropic models return the thinking text with a signature, and many open models return plain
    reasoning text only. The payloads below cover each.
'''

from typing import Any

WEATHER_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city":"Paris"}',
    "status": "completed",
}

OPENAI_REASONING = {
    "type": "reasoning",
    "id": "rs_openai",
    "summary": [{"type": "summary_text", "text": "Need the weather tool."}],
    "encrypted_content": "gAAAAB-encrypted",
    "format": "openai-responses-v1",
}

ANTHROPIC_REASONING = {
    "type": "reasoning",
    "id": "rs_anthropic",
    "summary": [],
    "content": [{"type": "reasoning_text", "text": "The user wants the weather in Paris."}],
    "signature": "EqQBCkYIBxgCKkA-signature",
    "format": "anthropic-claude-v1",
}

OPEN_MODEL_REASONING = {
    "type": "reasoning",
    "id": "rs_open",
    "summary": [],
    "content": [{"type": "reasoning_text", "text": "I should call get_weather."}],
    "format": "unknown",
}


def message(text: str, annotations: list[dict] | None = None) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": annotations or []}],
    }


def usage(**overrides) -> dict[str, Any]:
    return {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens": 40,
        "output_tokens_details": {"reasoning_tokens": 25},
        "total_tokens": 160,
        "cost": 0.00042,
        **overrides,
    }


def response(output: list[dict], **overrides) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "response",
        "created_at": 1761800000,
        "completed_at": 1761800002,
        "model": "openai/gpt-5-mini",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "temperature": None,
        "top_p": None,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": usage(),
        **overrides,
    }

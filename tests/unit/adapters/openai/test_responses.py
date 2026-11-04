import pytest
from openai.types.responses import Response as OpenAINativeResponse

from tessaract.types.response import IncompleteReason, OpenAIResponse, ResponseStatus


def _response(status, incomplete_details=None):
    return OpenAINativeResponse.model_construct(
        id="resp_1",
        model="gpt-test",
        status=status,
        incomplete_details=incomplete_details,
        output=[],
        error=None,
        usage=None,
    )


@pytest.mark.parametrize(
    ("status", "incomplete_details", "expected_status", "expected_reason"),
    [
        ("completed", None, ResponseStatus.COMPLETED, None),
        ("incomplete", {"reason": "max_output_tokens"}, ResponseStatus.INCOMPLETE, IncompleteReason.MAX_TOKENS),
        ("incomplete", {"reason": "content_filter"}, ResponseStatus.INCOMPLETE, IncompleteReason.CONTENT_FILTER),
        ("incomplete", None, ResponseStatus.INCOMPLETE, None),
        ("failed", None, ResponseStatus.FAILED, None),
        (None, None, None, None),
    ],
)
def test_status_and_incomplete_reason_are_normalized(
    openai_adapter, status, incomplete_details, expected_status, expected_reason
):
    from openai.types.responses.response import IncompleteDetails

    details = IncompleteDetails.model_validate(incomplete_details) if incomplete_details else None

    response = openai_adapter._normalize_response(_response(status, details))

    assert isinstance(response, OpenAIResponse)
    assert response.status is expected_status
    assert response.incomplete_reason is expected_reason


def test_incomplete_stream_ends_as_completed():
    '''Through the real SDK stream, since it handles some events itself before they reach the adapter.'''
    import json

    import httpx

    from tessaract import OpenAIProvider, Tessaract

    def native_response(status, incomplete_details=None):
        return {
            "id": "resp_1",
            "object": "response",
            "created_at": 1761800000,
            "model": "gpt-test",
            "status": status,
            "incomplete_details": incomplete_details,
            "error": None,
            "instructions": None,
            "metadata": {},
            "output": [],
            "parallel_tool_calls": True,
            "temperature": 1,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
        }

    events = [
        {"type": "response.created", "sequence_number": 0, "response": native_response("in_progress")},
        {
            "type": "response.incomplete",
            "sequence_number": 1,
            "response": native_response("incomplete", {"reason": "max_output_tokens"}),
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = Tessaract(providers={"oai": OpenAIProvider(api_key="test", provider_args={"http_client": http_client})})

    started, completed = list(client.send(model="oai/gpt-test", input="hi", stream=True))

    assert started.type == "response_started"
    assert completed.type == "response.completed"
    assert completed.response.status is ResponseStatus.INCOMPLETE
    assert completed.response.incomplete_reason is IncompleteReason.MAX_TOKENS

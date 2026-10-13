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

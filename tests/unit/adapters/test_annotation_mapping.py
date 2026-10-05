import pytest
from openai.types.responses import ResponseOutputMessage

from tessaract.types.output_types import AssistantMessage, ProviderOutputItem


@pytest.mark.parametrize(
    ("native_annotation", "expected_type", "source", "title"),
    [
        (
            {"type": "url_citation", "start_index": 0, "end_index": 6,
             "title": "Example", "url": "https://example.com"},
            "citation", "https://example.com", "Example",
        ),
        (
            {"type": "file_citation", "file_id": "file_1", "filename": "facts.pdf", "index": 0},
            "citation", "file_1", "facts.pdf",
        ),
        (
            {"type": "container_file_citation", "container_id": "ctr_1",
             "start_index": 0, "end_index": 6, "file_id": "file_2", "filename": "data.csv"},
            "citation", "file_2", "data.csv",
        ),
        (
            {"type": "file_path", "file_id": "file_3", "index": 0},
            "file_path", "file_3", None,
        ),
    ],
)
def test_openai_annotations_become_normalized_text(
    openai_adapter, native_annotation, expected_type, source, title
):
    message = ResponseOutputMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "Source", "annotations": [native_annotation]}],
    })

    result = openai_adapter._normalize_output_item(message)

    assert isinstance(result, AssistantMessage)
    [text] = result.content
    [annotation] = text.annotations
    assert text.text == "Source"
    assert annotation.provider == "openai"
    assert annotation.provider_type == native_annotation["type"]
    assert annotation.type == expected_type
    assert annotation.source == source
    assert annotation.title == title
    assert annotation.provider_metadata == native_annotation


def test_openai_refusal_still_uses_provider_output(openai_adapter):
    message = ResponseOutputMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "refusal", "refusal": "Cannot help"}],
    })

    assert isinstance(openai_adapter._normalize_output_item(message), ProviderOutputItem)


@pytest.fixture
def anthropic_adapter():
    pytest.importorskip("anthropic")
    from tessaract.adapters.anthropic.anthropic_adapter import AnthropicAdapter
    from tessaract.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-api-key")
    yield AnthropicAdapter(provider)
    provider.client.close()


@pytest.mark.parametrize(
    ("native_citation", "source", "title"),
    [
        (
            {"type": "char_location", "cited_text": "Source", "document_index": 0,
             "document_title": "Notes", "start_char_index": 0, "end_char_index": 6,
             "file_id": "file_1"},
            "file_1", "Notes",
        ),
        (
            {"type": "page_location", "cited_text": "Source", "document_index": 0,
             "document_title": "Report", "start_page_number": 1, "end_page_number": 1,
             "file_id": "file_2"},
            "file_2", "Report",
        ),
        (
            {"type": "content_block_location", "cited_text": "Source", "document_index": 0,
             "document_title": "Document", "start_block_index": 0, "end_block_index": 1,
             "file_id": "file_3"},
            "file_3", "Document",
        ),
        (
            {"type": "web_search_result_location", "cited_text": "Source",
             "encrypted_index": "encrypted", "title": "Website", "url": "https://example.com"},
            "https://example.com", "Website",
        ),
        (
            {"type": "search_result_location", "cited_text": "Source",
             "start_block_index": 0, "end_block_index": 1, "search_result_index": 0,
             "source": "search://result", "title": "Search result"},
            "search://result", "Search result",
        ),
    ],
)
def test_anthropic_citations_become_normalized_text(
    anthropic_adapter, native_citation, source, title
):
    from anthropic.types import TextBlock

    block = TextBlock.model_validate({
        "type": "text", "text": "Source", "citations": [native_citation],
    })

    result = anthropic_adapter._normalize_output_item(block)

    assert isinstance(result, AssistantMessage)
    [text] = result.content
    [annotation] = text.annotations
    assert text.text == "Source"
    assert annotation.type == "citation"
    assert annotation.provider == "anthropic"
    assert annotation.provider_type == native_citation["type"]
    assert annotation.source == source
    assert annotation.title == title
    assert annotation.cited_text == "Source"
    assert annotation.provider_metadata == native_citation


def test_anthropic_text_without_citations_has_empty_annotations(anthropic_adapter):
    from anthropic.types import TextBlock

    block = TextBlock.model_validate({"type": "text", "text": "Hello"})

    result = anthropic_adapter._normalize_output_item(block)

    assert isinstance(result, AssistantMessage)
    assert result.content[0].annotations == []

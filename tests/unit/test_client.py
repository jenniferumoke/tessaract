import pytest

from tessaract import FunctionToolResult, SystemPrompt, UserMessage


def test_model_name_is_split_into_prefix_and_name(client):
    assert client._normalize_model_name("oai/gpt-test") == ("oai", "gpt-test")

def test_model_name_keeps_slashes_after_the_first(client):
    assert client._normalize_model_name("oai/org/gpt-test") == ("oai", "org/gpt-test")

@pytest.mark.parametrize("bad", ["gpt-test", "/gpt-test", "oai/", ""])
def test_model_name_without_prefix_is_rejected(client, bad):
    with pytest.raises(ValueError, match="provider prefix"):
        client._normalize_model_name(bad)

def test_unknown_prefix_is_rejected(client):
    with pytest.raises(ValueError, match="registered provider"):
        client.send(model="someprovider/model", input="hi")

def test_plain_input_string_becomes_one_user_message(client):
    request = client._build_request_model(
        model="gpt-test", provider="oai", input="hello",
        reasoning=None, tools=[], request_options={}, stream=False
    )

    assert request.input == [{"role": "user", "content": "hello"}]

def test_mixed_input_list_is_converted_in_order(client):
    request = client._build_request_model(
        model="gpt-test", provider="oai",
        input=[
            SystemPrompt(content="be brief"),
            "hi",
            FunctionToolResult(call_id="1", result="ok")
        ],
        reasoning=None, tools=[], request_options={}, stream=False
    )

    assert request.input == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"type": "function_call_output", "call_id": "c1", "result": "ok"}
    ]
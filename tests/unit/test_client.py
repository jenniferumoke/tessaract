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
        reasoning=None, max_tokens=None, tools=[], request_options={}, stream=False
    )

    assert request.input == [{"role": "user", "content": "hello"}]

def test_mixed_input_list_is_converted_in_order(client):
    request = client._build_request_model(
        model="gpt-test", provider="oai",
        input=[
            SystemPrompt(content="be brief"),
            "hi",
            FunctionToolResult(call_id="c1", result="ok")
        ],
        reasoning=None, max_tokens=None, tools=[], request_options={}, stream=False
    )

    assert request.input == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    ]

def test_unsupported_input_type_raises_type_error(client):
    with pytest.raises(TypeError, match="Unsupported input type: int"):
        client.send(model="oai/gpt-latest", input=[42])

def test_send_with_stream_true_generates_stream(client, monkeypatch):
    adapter = client.adapters["oai"]
    monkeypatch.setattr(adapter, "generate_stream", lambda request: ["streamed", request.stream])
    assert client.send(model="oai/gpt-latest", input="hi", stream=True) == ["streamed", True]

def test_send_with_stream_false_generates_sync(client, monkeypatch):
    adapter = client.adapters["oai"]
    monkeypatch.setattr(adapter, "generate_sync", lambda request: ("sync", request.stream))
    assert client.send(model="oai/gpt-latest", input="hi") == ("sync", False)

def test_instructions_are_put_on_the_request(client):
    request = client._build_request_model(
        model="gpt-test", provider="oai", input="hi",
        reasoning=None, max_tokens=None, tools=[], request_options={}, stream=False,
        instructions="Be brief."
    )

    assert request.instructions == "Be brief."

def test_cache_options_are_put_on_the_request(client, monkeypatch):
    from tessaract import CacheOptions

    adapter = client.adapters["oai"]
    monkeypatch.setattr(adapter, "generate_sync", lambda request: request.cache)
    cache = CacheOptions(retention="long")

    assert client.send(model="oai/gpt-latest", input="hi", cache=cache) is cache

def test_output_items_are_replayed_through_the_adapter(client, monkeypatch):
    from tessaract.types.output_types import ReasoningOutputItem

    adapter = client.adapters["oai"]
    monkeypatch.setattr(adapter, "map_output_item", lambda item: ("mapped", item.raw))

    request = client._build_request_model(
        model="gpt-test", provider="oai", input=[ReasoningOutputItem(raw="native")],
        reasoning=None, max_tokens=None, tools=[], request_options={}, stream=False
    )

    assert request.input == [("mapped", "native")]

def test_openai_output_items_replay_their_raw_form(openai_adapter):
    from tessaract.types.output_types import FunctionCallOutputItem

    raw = {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
    assert openai_adapter.map_output_item(FunctionCallOutputItem(raw=raw, call_id="c1", name="f")) is raw

def test_anthropic_stream_true_streams(anthropic_client, monkeypatch):
    adapter = anthropic_client.adapters["ant"]
    monkeypatch.setattr(adapter, "generate_stream", lambda request: ["streamed", request.stream])
    assert anthropic_client.send(model="ant/claude-test", input="hi", stream=True) == ["streamed", True]

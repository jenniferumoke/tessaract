import pytest

pytest.importorskip("anthropic")

from tessaract import AnthropicProvider


def test_api_key_is_read_from_anthropic_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    provider = AnthropicProvider()

    assert provider.api_key == "ant-key"
    provider.client.close()

def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()

def test_typed_client_fields_are_passed_to_the_sdk():
    provider = AnthropicProvider(api_key="k", base_url="https://proxy.example.com", timeout=12.0, max_retries=5)

    assert str(provider.client.base_url).startswith("https://proxy.example.com")
    assert provider.client.timeout == 12.0
    assert provider.client.max_retries == 5
    provider.client.close()

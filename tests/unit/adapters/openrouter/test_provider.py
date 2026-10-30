import pytest

pytest.importorskip("openrouter")

import httpx

from tessaract import OpenRouterProvider, Tessaract
from tests.fixtures import openrouter as payloads


def _serve(requests: list):
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payloads.response([payloads.message("hi")]))
    return handler


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    assert OpenRouterProvider().api_key == "env-key"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()


def test_typed_fields_are_translated_to_sdk_arguments():
    provider = OpenRouterProvider(
        api_key="key", base_url="https://proxy.example/api/v1", timeout=2.5, max_retries=0,
        app_url="https://myapp.example", app_title="My App",
    )

    kwargs = provider._client_kwargs()

    assert kwargs == {
        "api_key": "key",
        "server_url": "https://proxy.example/api/v1",
        "http_referer": "https://myapp.example",
        "x_open_router_title": "My App",
        "timeout_ms": 2500,
        # None turns the SDK's retries off
        "retry_config": None,
    }


def test_unset_fields_are_left_to_the_sdk_defaults():
    assert OpenRouterProvider(api_key="key")._client_kwargs() == {"api_key": "key"}


def test_a_retry_count_is_rejected():
    with pytest.raises(ValueError, match="retry_config"):
        OpenRouterProvider(api_key="key", max_retries=3)


def test_provider_args_win_over_typed_fields():
    provider = OpenRouterProvider(api_key="key", timeout=1, provider_args={"timeout_ms": 9000})

    assert provider._client_kwargs()["timeout_ms"] == 9000


def test_attribution_and_default_headers_are_sent(monkeypatch):
    requests: list = []
    client_class = httpx.Client
    # the provider builds its own httpx client for default_headers; give it the mock transport
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(_serve(requests)), **kwargs)
    )
    provider = OpenRouterProvider(
        api_key="key", app_url="https://myapp.example", app_title="My App",
        default_headers={"X-Custom": "1"}, default_query={"trace": "on"},
    )

    Tessaract(providers={"or": provider}).send(model="or/openai/gpt-5-mini", input="hi")

    [request] = requests
    assert request.headers["authorization"] == "Bearer key"
    assert request.headers["http-referer"] == "https://myapp.example"
    assert request.headers["x-openrouter-title"] == "My App"
    assert request.headers["x-custom"] == "1"
    assert request.url.params["trace"] == "on"


def test_default_headers_cannot_be_combined_with_a_custom_client():
    with pytest.raises(ValueError, match="default_headers"):
        OpenRouterProvider(api_key="key", default_headers={"X": "1"}, provider_args={"client": httpx.Client()})

import json
from unittest.mock import MagicMock

import pytest

from tessaract import Tessaract, OpenAIProvider
from tessaract.adapters.openai.openai_adapter import OpenAIAdapter

@pytest.fixture
def fake_openai():
    return MagicMock(name="FakeOpenAI")

@pytest.fixture
def openai_provider(fake_openai):
    provider = OpenAIProvider(api_key="test-api-key")

    provider._client = fake_openai
    return provider

@pytest.fixture
def openai_adapter(openai_provider):
    return OpenAIAdapter(openai_provider)

@pytest.fixture
def client(openai_provider):
    return Tessaract(providers={"oai": openai_provider})


class FakeAnthropicAPI:
    '''Serves canned Anthropic API responses over an httpx MockTransport and records each request.'''

    def __init__(self):
        self.requests: list = []
        self.message: dict | None = None
        self.sse_events: list[dict] | None = None

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)

    def handler(self, request):
        import httpx2

        self.requests.append(request)

        if json.loads(request.content).get("stream"):
            body = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in self.sse_events or []
            )
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        return httpx2.Response(200, json=self.message)

@pytest.fixture
def anthropic_api():
    return FakeAnthropicAPI()

@pytest.fixture
def anthropic_provider(anthropic_api):
    pytest.importorskip("anthropic")
    import httpx2

    from tessaract import AnthropicProvider

    # the Anthropic SDK is built on httpx2, so the mock transport comes from it too
    http_client = httpx2.Client(transport=httpx2.MockTransport(anthropic_api.handler))
    provider = AnthropicProvider(api_key="test-api-key", provider_args={"http_client": http_client})
    yield provider
    provider.client.close()

@pytest.fixture
def anthropic_adapter(anthropic_provider):
    from tessaract.adapters.anthropic.anthropic_adapter import AnthropicAdapter

    return AnthropicAdapter(anthropic_provider)

@pytest.fixture
def anthropic_client(anthropic_provider):
    return Tessaract(providers={"ant": anthropic_provider})


class FakeOpenRouterAPI:
    '''Serves canned OpenRouter Responses API payloads over an httpx MockTransport and records each request.

        responses are served in order; the last one is repeated.
    '''

    def __init__(self):
        self.requests: list = []
        self.responses: list[dict] = []

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def handler(self, request):
        import httpx

        self.requests.append(request)
        payload = self.responses[min(len(self.requests), len(self.responses)) - 1]
        return httpx.Response(200, json=payload)

@pytest.fixture
def openrouter_api():
    return FakeOpenRouterAPI()

@pytest.fixture
def openrouter_provider(openrouter_api):
    pytest.importorskip("openrouter")
    import httpx

    from tessaract import OpenRouterProvider

    http_client = httpx.Client(transport=httpx.MockTransport(openrouter_api.handler))
    provider = OpenRouterProvider(api_key="test-api-key", provider_args={"client": http_client})
    yield provider
    http_client.close()

@pytest.fixture
def openrouter_adapter(openrouter_provider):
    from tessaract.adapters.openrouter.openrouter_adapter import OpenRouterAdapter

    return OpenRouterAdapter(openrouter_provider)

@pytest.fixture
def openrouter_client(openrouter_provider):
    return Tessaract(providers={"or": openrouter_provider})

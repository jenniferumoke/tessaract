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

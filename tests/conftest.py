from unittest.mock import MagicMock

import pytest

from tessaract import Tessaract, OpenAIProvider
from tessaract.adapters.openai.open_adapter import OpenAIAdapter

@pytest.fixture
def fake_openai():
    return MagicMock(name="FakeOpenAI")

def openai_provider(fake_openai):
    provider = OpenAIProvider(api_key="test-api-key")

    provider._client = fake_openai
    return provider

def openai_adapter(openai_provider):
    return OpenAIAdapter(openai_provider)

def client(openai_provider):
    return Tessaract(providers={"oai": openai_provider})
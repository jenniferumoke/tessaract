import pytest
from pydantic import ValidationError

from tessaract import ProviderTool, UserLocation, WebSearchTool


def test_web_search_tool_defaults_set_nothing():
    tool = WebSearchTool()

    assert tool.type == "web_search"
    assert (tool.max_uses, tool.allowed_domains, tool.blocked_domains, tool.user_location) == (None, None, None, None)
    assert tool.cache is False
    assert tool.provider_options == {}

def test_allowed_and_blocked_domains_cannot_be_combined():
    with pytest.raises(ValidationError, match="cannot be used together"):
        WebSearchTool(allowed_domains=["a.com"], blocked_domains=["b.com"])

def test_max_uses_must_be_positive():
    with pytest.raises(ValidationError):
        WebSearchTool(max_uses=0)

def test_user_location_fields_are_optional():
    assert UserLocation(country="NL").model_dump(exclude_none=True) == {"country": "NL"}

def test_provider_tool_requires_a_known_provider():
    with pytest.raises(ValidationError):
        ProviderTool(provider="someone", definition={"type": "x"})

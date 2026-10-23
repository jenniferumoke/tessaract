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

@pytest.mark.parametrize("adapter", ["openai_adapter", "anthropic_adapter"])
@pytest.mark.parametrize("tool_class", [WebSearchTool, ProviderTool])
def test_a_tool_class_instead_of_an_instance_says_how_to_fix_it(request, adapter, tool_class):
    name = tool_class.__name__

    with pytest.raises(TypeError, match=rf"Pass {name}\(\), an instance, not the class {name}\."):
        request.getfixturevalue(adapter).map_tools([tool_class])

def test_a_tool_class_is_caught_through_send(client):
    with pytest.raises(TypeError, match=r"Pass WebSearchTool\(\)"):
        client.send(model="oai/gpt-test", input="hi", tools=[WebSearchTool])

def test_an_unknown_tool_type_lists_the_accepted_types(openai_adapter):
    with pytest.raises(TypeError, match="Unsupported tool type: dict. Use FunctionTool, WebSearchTool"):
        openai_adapter.map_tools([{"type": "web_search"}])

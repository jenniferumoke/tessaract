from typing import Any, Literal

from pydantic import BaseModel


class ProviderTool(BaseModel):
    '''A native tool definition, sent unchanged.

        Use it for provider tools Tessaract has no canonical type for yet, e.g. Anthropic's web_fetch:
            ProviderTool(provider="anthropic", definition={"type": "web_fetch_20250910", "name": "web_fetch"})

        provider is the provider the definition was written for. Sending it to a different
        provider raises, since the native shape wouldn't be valid there.
    '''
    provider: Literal["anthropic", "openai", "openrouter"]
    definition: dict[str, Any]

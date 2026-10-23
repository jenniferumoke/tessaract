from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class UserLocation(BaseModel):
    '''The approximate location of the user, used to make search results more relevant.

        country: a two-letter ISO country code, e.g. "NL"
        timezone: an IANA timezone, e.g. "Europe/Amsterdam"
    '''
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class WebSearchTool(BaseModel):
    '''A web search tool the provider runs on its own servers. Results come back as WebSearchCallOutputItems.

        max_uses: the maximum number of searches in one request. Anthropic only.
        allowed_domains: only search these domains (subdomains included).
        blocked_domains: never return results from these domains. Anthropic only.
        user_location: the user's approximate location.
        cache: add a cache breakpoint to the tool definition. Anthropic only; OpenAI caches automatically.
        provider_options: native tool fields, e.g. {"search_context_size": "high"} for OpenAI.
            "type" pins the native tool version, e.g. {"type": "web_search_20260209"} for Anthropic.

        An option the target provider doesn't support raises UnsupportedToolOptionError.
    '''
    type: Literal["web_search"] = "web_search"
    max_uses: int | None = Field(default=None, ge=1)
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: UserLocation | None = None
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _model_validator(self):
        if self.allowed_domains is not None and self.blocked_domains is not None:
            raise ValueError("allowed_domains and blocked_domains cannot be used together")

        return self

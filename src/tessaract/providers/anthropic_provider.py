from typing import TYPE_CHECKING, cast
import os

from .provider import Provider

if TYPE_CHECKING:
    from anthropic import Anthropic


class AnthropicProvider(Provider):
    def __post_init__(self):
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as exc:
            if exc.name != "anthropic":
                raise
            raise ImportError(
                "Anthropic support requires the optional dependency"
                'Install it with: pip install "tessaract[anthropic]"'
            ) from exc
        
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if self.api_key is None:
            raise ValueError(
                "No Anthropic API key found. Pass api_key to AnthropicProvider "
                "or set the ANTHROPIC_API_KEY environment variable."
            )
        
        self._client = Anthropic(api_key=self.api_key, **self.provider_args)

    @property
    def client(self) -> "Anthropic":
        return cast("Anthropic", self._client)
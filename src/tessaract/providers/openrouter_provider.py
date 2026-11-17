import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .provider import Provider

if TYPE_CHECKING:
    from openrouter import OpenRouter


@dataclass
class OpenRouterProvider(Provider):
    '''Holds the OpenRouter SDK client.

        app_url / app_title: your app's URL and name, sent as OpenRouter's HTTP-Referer and
            X-OpenRouter-Title headers for app attribution.

        The OpenRouter SDK takes different arguments from the OpenAI and Anthropic SDKs, so the typed
        fields are translated: base_url is sent as server_url, and timeout (seconds) as timeout_ms.
        The SDK retries with a time-based backoff rather than a retry count, so max_retries only
        accepts 0, which turns retries off; pass a RetryConfig as provider_args={"retry_config": ...}
        for anything else. default_headers and default_query are set on the HTTP client.
    '''
    app_url: str | None = None
    app_title: str | None = None

    def __post_init__(self) -> None:
        try:
            from openrouter import OpenRouter
        except ModuleNotFoundError as exc:
            if exc.name != "openrouter":
                raise
            raise ImportError(
                "OpenRouter support requires the optional dependency. "
                'Install it with: pip install "tessaract[openrouter]"'
            ) from exc

        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

        if self.api_key is None:
            raise ValueError(
                "No OpenRouter API key found. Pass api_key to OpenRouterProvider "
                "or set the OPENROUTER_API_KEY environment variable."
            )

        self._client = OpenRouter(**self._client_kwargs())

    def _client_kwargs(self) -> dict:
        '''Keyword arguments for the OpenRouter SDK client: the typed fields that are set, overridden by provider_args.'''
        typed_args: dict[str, object] = {
            "api_key": self.api_key,
            "server_url": self.base_url,
            "http_referer": self.app_url,
            "x_open_router_title": self.app_title,
        }

        if self.timeout is not None:
            typed_args["timeout_ms"] = int(self.timeout * 1000)

        if self.max_retries is not None:
            if self.max_retries != 0:
                raise ValueError(
                    "The OpenRouter SDK retries with a time-based backoff, not a retry count, so max_retries "
                    "only accepts 0. Pass provider_args={'retry_config': RetryConfig(...)} instead."
                )
            # None turns the SDK's default retries off; leaving it out keeps them
            typed_args["retry_config"] = None

        if self.default_headers is not None or self.default_query is not None:
            if "client" in self.provider_args:
                raise ValueError(
                    "default_headers and default_query are set on the HTTP client. Set them on the "
                    "client you pass in provider_args instead."
                )

            import httpx

            typed_args["client"] = httpx.Client(
                follow_redirects=True,
                headers=self.default_headers,
                params=cast(dict, self.default_query),
            )

        return {
            **{key: value for key, value in typed_args.items() if value is not None or key == "retry_config"},
            **self.provider_args,
        }

    @property
    def client(self) -> "OpenRouter":
        return cast("OpenRouter", self._client)

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload

from .adapters import AnthropicAdapter, OpenAIAdapter
from .providers import AnthropicProvider, OpenAIProvider
from .tools import Tool
from .types.input_types import InputType, UserMessage
from .types.output_types import AssistantMessage, OutputItem
from .types.request import CacheOptions, ReasoningOptions, Request
from .types.response import Response
from .types.streaming.event_types import StreamEventUnion

if TYPE_CHECKING:
    from .adapters.anthropic.anthropic_adapter import AnthropicAdapter
    from .adapters.openai.openai_adapter import OpenAIAdapter

InputItem: TypeAlias = str | InputType | AssistantMessage | OutputItem
Input: TypeAlias = str | Sequence[InputItem]

class Tessaract:
    def __init__(self, providers: dict[str, OpenAIProvider | AnthropicProvider]):
        self.providers = providers
        self.adapters: dict[str, OpenAIAdapter | AnthropicAdapter] = {}
        self.register_adapter()

    def register_adapter(self):
        for prefix, provider in self.providers.items():
            if isinstance(provider, OpenAIProvider):
                adapter = OpenAIAdapter(provider)
            elif isinstance(provider, AnthropicProvider):
                adapter = AnthropicAdapter(provider)
            else:
                raise NotImplementedError("not yet implemented")
            self.adapters[prefix] = adapter

    def _normalize_model_name(self, model: str):
        model_prefix, separator, model_name = model.partition("/")

        if not separator or not model_prefix or not model_name:
            raise ValueError(
                f"model must be of the form '<provider prefix>/<model name>', got {model!r}"
            )

        return (model_prefix, model_name)

    def _build_request_model(
        self,
        model: str,
        provider: str,
        input: Input,
        reasoning: ReasoningOptions | None,
        max_tokens: int | None,
        tools: Sequence[Tool],
        request_options: dict[str, Any],
        stream: bool,
        instructions: str | None = None,
        cache: CacheOptions | None = None
    ) -> Request:

        all_items = []
        
        if isinstance(input, str):
            converted_item = UserMessage(content=input)
            all_items.append(converted_item.raw(self.adapters[provider]))

        else:
            for message in input:
                if isinstance(message, str):
                    converted_item = UserMessage(content=message)
                    all_items.append(converted_item.raw(self.adapters[provider]))
                    
                elif isinstance(message, InputType):
                    all_items.append(message.raw(self.adapters[provider]))

                elif isinstance(message, AssistantMessage):  # noqa: SIM114
                    all_items.append(self.adapters[provider].map_assistant_message(message))

                elif isinstance(message, OutputItem):
                    all_items.append(self.adapters[provider].map_output_item(message))

                else:
                    raise TypeError(f"Unsupported input type: {type(message).__name__}")

        return Request(
            model=model,
            input=all_items,
            reasoning=reasoning,
            tools=list(tools),
            provider_options=request_options,
            stream=stream,
            max_tokens=max_tokens,
            instructions=instructions,
            cache=cache
        )

    @overload
    def send(
            self, model: str, 
            input: Input,
            stream: Literal[False] = False,
            reasoning: ReasoningOptions | None = None,
            tools: Sequence[Tool] | None = None,
            request_options: dict[str, Any] | None = None,
            max_tokens: int | None = None,
            instructions: str | None = None,
            cache: CacheOptions | None = None
            ) -> Response: ...


    @overload
    def send(
            self, model: str, 
            input: Input,
            stream: Literal[True],
            reasoning: ReasoningOptions | None = None,
            tools: Sequence[Tool] | None = None,
            request_options: dict[str, Any] | None = None,
            max_tokens: int | None = None,
            instructions: str | None = None,
            cache: CacheOptions | None = None
            ) -> Iterator[StreamEventUnion]: ...

    @overload
    def send(
        self,
        model: str,
        input: Input,
        stream: bool,
        reasoning: ReasoningOptions | None = None,
        tools: Sequence[Tool] | None = None,
        request_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        instructions: str | None = None,
        cache: CacheOptions | None = None
    ) -> Response | Iterator[StreamEventUnion]: ...


    def send(
            self, model: str, 
            input: Input,
            stream: bool = False,
            reasoning: ReasoningOptions | None = None,
            tools: Sequence[Tool] | None = None,
            request_options: dict[str, Any] | None = None,
            max_tokens: int | None = None,
            instructions: str | None = None,
            cache: CacheOptions | None = None
        ) -> Response | Iterator[StreamEventUnion]:
        '''Send a request to the provider registered under the model's prefix.

            Args:
                model: "<provider prefix>/<model name>", e.g. "oai/gpt-5.6-luna" or "ant/claude-opus-5-5".
                input: a string, or a list of strings, input items and output items from earlier responses.
                stream: return an iterator of stream events instead of a Response.
                reasoning: canonical reasoning options, mapped to each provider's native parameters.
                tools: tools the model may use: FunctionTools you run, server tools the provider runs
                    (WebSearchTool), and native definitions (ProviderTool).
                request_options: native request parameters, merged in. Canonical parameters win.
                max_tokens: the maximum number of output tokens. Sent to OpenAI as max_output_tokens
                    (omitted when None). Anthropic requires it: when None, the Anthropic adapter sends
                    DEFAULT_ANTHROPIC_MAX_TOKENS (16,000). Setting it explicitly is recommended for
                    Anthropic models, since output limits differ between models.
                instructions: system instructions for this request. Sent to OpenAI as the native
                    instructions parameter, and to Anthropic as the first block of system.
                cache: canonical prompt caching options. Sent to Anthropic as top-level cache_control
                    (automatic caching), and to OpenAI as prompt_cache_key.
        '''

        provider, model = self._normalize_model_name(model=model)

        if provider not in self.providers:
            raise ValueError("provider prefix must match registered provider in tessaract object")

        _tools = tools if tools is not None else []

        _request_options = request_options if request_options is not None else {}

        _tessaract_request = self._build_request_model(
            model=model,
            input=input,
            provider=provider,
            reasoning=reasoning,
            tools=_tools,
            request_options=_request_options,
            stream=stream,
            max_tokens=max_tokens,
            instructions=instructions,
            cache=cache
        )

        adapter = self.adapters[provider]

        if stream:
            return adapter.generate_stream(request=_tessaract_request)

        return adapter.generate_sync(request=_tessaract_request)

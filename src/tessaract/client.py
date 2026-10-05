from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, overload

from .adapters import AnthropicAdapter, OpenAIAdapter
from .providers import AnthropicProvider, OpenAIProvider
from .tools.function import FunctionTool
from .types.input_types import InputType, UserMessage
from .types.output_types import AssistantMessage, OutputItem
from .types.request import ReasoningOptions, Request
from .types.response import Response
from .types.streaming.event_types import StreamEventUnion

if TYPE_CHECKING:
    from .adapters.anthropic.anthropic_adapter import AnthropicAdapter
    from .adapters.openai.openai_adapter import OpenAIAdapter

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
        input: str | list[str | InputType],
        reasoning: ReasoningOptions | None,
        max_tokens: int | None,
        tools: list[FunctionTool],
        request_options: dict[str, Any],
        stream: bool
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
                    all_items.append(message.raw)

                elif isinstance(message, OutputItem):
                    all_items.append(message.raw)

                else:
                    raise TypeError(f"Unsupported input type: {type(message).__name__}")

        return Request(
            model=model,
            input=all_items,
            reasoning=reasoning,
            tools=tools,
            provider_options=request_options,
            stream=stream,
            max_tokens=max_tokens
        )

    @overload
    def send(
            self, model: str, 
            input: str | list[str | InputType],
            stream: Literal[False],
            reasoning: ReasoningOptions | None = None,
            tools: list[FunctionTool] | None = None,
            request_options: dict[str, Any] | None = None,
            max_tokens: int | None = None
            ) -> Response: ...

    @overload
    def send(
            self, model: str, 
            input: str | list[str | InputType],
            stream: Literal[False],
            max_tokens: int,
            reasoning: ReasoningOptions | None = None,
            tools: list[FunctionTool] | None = None,
            request_options: dict[str, Any] | None = None,

            ) -> Response: ...

    @overload
    def send(
            self, model: str, 
            input: str | list[str | InputType],
            stream: Literal[True],
            reasoning: ReasoningOptions | None = None,
            tools: list[FunctionTool] | None = None,
            request_options: dict[str, Any] | None = None,
            max_tokens: int | None = None
            ) -> Iterator[StreamEventUnion]: ...

    @overload
    def send(
        self,
        model: str,
        input: str | list[str | InputType],
        stream: bool,
        reasoning: ReasoningOptions | None = None,
        tools: list[FunctionTool] | None = None,
        request_options: dict[str, Any] | None = None,
        max_tokens: int | None = None
    ) -> Response | Iterator[StreamEventUnion]: ...


    def send(
            self, model: str, 
            input: str | list[str | InputType],
            stream: bool = False,
            reasoning: ReasoningOptions | None = None,
            tools: list[FunctionTool] | None = None,
            request_options: dict[str, Any] | None = None,
            max_tokens: int | None = None
        ) -> Response | Iterator[StreamEventUnion]:

        provider, model = self._normalize_model_name(model=model)

        if provider not in self.providers:
            raise ValueError("provider prefix must match registered provider in tessaract object")

        _request_provider = self.providers[provider]

        _tools = tools if tools is not None else []

        # if reasoning is not None:
        #     _reasoning = reasoning

        _request_options = request_options if request_options is not None else {}

        _tessaract_request = self._build_request_model(model=model, input=input, provider=provider, reasoning=reasoning, tools=_tools, request_options=_request_options, stream=stream, max_tokens=max_tokens)

        adapter = self.adapters[provider]

        if isinstance(_request_provider, OpenAIProvider):
            if stream == True:
                return adapter.generate_stream(request=_tessaract_request) 

            return adapter.generate_sync(request=_tessaract_request)

        elif isinstance(_request_provider, AnthropicProvider):
            return adapter.generate_sync(request=_tessaract_request)

        raise NotImplementedError("Unsupported provider")
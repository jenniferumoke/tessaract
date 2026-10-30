import json
from collections.abc import Iterator, Sequence
from typing import Any

from pydantic import BaseModel

from ...providers.openrouter_provider import OpenRouterProvider
from ...tools.computer import ComputerUseTool
from ...tools.function import InputSchema, Property
from ...tools.shell import ShellTool
from ...tools.web_search import WebSearchTool
from ...types.input_types import ComputerToolResult, ShellToolResult
from ...types.output_types import (
    Annotation,
    AssistantMessage,
    FunctionCallOutputItem,
    OutputItem,
    OutputType,
    ProviderOutputItem,
    ReasoningOutputItem,
    TextOutputItem,
)
from ...types.request import Request
from ...types.response import (
    IncompleteReason,
    OpenRouterResponse,
    ResponseError,
    ResponseStatus,
    Usage,
)
from ...types.streaming.event_types import StreamEventUnion
from ..adapter import (
    Adapter,
    AssistantMessageProtocol,
    CacheParamsProtocol,
    FunctionToolResultProtocol,
    FunctionToolSchemaProtocol,
    ReasoningParamsProtocol,
    SystemPromptProtocol,
    UserMessageProtocol,
)

# OpenRouter stores nothing between requests, so reasoning is only replayed from the history when
# its encrypted content (or Anthropic's signature) comes back with the response
ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"


def _value(value: Any) -> Any:
    '''The SDK marks fields the response left out with an Unset object; they read as None here.'''
    return None if type(value).__name__ == "Unset" else value


class OpenRouterAdapter(Adapter):
    '''Translates canonical types to and from OpenRouter's Responses API, through the OpenRouter SDK.

        Not supported yet: streaming, prompt caching, and the canonical server and client tools
        (WebSearchTool, ComputerUseTool, ShellTool). ProviderTool definitions are sent unchanged.

        The SDK validates every request against its own models: native fields it doesn't model are
        dropped, and an output item type it doesn't know comes back as a ProviderOutputItem but can't
        be replayed until the SDK adds it.
    '''
    provider_name = "openrouter"

    def __init__(self, provider: OpenRouterProvider):
        super().__init__(provider)
        self._client = provider.client

    def _not_supported_yet(self, feature: str) -> NotImplementedError:
        return NotImplementedError(f"{feature} isn't supported by the OpenRouter adapter yet.")

    def map_input_message(self, item: UserMessageProtocol | SystemPromptProtocol):
        if item.cache:
            raise self._not_supported_yet("Prompt caching (cache=True)")

        return {"role": item.role, "content": item.content}

    def _native_item(self, raw: Any) -> Any:
        '''The SDK only accepts plain dicts as input, so output items are dumped before they are replayed.'''
        # an output item the SDK doesn't know keeps its payload in .raw
        if getattr(raw, "is_unknown", False):
            return raw.raw

        if isinstance(raw, BaseModel):
            return raw.model_dump(mode="json", by_alias=True)

        return raw

    def map_assistant_message(self, item: AssistantMessageProtocol):
        return self._native_item(item.raw)

    def map_output_item(self, item: OutputItem) -> Any:
        return self._native_item(item.raw)

    def map_reasoning_params(self, reasoning: ReasoningParamsProtocol | None) -> dict[str, Any] | None:
        if reasoning is None:
            return None

        native_reasoning: dict[str, Any] = {}

        mode = reasoning.mode
        match mode:
            case None:
                pass
            case "standard" | "pro":
                native_reasoning["mode"] = mode
            case "enabled" | "disabled":
                native_reasoning["enabled"] = mode == "enabled"
            case _:
                raise ValueError(
                    f"OpenRouter does not support mode: {mode!r}. "
                    'Use "standard", "pro", "enabled" or "disabled".'
                )

        summary = reasoning.summary
        match summary:
            case None:
                pass
            case "auto" | "concise" | "detailed":
                native_reasoning["summary"] = summary
            case _:
                raise ValueError(
                    f"OpenRouter does not accept reasoning summary {summary!r}. "
                    'Use "auto", "concise" or "detailed".'
                )

        if reasoning.effort is not None:
            native_reasoning["effort"] = (
                "xhigh" if reasoning.effort == "extra_high" else reasoning.effort
            )

        # OpenRouter sends the budget to models that take one, such as Anthropic and Gemini models
        if reasoning.budget is not None:
            native_reasoning["max_tokens"] = reasoning.budget

        # native_options add native reasoning fields; the typed fields win
        return {**(reasoning.native_options or {}), **native_reasoning}

    def map_cache_params(self, cache: CacheParamsProtocol | None) -> dict[str, Any]:
        if cache is None:
            return {}

        raise self._not_supported_yet("Prompt caching (CacheOptions)")

    def _native_property(self, prop: Property) -> dict:
        native: dict[str, object] = {"type": prop.type}

        if prop.description is not None:
            native["description"] = prop.description

        if prop.enum is not None:
            native["enum"] = prop.enum

        if prop.items is not None:
            native["items"] = self._native_property(prop.items)

        if prop.properties is not None:
            native["properties"] = {
                name: self._native_property(nested)
                for name, nested in prop.properties.items()
            }

        if prop.required is not None:
            native["required"] = prop.required

        # Strict mode requires additionalProperties: false on every object
        if "object" in prop._types():
            native["additionalProperties"] = prop.additionalProperties if prop.additionalProperties is not None else False

        return native

    def _native_tool_parameters(self, input_schema: InputSchema):
        return {
            "type": input_schema.type,
            "properties": {
                prop_name: self._native_property(prop_schema)
                for prop_name, prop_schema in input_schema.properties.items()
            },
            "required": input_schema.required,
            "additionalProperties": input_schema.additionalProperties if input_schema.additionalProperties is not None else False
        }

    def map_function_schema(self, tools: Sequence[FunctionToolSchemaProtocol]) -> list:
        native_tools = []

        for tool in tools:
            if tool.cache:
                raise self._not_supported_yet("Prompt caching (cache=True)")

            native_tool: dict[str, object] = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "strict": tool.strict if tool.strict is not None else True,
            }

            if tool.input_schema is not None:
                native_tool["parameters"] = self._native_tool_parameters(tool.input_schema)

            # native fields the SDK doesn't model are dropped by its request validation
            extra_fields = self._extra_fields(tool.provider_options, set(native_tool) | {"parameters"})
            native_tools.append({**extra_fields, **native_tool})

        return native_tools

    def map_web_search_tool(self, tool: WebSearchTool) -> dict[str, Any]:
        raise self._not_supported_yet("WebSearchTool")

    def map_computer_use_tool(self, tool: ComputerUseTool) -> dict[str, Any]:
        raise self._not_supported_yet("ComputerUseTool")

    def map_shell_tool(self, tool: ShellTool) -> dict[str, Any]:
        raise self._not_supported_yet("ShellTool")

    def map_computer_tool_result(self, item: ComputerToolResult) -> Any:
        raise self._not_supported_yet("ComputerToolResult")

    def map_shell_tool_result(self, item: ShellToolResult) -> Any:
        raise self._not_supported_yet("ShellToolResult")

    def _extra_fields(self, provider_options: dict[str, Any], canonical_keys: set[str]) -> dict[str, Any]:
        return {key: value for key, value in provider_options.items() if key not in canonical_keys}

    def _is_content_parts(self, result) -> bool:
        return isinstance(result, list) and all(
            isinstance(part, dict) and "type" in part for part in result
        )

    def map_tool_result(self, item: FunctionToolResultProtocol):
        if item.cache:
            raise self._not_supported_yet("Prompt caching (cache=True)")

        # function_call_output has no error flag, so errors are marked in the output itself
        if item.is_error:
            output = json.dumps({"error": item.result}, default=str)

        elif isinstance(item.result, str) or self._is_content_parts(item.result):
            output = item.result

        else:
            output = json.dumps(item.result, default=str)

        return {
            **self._extra_fields(item.provider_options, {"type", "call_id", "output"}),
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": output
        }

    def _normalize_annotation(self, annotation) -> Annotation:
        native = annotation.raw if getattr(annotation, "is_unknown", False) else annotation.model_dump(mode="json", by_alias=True)
        provider_type = native.get("type")

        return Annotation(
            type="file_path" if provider_type == "file_path" else "citation",
            provider="openrouter",
            provider_type=provider_type,
            source=native.get("url") or native.get("file_id"),
            title=native.get("title") or native.get("filename"),
            provider_metadata=native,
        )

    def _provider_item(self, output_item) -> ProviderOutputItem:
        provider_type = output_item.raw.get("type", "unknown") if getattr(output_item, "is_unknown", False) else output_item.type
        return ProviderOutputItem(raw=output_item, provider_type=provider_type)

    def _normalize_output_item(self, output_item) -> OutputType:
        if getattr(output_item, "is_unknown", False):
            return self._provider_item(output_item)

        match output_item.type:
            # a message with a refusal or a part the SDK doesn't know stays native
            case "message" if all(part.type == "output_text" for part in output_item.content):
                return AssistantMessage(
                    raw=output_item,
                    content=[
                        TextOutputItem(
                            raw=part,
                            text=part.text,
                            annotations=[
                                self._normalize_annotation(annotation)
                                for annotation in part.annotations or []
                            ]
                        )
                        for part in output_item.content
                    ]
                )

            case "reasoning":
                # encrypted_content (OpenAI and others) or signature (Anthropic) stays in raw, and is replayed
                return ReasoningOutputItem(
                    raw=output_item,
                    id=output_item.id,
                    content="".join(part.text for part in _value(output_item.content) or []),
                    text="".join(part.text for part in output_item.summary)
                )

            case "function_call":
                return FunctionCallOutputItem(
                    raw=output_item,
                    call_id=output_item.call_id,
                    name=output_item.name,
                    arguments=json.loads(output_item.arguments) if output_item.arguments else {}
                )

            case _:
                return self._provider_item(output_item)

    def _normalize_status(self, status) -> ResponseStatus | None:
        try:
            return ResponseStatus(status) if status is not None else None
        except ValueError:
            return None

    def _normalize_incomplete_reason(self, raw_response) -> IncompleteReason | None:
        details = _value(raw_response.incomplete_details)
        match getattr(details, "reason", None):
            case "max_output_tokens":
                return IncompleteReason.MAX_TOKENS
            case "content_filter":
                return IncompleteReason.CONTENT_FILTER
            case _:
                return None

    def _normalize_usage(self, provider_usage) -> Usage | None:
        if provider_usage is None:
            return None

        return Usage(
            input_tokens=provider_usage.input_tokens,
            output_tokens=provider_usage.output_tokens,
            total_tokens=provider_usage.total_tokens,
            cached_input_tokens=provider_usage.input_tokens_details.cached_tokens,
            cache_write_input_tokens=_value(provider_usage.input_tokens_details.cache_write_tokens),
            reasoning_tokens=provider_usage.output_tokens_details.reasoning_tokens,
        )

    def _normalize_response(self, raw_response) -> OpenRouterResponse:
        provider_usage = _value(raw_response.usage)
        error = _value(raw_response.error)

        return OpenRouterResponse(
            id=raw_response.id,
            model=raw_response.model,
            status=self._normalize_status(raw_response.status),
            incomplete_reason=self._normalize_incomplete_reason(raw_response),
            output=[self._normalize_output_item(item) for item in raw_response.output],
            error=ResponseError(message=error.message, code=error.code) if error is not None else None,
            usage=self._normalize_usage(provider_usage),
            provider_usage=provider_usage,
            raw_response=raw_response,
        )

    def _build_request_kwargs(self, request: Request) -> dict[str, Any]:
        if request.cache is not None:
            self.map_cache_params(request.cache)

        provider_options = dict(request.provider_options or {})

        # replayed reasoning needs its encrypted content, so it is always asked for
        include = list(provider_options.get("include") or [])
        if ENCRYPTED_REASONING_INCLUDE not in include:
            include.append(ENCRYPTED_REASONING_INCLUDE)

        canonical_params: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
            "include": include,
        }

        reasoning = self.map_reasoning_params(request.reasoning)
        if reasoning is not None:
            canonical_params["reasoning"] = reasoning

        # only sent when set, so native tools passed in request_options are kept
        if request.tools:
            canonical_params["tools"] = self.map_tools(request.tools)

        if request.max_tokens is not None:
            canonical_params["max_output_tokens"] = request.max_tokens

        if request.instructions is not None:
            canonical_params["instructions"] = request.instructions

        # request_options are keyword arguments of responses.send(), e.g. provider, models or plugins
        return {**provider_options, **canonical_params}

    def generate_sync(self, request: Request) -> OpenRouterResponse:
        kwargs = self._build_request_kwargs(request)

        raw_response = self._client.responses.send(**kwargs, stream=False)

        return self._normalize_response(raw_response)

    def generate_stream(self, request: Request) -> Iterator[StreamEventUnion]:
        raise self._not_supported_yet("Streaming (stream=True)")

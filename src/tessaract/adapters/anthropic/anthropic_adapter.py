import json
from collections.abc import Iterator, Sequence
from typing import cast

from ..adapter import (
    Adapter,
    FunctionToolResultProtocol,
    FunctionToolSchemaProtocol,
    ReasoningParamsProtocol,
    UserMessageProtocol,
    SystemPromptProtocol
)

from ...providers.anthropic_provider import AnthropicProvider

from ...tools.function import InputSchema, Property
from ...types.output_types import (
    Annotation,
    AssistantMessage,
    FunctionCallOutputItem,
    OutputType,
    ProviderOutputItem,
    ReasoningOutputItem,
    TextOutputItem,
)
from ...types.request import Request
from ...types.response import AnthropicResponse, StopReason, ResponseError, Usage
from ...types.streaming.event_types import (
    CustomProviderEvent,
    FunctionCallArgumentDeltaEvent,
    OutputItemCompletedEvent,
    ReasoningStartedEvent,
    ReasoningSummaryDeltaEvent,
    ReasoningTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseStartedEvent,
    StreamEventUnion,
    TextDeltaEvent,
)


class AnthropicAdapter(Adapter):
    def __init__(self, provider: AnthropicProvider):
        super().__init__(provider)
        self._client = provider.client

    
    def map_input_message(self, item: UserMessageProtocol | SystemPromptProtocol):
        return {
            "role": item.role,
            "content": item.content
        }

    def map_reasoning_params(self, reasoning: ReasoningParamsProtocol | None) -> dict | None:
        pass

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

        if "object" in prop._types():
            native["additionalProperties"] = prop.additionalProperties if prop.additionalProperties is not None else False

        return native

    def _native_tool_parameters(self, input_schema: InputSchema):
        _properties = {
            prop_name: self._native_property(prop_schema)
            for prop_name, prop_schema in input_schema.properties.items()
        }

        return {
            "type": input_schema.type,
            "properties": _properties,
            "required": input_schema.required,
        }

    def  map_function_schema(self, tools: Sequence[FunctionToolSchemaProtocol]) -> list:

        _native_tools_list = []
        for tool in tools:
            _native_tool_schema: dict[str, object] = {}

            _native_tool_schema["type"] = "custom"
            _native_tool_schema["name"] = tool.name
            _native_tool_schema["description"] = tool.description
            _native_tool_schema["strict"] = tool.strict if tool.strict is not None else True

            if tool.input_schema is not None:
                _native_tool_schema["input_schema"] = self._native_tool_parameters(tool.input_schema) if tool.input_schema is not None else None

            canonical_keys = {
                "type",
                "name",
                "description",
                "strict",
                "parameters"
            }

            extra_fields = {
                key: value
                for key, value in tool.provider_options.items()
                if key not in canonical_keys
            }
            _native_tools_list.append({**extra_fields, **_native_tool_schema})
        return _native_tools_list

    def _normalize_annotation(self, citation) -> Annotation:
        return Annotation(
            provider="anthropic",
            provider_type=citation.type,
            source=(
                getattr(citation, "url", None)
                or getattr(citation, "source", None)
                or getattr(citation, "file_id", None)
            ),
            title=getattr(citation, "title", None) or getattr(citation, "document_title", None),
            cited_text=citation.cited_text,
            provider_metadata=citation.model_dump(mode="json"),
        )

    def _normalize_output_item(self, content_block) -> OutputType:
        match content_block.type:
            case "text":
                return AssistantMessage(
                        raw=content_block,
                        content=[
                            TextOutputItem(
                                raw=content_block,
                                text=content_block.text,
                                annotations=[
                                    self._normalize_annotation(citation)
                                    for citation in content_block.citations or []
                                ]
                            )
                        ]
                    )

            case "thinking":
                return ReasoningOutputItem(
                    raw=content_block,
                    id=content_block.signature,
                    text=content_block.thinking
                ) 

            case "redacted_thinking":
                return ProviderOutputItem(
                    provider_type="redacted_thinking",
                    raw=content_block
                )

            case "tool_use":
                return FunctionCallOutputItem(
                    raw=content_block,
                    call_id=content_block.id,
                    name=content_block.name,
                    arguments=json.loads(content_block.input)
                )

            case _:
                raise ValueError(f"Unsupported Anthropic output item type: {content_block.type!r}")

    def _normalize_output(self, output_items) -> list[OutputType]:
        return [self._normalize_output_item(item) for item in output_items]


    def _build_request_kwargs(self, request: Request):
        if request.max_tokens is None:
            raise ValueError("max_tokens required for Anthropic requests.")
        
        canonical_params = {
            "model": request.model,
            "messages": request.input,
            "tools": self.map_function_schema(request.tools or []),
            "max_tokens": request.max_tokens
        }

        provider_options = dict(**request.provider_options or {})

        extra_body = provider_options.get("extra_body")

        if extra_body is not None:
            provider_options["extra_body"] = {
                key: value
                for key, value in extra_body.items()
                if key not in canonical_params
            }


        return {**provider_options, **canonical_params}

    def generate_sync(self, request: Request) -> AnthropicResponse:

        kwargs = self._build_request_kwargs(request)

        _raw_response = self._client.messages.create(**kwargs, stream=False)
        reason = _raw_response.stop_reason
        if reason is None:
            raise ValueError("Anthropic response has no stop_reason")
        
        response = AnthropicResponse(
            id=_raw_response.id,
            model=request.model,
            stop_reason=StopReason(reason),
            output=self._normalize_output(_raw_response.content),
            raw_response=_raw_response
        )

        return cast(AnthropicResponse, response)

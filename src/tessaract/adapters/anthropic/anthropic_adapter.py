import json
from collections.abc import Iterator, Sequence
from typing import Any

from ..adapter import (
    Adapter,
    AssistantMessageProtocol,
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
    OutputItem,
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
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseStartedEvent,
    StreamEventUnion,
    TextDeltaEvent,
    ToolCallStartedEvent,
)

# Anthropic requires max_tokens on every request. This is sent when the caller doesn't set one.
# It stays below the SDK's limit for non-streaming requests (about 21k tokens without a custom timeout).
DEFAULT_ANTHROPIC_MAX_TOKENS = 16_000

# Thinking features only available on the beta Messages API, with the header each one needs
THINKING_DISPLAY_UPDATES_BETA = "thinking-display-updates-2026-08-18"
THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"

MIN_THINKING_BUDGET = 1024

# Convenience events the SDK's MessageStream adds on top of the raw events; they duplicate the deltas
_STREAM_HELPER_EVENTS = {"text", "thinking", "input_json", "signature", "citation"}


class AnthropicAdapter(Adapter):
    def __init__(self, provider: AnthropicProvider):
        super().__init__(provider)
        self._client = provider.client

    def map_input_message(self, item: UserMessageProtocol | SystemPromptProtocol):
        # system messages are moved to the top-level system parameter in _build_request_kwargs
        return {
            "role": item.role,
            "content": item.content
        }

    def map_assistant_message(self, item: AssistantMessageProtocol):
        parts = item.content if isinstance(item.content, list) else [item.content]
        return {
            "role": item.role,
            "content": [{"type": "text", "text": part.text} for part in parts]
        }

    def _block_param(self, block) -> dict:
        if isinstance(block, dict):
            return block
        return block.model_dump(mode="json", exclude_none=True)

    def map_output_item(self, item: OutputItem):
        # thinking, redacted_thinking, tool_use and server tool blocks are replayed unchanged,
        # thinking signatures included, and merged into one assistant turn in _build_request_kwargs
        return {
            "role": "assistant",
            "content": [self._block_param(item.raw)]
        }

    def _is_content_parts(self, result) -> bool:
        return isinstance(result, list) and all(
            isinstance(part, dict) and "type" in part for part in result
        )

    def map_tool_result(self, item: FunctionToolResultProtocol):
        if isinstance(item.result, str) or self._is_content_parts(item.result):
            content = item.result
        else:
            content = json.dumps(item.result, default=str)

        canonical_keys = {"type", "tool_use_id", "content", "is_error"}
        extra_fields = {
            key: value
            for key, value in item.provider_options.items()
            if key not in canonical_keys
        }

        block = {
            **extra_fields,
            "type": "tool_result",
            "tool_use_id": item.call_id,
            "content": content,
        }

        if item.is_error:
            block["is_error"] = True

        return {
            "role": "user",
            "content": [block]
        }

    def _map_effort(self, effort) -> str | None:
        match effort:
            case None | "none":
                return None
            case "low" | "medium" | "high" | "max":
                return effort
            case "extra_high":
                return "xhigh"
            case _:
                raise ValueError(
                    f"Anthropic does not support effort: {effort!r}. "
                    'Use "none", "low", "medium", "high", "extra_high" or "max".'
                )

    def _map_display(self, summary) -> str | None:
        match summary:
            case None:
                return None
            # Anthropic has a single level of summarized thinking
            case "auto" | "concise" | "detailed":
                return "summarized"
            case "omitted" | "updates":
                return summary
            case _:
                raise ValueError(
                    f"Anthropic does not accept reasoning summary {summary!r}. "
                    'Use "auto", "concise", "detailed", "omitted" or "updates".'
                )

    def map_reasoning_params(
        self,
        reasoning: ReasoningParamsProtocol | None,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
    ) -> dict[str, Any]:
        '''Maps ReasoningOptions to Anthropic's thinking and output_config.effort parameters.

            Returns the native request parameters to send, which is empty when reasoning is None.
        '''
        if reasoning is None:
            return {}

        effort = self._map_effort(reasoning.effort)
        display = self._map_display(reasoning.summary)
        budget = reasoning.budget

        mode = reasoning.mode
        if mode is None:
            if reasoning.effort == "none":
                mode = "disabled"
            else:
                mode = "enabled" if budget is not None else "adaptive"

        if reasoning.effort == "none" and mode != "disabled":
            raise ValueError(f'effort="none" turns thinking off and cannot be used with mode {mode!r}.')

        thinking: dict[str, Any] = {"type": mode}

        match mode:
            case "enabled":
                if budget is None:
                    raise ValueError('mode="enabled" requires a thinking budget. Set budget, e.g. budget=4096.')
                if budget < MIN_THINKING_BUDGET:
                    raise ValueError(f"Anthropic thinking budget must be at least {MIN_THINKING_BUDGET}, got {budget}.")
                if budget >= max_tokens:
                    raise ValueError(
                        f"Anthropic thinking budget ({budget}) must be less than max_tokens ({max_tokens}). "
                        f"When max_tokens is not set, it defaults to {DEFAULT_ANTHROPIC_MAX_TOKENS}."
                    )
                thinking["budget_tokens"] = budget

            case "adaptive":
                if budget is not None:
                    raise ValueError('budget is only used with mode="enabled". Adaptive thinking picks its own budget; use effort instead.')

            case "between_tools" | "disabled":
                if budget is not None or display is not None:
                    raise ValueError(f"budget and summary cannot be used with mode {mode!r}.")

            case _:
                raise ValueError(
                    f"Anthropic does not support mode: {mode!r}. "
                    'Use "enabled", "adaptive", "between_tools" or "disabled".'
                )

        if display is not None:
            thinking["display"] = display

        # native_options add native thinking fields such as block_binding; the typed fields win
        native: dict[str, Any] = {"thinking": {**(reasoning.native_options or {}), **thinking}}

        if effort is not None:
            native["output_config"] = {"effort": effort}

        return native

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

    def _native_tool_parameters(self, input_schema: InputSchema | None):
        # Anthropic requires an input_schema, even for tools that take no arguments
        if input_schema is None:
            return {"type": "object", "properties": {}}

        _properties = {
            prop_name: self._native_property(prop_schema)
            for prop_name, prop_schema in input_schema.properties.items()
        }

        return {
            "type": input_schema.type,
            "properties": _properties,
            "required": input_schema.required,
            "additionalProperties": input_schema.additionalProperties if input_schema.additionalProperties is not None else False
        }

    def map_function_schema(self, tools: Sequence[FunctionToolSchemaProtocol]) -> list:

        _native_tools_list = []
        for tool in tools:
            _native_tool_schema: dict[str, object] = {}

            _native_tool_schema["type"] = "custom"
            _native_tool_schema["name"] = tool.name
            _native_tool_schema["description"] = tool.description
            _native_tool_schema["strict"] = tool.strict if tool.strict is not None else True
            _native_tool_schema["input_schema"] = self._native_tool_parameters(tool.input_schema)

            canonical_keys = {
                "type",
                "name",
                "description",
                "strict",
                "input_schema"
            }

            extra_fields = {
                key: value
                for key, value in tool.provider_options.items()
                if key not in canonical_keys
            }
            _native_tools_list.append({**extra_fields, **_native_tool_schema})
        return _native_tools_list

    def _as_blocks(self, content) -> list:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return list(content)

    def _assemble_messages(self, items: list) -> tuple[list, list]:
        '''Splits the mapped input into system blocks and alternating user/assistant turns.

            Anthropic keeps system prompts out of messages, and expects one assistant turn per model
            response (thinking, text and tool_use blocks together) followed by one user turn holding
            every tool_result. Neighbouring items with the same role are merged into one turn.
        '''
        system: list = []
        messages: list = []

        for item in items:
            if not isinstance(item, dict) or "role" not in item:
                raise TypeError(f"Unsupported Anthropic input item: {item!r}")

            role = item["role"]
            blocks = self._as_blocks(item["content"])

            if role == "system":
                system.extend(blocks)
            elif messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({**item, "content": blocks})

        # tool_result blocks must come first in a user turn
        for message in messages:
            if message["role"] == "user":
                message["content"].sort(key=lambda block: not (isinstance(block, dict) and block.get("type") == "tool_result"))

        return system, messages

    def _build_request_kwargs(self, request: Request) -> dict[str, Any]:
        max_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_ANTHROPIC_MAX_TOKENS

        provider_options = dict(**request.provider_options or {})

        if "instructions" in provider_options:
            raise ValueError(
                "Anthropic has no instructions request option. "
                "Pass instructions=... to send(), or add a SystemPrompt to the input."
            )

        system, messages = self._assemble_messages(request.input)

        if request.instructions is not None:
            system = [{"type": "text", "text": request.instructions}, *system]

        canonical_params: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if system:
            canonical_params["system"] = system

        if request.tools:
            canonical_params["tools"] = self.map_function_schema(request.tools)

        reasoning_params = self.map_reasoning_params(request.reasoning, max_tokens=max_tokens)

        if "thinking" in reasoning_params:
            canonical_params["thinking"] = reasoning_params["thinking"]

        if "output_config" in reasoning_params:
            # keep other native output_config fields, such as format; the typed effort wins
            canonical_params["output_config"] = {
                **(provider_options.get("output_config") or {}),
                **reasoning_params["output_config"],
            }

        extra_body = provider_options.get("extra_body")

        if extra_body is not None:
            provider_options["extra_body"] = {
                key: value
                for key, value in extra_body.items()
                if key not in canonical_params
            }

        return {**provider_options, **canonical_params}

    def _required_betas(self, kwargs: dict[str, Any]) -> list[str]:
        thinking = kwargs.get("thinking") or {}
        betas = []

        if thinking.get("display") == "updates":
            betas.append(THINKING_DISPLAY_UPDATES_BETA)

        if thinking.get("block_binding") is not None:
            betas.append(THINKING_BINDING_CONTROLS_BETA)

        return betas

    def _messages_api(self, kwargs: dict[str, Any]):
        '''Returns the Messages API to call: the beta API when the request uses a beta-only feature.'''
        required = self._required_betas(kwargs)

        if not required:
            return self._client.messages, kwargs

        betas = list(kwargs.get("betas") or [])
        betas.extend(beta for beta in required if beta not in betas)

        return self._client.beta.messages, {**kwargs, "betas": betas}

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
                # raw is the list of native text blocks, since one message can span several blocks
                return AssistantMessage(
                        raw=[content_block],
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
                # Anthropic returns summarized thinking, so it goes in text like an OpenAI summary.
                # The signature has no id role; it stays in raw, which is replayed unchanged.
                return ReasoningOutputItem(
                    raw=content_block,
                    text=content_block.thinking
                )

            case "tool_use":
                return FunctionCallOutputItem(
                    raw=content_block,
                    call_id=content_block.id,
                    name=content_block.name,
                    arguments=dict(content_block.input)
                )

            case _:
                # redacted_thinking, server tool blocks and anything newer pass through unchanged
                return ProviderOutputItem(
                    provider_type=content_block.type,
                    raw=content_block
                )

    def _text_parts(self, message: AssistantMessage) -> list[TextOutputItem]:
        return message.content if isinstance(message.content, list) else [message.content]

    def _normalize_output(self, output_items) -> list[OutputType]:
        output: list[OutputType] = []

        for block in output_items:
            item = self._normalize_output_item(block)

            # Anthropic splits text at every citation; neighbouring text blocks are one message
            if isinstance(item, AssistantMessage) and output and isinstance(output[-1], AssistantMessage):
                previous = output[-1]
                output[-1] = AssistantMessage(
                    raw=[*previous.raw, *item.raw],
                    content=[*self._text_parts(previous), *self._text_parts(item)],
                )
            else:
                output.append(item)

        return output

    def _normalize_usage(self, provider_usage) -> Usage | None:
        if provider_usage is None:
            return None

        cache_read = provider_usage.cache_read_input_tokens
        cache_write = provider_usage.cache_creation_input_tokens

        # Anthropic's input_tokens leaves out cached tokens; OpenAI's includes them.
        # input_tokens is normalized to the total so it means the same on both providers.
        input_tokens = provider_usage.input_tokens + (cache_read or 0) + (cache_write or 0)

        output_details = getattr(provider_usage, "output_tokens_details", None)

        return Usage(
            input_tokens=input_tokens,
            output_tokens=provider_usage.output_tokens,
            total_tokens=input_tokens + provider_usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            reasoning_tokens=output_details.thinking_tokens if output_details is not None else None,
        )

    def _normalize_stop_reason(self, reason) -> StopReason | str | None:
        if reason is None:
            return None
        try:
            return StopReason(reason)
        except ValueError:
            return reason

    def _normalize_response(self, raw_response) -> AnthropicResponse:
        return AnthropicResponse(
            id=raw_response.id,
            model=raw_response.model,
            stop_reason=self._normalize_stop_reason(raw_response.stop_reason),
            output=self._normalize_output(raw_response.content),
            usage=self._normalize_usage(raw_response.usage),
            provider_usage=raw_response.usage,
            raw_response=raw_response
        )

    def _normalize_stream_event(self, event) -> Iterator[StreamEventUnion]:
        match event.type:
            case "message_start":
                yield ResponseStartedEvent(raw_event=event)

            case "content_block_start" if event.content_block.type == "tool_use":
                yield ToolCallStartedEvent(
                    call_id=event.content_block.id,
                    name=event.content_block.name,
                    output_index=event.index,
                    raw_event=event
                )

            case "content_block_start" if event.content_block.type == "thinking":
                yield ReasoningStartedEvent(
                    output_index=event.index,
                    raw_event=event
                )

            case "content_block_delta" if event.delta.type == "text_delta":
                yield TextDeltaEvent(
                    delta=event.delta.text,
                    provider="anthropic",
                    output_index=event.index,
                    raw_event=event
                )

            case "content_block_delta" if event.delta.type == "thinking_delta":
                yield ReasoningSummaryDeltaEvent(
                    delta=event.delta.thinking,
                    output_index=event.index,
                    raw_event=event
                )

            case "content_block_delta" if event.delta.type == "input_json_delta":
                yield FunctionCallArgumentDeltaEvent(
                    delta=event.delta.partial_json,
                    output_index=event.index,
                    raw_event=event
                )

            # MessageStream adds the finished block and message to these events
            case "content_block_stop" if getattr(event, "content_block", None) is not None:
                yield OutputItemCompletedEvent(
                    item=self._normalize_output_item(event.content_block),
                    output_index=event.index,
                    raw_event=event
                )

            case "message_stop" if getattr(event, "message", None) is not None:
                yield ResponseCompletedEvent(
                    response=self._normalize_response(event.message),
                    raw_event=event
                )

            case helper if helper in _STREAM_HELPER_EVENTS:
                return

            case _:
                yield CustomProviderEvent(
                    provider_type=event.type,
                    raw_event=event
                )

    def _response_error(self, exc) -> ResponseError:
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        code = error.get("type") if isinstance(error, dict) else None
        return ResponseError(message=getattr(exc, "message", None) or str(exc), code=code)

    def generate_sync(self, request: Request) -> AnthropicResponse:
        messages_api, kwargs = self._messages_api(self._build_request_kwargs(request))

        _raw_response = messages_api.create(**kwargs, stream=False)

        return self._normalize_response(_raw_response)

    def generate_stream(self, request: Request) -> Iterator[StreamEventUnion]:
        from anthropic import APIStatusError

        messages_api, kwargs = self._messages_api(self._build_request_kwargs(request))

        with messages_api.stream(**kwargs) as stream:
            try:
                for raw_event in stream:
                    yield from self._normalize_stream_event(raw_event)
            except APIStatusError as exc:
                # error events sent mid-stream; errors before the stream starts are raised
                error = self._response_error(exc)
                yield ResponseFailedEvent(
                    message=error.message,
                    error=error,
                    raw_event=exc
                )

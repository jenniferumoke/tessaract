import json
from collections.abc import Iterator, Sequence
from typing import Any

from pydantic import BaseModel

from ...errors import UnsupportedToolOptionError
from ...providers.openrouter_provider import OpenRouterProvider
from ...tools.computer import ComputerUseTool
from ...tools.function import InputSchema, Property
from ...tools.shell import ContainerEnvironment, ShellTool
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
    ShellCallOutputItem,
    ShellCommandOutput,
    TextOutputItem,
    WebSearchCallOutputItem,
    WebSource,
)
from ...types.request import Request
from ...types.response import (
    IncompleteReason,
    OpenRouterResponse,
    ResponseError,
    ResponseStatus,
    Usage,
)
from ...types.streaming.event_types import (
    CustomProviderEvent,
    FunctionCallArgumentDeltaEvent,
    OutputItemCompletedEvent,
    ReasoningStartedEvent,
    ReasoningSummaryDeltaEvent,
    ReasoningTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseStartedEvent,
    ServerToolCallStartedEvent,
    StreamEventUnion,
    TextDeltaEvent,
    ToolCallStartedEvent,
)
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

_HOSTED_TOOL_CALL_ITEMS = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_interpreter",
    "image_generation_call": "image_generation",
    "shell_call": "shell",
}

_WEB_SEARCH_STATUSES = {"in_progress", "searching", "completed", "failed", "incomplete"}

# the events that end a stream, which must not end up as a CustomProviderEvent
_TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed", "error"}


def _value(value: Any) -> Any:
    '''The SDK marks fields the response left out with an Unset object; they read as None here.'''
    return None if type(value).__name__ == "Unset" else value


class OpenRouterAdapter(Adapter):
    '''Translates canonical types to and from OpenRouter's Responses API, through the OpenRouter SDK.

        WebSearchTool maps to openrouter:web_search, and a ShellTool with a ContainerEnvironment to
        openrouter:shell. Depending on the model, their calls come back as OpenRouter's own items or as
        the upstream provider's (web_search_call, shell_call and shell_call_output); both are normalized.

        Not supported yet: prompt caching and the client tools (ComputerUseTool, a local ShellTool).
        ProviderTool definitions are sent unchanged.

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
        # a provider shell call is a shell_call and a shell_call_output, and both are replayed
        if isinstance(raw, list):
            return [self._native_item(item) for item in raw]

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
        if tool.cache:
            raise self._not_supported_yet("Prompt caching (cache=True)")

        if "type" in tool.provider_options:
            raise UnsupportedToolOptionError(
                "OpenRouter web search has no native tool versions to pin. Remove \"type\", or send "
                "another native definition in a ProviderTool."
            )

        # native fields such as engine or max_results; the typed fields win
        parameters = dict(tool.provider_options)

        if tool.max_uses is not None:
            parameters["max_uses"] = tool.max_uses

        if tool.allowed_domains is not None:
            parameters["allowed_domains"] = tool.allowed_domains

        if tool.blocked_domains is not None:
            parameters["excluded_domains"] = tool.blocked_domains

        if tool.user_location is not None:
            parameters["user_location"] = {
                "type": "approximate",
                **tool.user_location.model_dump(exclude_none=True),
            }

        native: dict[str, Any] = {"type": "openrouter:web_search"}
        if parameters:
            native["parameters"] = parameters

        return native

    def _native_container(self, container: ContainerEnvironment) -> dict[str, Any]:
        if container.memory_limit is not None:
            raise UnsupportedToolOptionError("OpenRouter containers do not support memory_limit. Remove it.")

        if container.container_id is not None:
            return {**container.provider_options, "type": "container_reference", "container_id": container.container_id}

        native: dict[str, Any] = {"type": "container_auto"}

        if container.file_ids is not None:
            native["file_ids"] = container.file_ids

        if container.network_disabled:
            native["network_policy"] = {"type": "disabled"}
        elif container.allowed_domains is not None:
            native["network_policy"] = {"type": "allowlist", "allowed_domains": container.allowed_domains}

        return {**container.provider_options, **native}

    def map_computer_use_tool(self, tool: ComputerUseTool) -> dict[str, Any]:
        raise self._not_supported_yet("ComputerUseTool")

    def map_shell_tool(self, tool: ShellTool) -> dict[str, Any]:
        if tool.environment == "local":
            raise self._not_supported_yet("A ShellTool with a local environment")

        if tool.cache:
            raise self._not_supported_yet("Prompt caching (cache=True)")

        # native fields such as engine are kept; the typed environment wins
        parameters = dict(tool.provider_options)
        parameters["environment"] = {
            **(tool.provider_options.get("environment") or {}),
            **self._native_container(tool.environment),
        }

        return {"type": "openrouter:shell", "parameters": parameters}

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

            # a status the SDK lets through but Tessaract doesn't know stays native
            case "web_search_call" | "openrouter:web_search" if output_item.status in _WEB_SEARCH_STATUSES:
                return self._normalize_web_search_call(output_item)

            case "openrouter:shell":
                return self._shell_item(output_item, output_item)

            case "shell_call":
                return self._shell_item(output_item, None)

            case "shell_call_output":
                return self._shell_item(None, output_item)

            case _:
                return self._provider_item(output_item)

    def _server_call_id(self, item) -> str | None:
        return _value(getattr(item, "call_id", None)) or getattr(item, "id", None)

    def _normalize_web_search_call(self, item) -> WebSearchCallOutputItem:
        action = item.action

        # openrouter:web_search and OpenAI's older items have one query; newer OpenAI items list them
        queries = getattr(action, "queries", None) or (
            [action.query] if getattr(action, "query", None) else []
        )

        return WebSearchCallOutputItem(
            raw=item,
            provider="openrouter",
            call_id=self._server_call_id(item) or "",
            status=item.status,
            action=getattr(action, "type", "search"),
            queries=queries,
            url=getattr(action, "url", None),
            pattern=getattr(action, "pattern", None),
            sources=[WebSource(url=source.url) for source in getattr(action, "sources", None) or []],
            error=getattr(item, "error", None),
        )

    def _shell_item(self, call, output) -> ShellCallOutputItem:
        '''Builds a provider shell call from its call and output: one openrouter:shell item passed as
            both, or a shell_call and a shell_call_output, either of which can be None. A call without
            output is still running.
        '''
        action = getattr(call, "action", None)
        native = call if call is not None else output
        raw = call if call is output else [item for item in (call, output) if item is not None]

        # the Literal has no "failed": a failed call is incomplete, with its error set
        status = _value(getattr(output if output is not None else call, "status", None))
        status = "incomplete" if status == "failed" else status

        return ShellCallOutputItem(
            raw=raw,
            provider="openrouter",
            executed_by="provider",
            call_id=self._server_call_id(native) or "",
            commands=list(action.commands) if action is not None else [],
            output=[
                ShellCommandOutput(
                    stdout=chunk.stdout,
                    stderr=chunk.stderr,
                    exit_code=getattr(chunk.outcome, "exit_code", 0),
                    timed_out=chunk.outcome.type == "timeout",
                )
                for chunk in getattr(output, "output", None) or []
            ],
            file_ids=[file.file_id for file in getattr(output, "files", None) or []],
            error=getattr(output, "error", None),
            timeout_ms=_value(action.timeout_ms) if action is not None else None,
            max_output_length=_value(action.max_output_length) if action is not None else None,
            status=status if status in {"in_progress", "completed", "incomplete"} else None,
        )

    def _normalize_output(self, output_items) -> list[OutputType]:
        output: list[OutputType] = []
        calls: dict[str, Any] = {}

        for native in output_items:
            native_type = getattr(native, "type", None)

            if native_type == "shell_call":
                calls[native.call_id] = native

            previous = output[-1] if output else None

            # only neighbours merge, so replaying the output keeps the native item order
            if (
                native_type == "shell_call_output"
                and isinstance(previous, ShellCallOutputItem)
                and isinstance(previous.raw, list)
                and [item.type for item in previous.raw] == ["shell_call"]
                and previous.call_id == native.call_id
            ):
                output[-1] = self._shell_item(previous.raw[0], native)

            elif native_type == "shell_call_output":
                # not next to its call: it stays its own item, and still gets the commands
                item = self._shell_item(None, native)
                call_action = getattr(calls.get(native.call_id), "action", None)
                if call_action is not None:
                    item.commands = list(call_action.commands)
                output.append(item)

            else:
                output.append(self._normalize_output_item(native))

        return output

    def _container_id(self, output_items) -> str | None:
        for item in output_items:
            if getattr(item, "type", None) in ("openrouter:shell", "shell_call_output"):
                container_id = getattr(item, "container_id", None)
                if container_id:
                    return container_id

        return None

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

    def _normalize_usage(self, provider_usage, output: list[OutputType]) -> Usage | None:
        if provider_usage is None:
            return None

        # counted from the output, since OpenRouter's own counts leave out the upstream provider's native tools
        server_tool_requests = {
            name: count
            for name, count in (
                ("web_search", sum(isinstance(item, WebSearchCallOutputItem) for item in output)),
                ("shell", sum(isinstance(item, ShellCallOutputItem) for item in output)),
            )
            if count
        }

        return Usage(
            input_tokens=provider_usage.input_tokens,
            output_tokens=provider_usage.output_tokens,
            total_tokens=provider_usage.total_tokens,
            cached_input_tokens=provider_usage.input_tokens_details.cached_tokens,
            cache_write_input_tokens=_value(provider_usage.input_tokens_details.cache_write_tokens),
            reasoning_tokens=provider_usage.output_tokens_details.reasoning_tokens,
            server_tool_requests=server_tool_requests or None,
        )

    def _normalize_response(self, raw_response) -> OpenRouterResponse:
        provider_usage = _value(raw_response.usage)
        error = _value(raw_response.error)
        output = self._normalize_output(raw_response.output)

        return OpenRouterResponse(
            id=raw_response.id,
            model=raw_response.model,
            status=self._normalize_status(raw_response.status),
            incomplete_reason=self._normalize_incomplete_reason(raw_response),
            output=output,
            container_id=self._container_id(raw_response.output),
            error=ResponseError(message=error.message, code=error.code) if error is not None else None,
            usage=self._normalize_usage(provider_usage, output),
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
        kwargs = self._build_request_kwargs(request)
        pending_shell_calls: dict[str, Any] = {}

        with self._client.responses.send(**kwargs, stream=True) as stream:
            for raw_event in stream:
                yield from self._normalize_stream_event(raw_event, pending_shell_calls)

    def _server_tool_name(self, item) -> str | None:
        '''The tool name of a call OpenRouter or the model's upstream provider runs, or None for any other item.'''
        if item.type in _HOSTED_TOOL_CALL_ITEMS:
            return _HOSTED_TOOL_CALL_ITEMS[item.type]

        if item.type.startswith("openrouter:") and item.type != "openrouter:armor_blocked":
            return item.type.removeprefix("openrouter:")

        return None

    def _normalize_unknown_event(self, event) -> StreamEventUnion:
        '''An event the SDK couldn't validate. It keeps the payload in .raw, under the type "UNKNOWN".'''
        native = event.raw
        provider_type = native.get("type", "unknown")

        if provider_type not in _TERMINAL_EVENTS:
            return CustomProviderEvent(provider_type=provider_type, raw_event=event)

        # the stream ends here, so this can't pass through as if nothing happened
        details = native.get("error") or (native.get("response") or {}).get("error") or native
        message = details.get("message") or f"The OpenRouter SDK couldn't read a {provider_type} event."
        code = details.get("code")

        return ResponseFailedEvent(
            message=message,
            error=ResponseError(message=message, code=str(code) if code is not None else None),
            raw_event=event,
        )

    def _normalize_stream_event(self, event, pending_shell_calls: dict[str, Any]) -> Iterator[StreamEventUnion]:
        '''Normalizes one native stream event.

            pending_shell_calls holds finished shell_call items by call_id until their shell_call_output
            arrives. Pass the same dict for every event of one stream.
        '''
        is_unknown = getattr(event, "is_unknown", False)

        # a call whose output never arrived still gets its output_item.done before the stream ends
        if (event.raw.get("type") if is_unknown else event.type) in _TERMINAL_EVENTS:
            for call in pending_shell_calls.values():
                yield OutputItemCompletedEvent(item=self._shell_item(call, None), raw_event=event)
            pending_shell_calls.clear()

        if is_unknown:
            yield self._normalize_unknown_event(event)
            return

        match event.type:
            case "response.created":
                yield ResponseStartedEvent(raw_event=event)

            # a response cut short still ends normally, with its incomplete_reason, as it does without streaming
            case "response.completed" | "response.incomplete":
                yield ResponseCompletedEvent(
                    response=self._normalize_response(event.response),
                    raw_event=event,
                )

            case "response.failed":
                response = self._normalize_response(event.response)
                error = response.error or ResponseError(message="Response failed")
                yield ResponseFailedEvent(
                    message=error.message,
                    error=error,
                    response=response,
                    raw_event=event,
                )

            case "error":
                yield ResponseFailedEvent(
                    message=event.message,
                    error=ResponseError(message=event.message, code=event.code),
                    raw_event=event,
                )

            case "response.output_item.added" if event.item.type == "function_call":
                yield ToolCallStartedEvent(
                    call_id=event.item.call_id,
                    name=event.item.name,
                    output_index=event.output_index,
                    raw_event=event,
                )

            # started from the item, since models that only return reasoning text have no summary parts
            case "response.output_item.added" if event.item.type == "reasoning":
                yield ReasoningStartedEvent(
                    item_id=event.item.id,
                    output_index=event.output_index,
                    raw_event=event,
                )

            case "response.output_item.added" if (
                (tool_name := self._server_tool_name(event.item))
                and (call_id := self._server_call_id(event.item))
            ):
                yield ServerToolCallStartedEvent(
                    call_id=call_id,
                    name=tool_name,
                    output_index=event.output_index,
                    raw_event=event,
                )

            # a provider shell call is done when its output arrives; output_item.done then carries both items
            case "response.output_item.done" if event.item.type == "shell_call":
                pending_shell_calls[event.item.call_id] = event.item

            case "response.output_item.done" if event.item.type == "shell_call_output":
                yield OutputItemCompletedEvent(
                    item=self._shell_item(pending_shell_calls.pop(event.item.call_id, None), event.item),
                    output_index=event.output_index,
                    raw_event=event,
                )

            case "response.output_item.done":
                yield OutputItemCompletedEvent(
                    item=self._normalize_output_item(event.item),
                    output_index=event.output_index,
                    raw_event=event,
                )

            case "response.output_text.delta":
                yield TextDeltaEvent(
                    delta=event.delta,
                    provider="openrouter",
                    output_index=event.output_index,
                    content_index=event.content_index,
                    item_id=event.item_id,
                    raw_event=event,
                )

            case "response.reasoning_summary_text.delta":
                yield ReasoningSummaryDeltaEvent(
                    delta=event.delta,
                    output_index=event.output_index,
                    item_id=event.item_id,
                    raw_event=event,
                )

            case "response.reasoning_text.delta":
                yield ReasoningTextDeltaEvent(
                    delta=event.delta,
                    output_index=event.output_index,
                    content_index=event.content_index,
                    item_id=event.item_id,
                    raw_event=event,
                )

            case "response.function_call_arguments.delta":
                yield FunctionCallArgumentDeltaEvent(
                    delta=event.delta,
                    output_index=event.output_index,
                    item_id=event.item_id,
                    raw_event=event,
                )

            case _:
                yield CustomProviderEvent(provider_type=event.type, raw_event=event)

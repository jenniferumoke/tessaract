import base64
import json
from collections.abc import Iterator, Sequence
from typing import Any, cast

from openai import Omit
from openai.types.shared_params import Reasoning as OpenAIReasoningParams

from ...errors import UnsupportedToolOptionError
from ...providers.openai_provider import OpenAIProvider
from ...tools.computer import ComputerUseTool
from ...tools.function import InputSchema, Property
from ...tools.shell import ContainerEnvironment, ShellTool
from ...tools.web_search import WebSearchTool
from ...types import computer_actions as actions
from ...types.input_types import ComputerToolResult, ShellToolResult
from ...types.output_types import (
    OutputItem,
    Annotation,
    AssistantMessage,
    ComputerCallOutputItem,
    FunctionCallOutputItem,
    ProviderOutputItem,
    SafetyCheck,
    ShellCallOutputItem,
    ShellCommandOutput,
    OutputType,
    ReasoningOutputItem,
    TextOutputItem,
    WebSearchCallOutputItem,
    WebSource,
)
from ...types.request import Request
from ...types.response import (
    IncompleteReason,
    OpenAIResponse,
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
    FunctionToolResultProtocol,
    FunctionToolSchemaProtocol,
    CacheParamsProtocol,
    ReasoningParamsProtocol,
    SystemPromptProtocol,
    UserMessageProtocol,
    AssistantMessageProtocol
)

# Without this include, web_search_call items come back without their sources
WEB_SEARCH_SOURCES_INCLUDE = "web_search_call.action.sources"

# output items for tools OpenAI runs itself, by the tool name used in ServerToolCallStartedEvent
_SERVER_TOOL_CALL_ITEMS = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_interpreter",
    "image_generation_call": "image_generation",
}


# OpenAI mouse button names; OpenAI calls the middle button "wheel"
_MOUSE_BUTTONS: dict[str, actions.MouseButton] = {
    "left": "left", "right": "right", "wheel": "middle", "back": "back", "forward": "forward",
}


class OpenAIAdapter(Adapter):
    provider_name = "openai"

    def __init__(self, provider: OpenAIProvider):
        super().__init__(provider)
        self._client = provider.client

    def map_input_message(self, item: UserMessageProtocol | SystemPromptProtocol):
        return {
            "role": item.role,
            "content": self._mark_last_part(item.content) if item.cache else item.content
        }

    def _mark_last_part(self, content) -> list:
        '''Adds an explicit prompt_cache_breakpoint to the last content part. One it already has is kept.

            Only gpt-5.6 and later accept breakpoints; older models reject them.
        '''
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else list(content)

        if not parts or "prompt_cache_breakpoint" in parts[-1]:
            return parts

        return [*parts[:-1], {**parts[-1], "prompt_cache_breakpoint": {"mode": "explicit"}}]

    def map_assistant_message(self, item: AssistantMessageProtocol):
        return item.raw

    def map_reasoning_params(self, reasoning: ReasoningParamsProtocol | None) -> OpenAIReasoningParams | Omit | None:
        if reasoning is None:
            return Omit()

        if reasoning.budget is not None:
            raise ValueError("Not a valid reasoning configuration option for OpenAI. Remove it.") 

        native_reasoning: OpenAIReasoningParams = {}

        mode = reasoning.mode
        match mode:
            case None:
                pass
            case "standard" | "pro":
                native_reasoning["mode"] = mode
            case _:
                raise ValueError(
                    f"OpenAI does not support mode: {mode!r}."
                    'Use "standard" or "pro".'
                )

        summary = reasoning.summary
        match summary:
            case None:
                pass
            case "auto" | "concise" | "detailed":
                native_reasoning["summary"] = summary
            case _:
                raise ValueError(
                    f"OpenAI does not accept reasoning summary {summary!r}."
                    'Use "auto", "concise" or "detailed".'
                )

        if reasoning.effort is not None:
            native_reasoning["effort"] = (
                "xhigh" if reasoning.effort == "extra_high" else reasoning.effort
            )

        # native_options add native reasoning fields; the typed fields win
        return cast(OpenAIReasoningParams, {**(reasoning.native_options or {}), **native_reasoning})

    def map_cache_params(self, cache: CacheParamsProtocol | None) -> dict[str, Any]:
        '''Maps CacheOptions.mode to prompt_cache_options.mode and key to prompt_cache_key.

            prompt_cache_options is only sent when mode is set, since only gpt-5.6 and later accept it.
            CacheOptions.retention is ignored: prompt_cache_retention is deprecated, gpt-5.5 rejects
            in_memory, and gpt-5.6's prompt_cache_options.ttl accepts only its default, so no value
            maps safely across current models. Use request_options for model-specific cache fields.
        '''
        if cache is None:
            return {}

        native_cache: dict[str, Any] = {}

        if cache.mode is not None:
            native_cache["prompt_cache_options"] = {"mode": cache.mode}

        if cache.key is not None:
            native_cache["prompt_cache_key"] = cache.key

        return native_cache

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

    def  map_function_schema(self, tools: Sequence[FunctionToolSchemaProtocol]) -> list:

        _native_tools_list = []
        for tool in tools:
            _native_tool_schema: dict[str, object] = {}

            _native_tool_schema["type"] = "function"
            _native_tool_schema["name"] = tool.name
            _native_tool_schema["description"] = tool.description
            _native_tool_schema["strict"] = tool.strict if tool.strict is not None else True

            if tool.input_schema is not None:
                _native_tool_schema["parameters"] = self._native_tool_parameters(tool.input_schema) if tool.input_schema is not None else None

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

    def map_web_search_tool(self, tool: WebSearchTool) -> dict[str, Any]:
        unsupported = [
            name for name in ("max_uses", "blocked_domains")
            if getattr(tool, name) is not None
        ]
        if unsupported:
            raise UnsupportedToolOptionError(
                f"OpenAI web_search does not support {', '.join(unsupported)}. Remove "
                f"{'them' if len(unsupported) > 1 else 'it'}, or pass native fields via provider_options."
            )

        native: dict[str, Any] = {"type": tool.provider_options.get("type", "web_search")}

        if tool.allowed_domains is not None:
            # keep other native filters; the typed allowed_domains wins
            native["filters"] = {
                **(tool.provider_options.get("filters") or {}),
                "allowed_domains": tool.allowed_domains,
            }

        if tool.user_location is not None:
            native["user_location"] = self._native_user_location(tool.user_location)

        # tool.cache is ignored, as on function tools: OpenAI caches tool definitions automatically
        return {**tool.provider_options, **native}

    def map_computer_use_tool(self, tool: ComputerUseTool) -> dict[str, Any]:
        # tool.cache is ignored, as on function tools: OpenAI caches tool definitions automatically
        return {**tool.provider_options, "type": tool.provider_options.get("type", "computer")}

    def _native_container(self, container: ContainerEnvironment) -> dict[str, Any]:
        if container.container_id is not None:
            return {**container.provider_options, "type": "container_reference", "container_id": container.container_id}

        native: dict[str, Any] = {"type": "container_auto"}

        if container.file_ids is not None:
            native["file_ids"] = container.file_ids

        if container.memory_limit is not None:
            native["memory_limit"] = container.memory_limit

        if container.network_disabled:
            native["network_policy"] = {"type": "disabled"}
        elif container.allowed_domains is not None:
            # keep other native network policy fields, such as domain_secrets; the typed domains win
            native["network_policy"] = {
                **(container.provider_options.get("network_policy") or {}),
                "type": "allowlist",
                "allowed_domains": container.allowed_domains,
            }

        return {**container.provider_options, **native}

    def map_shell_tool(self, tool: ShellTool) -> dict[str, Any]:
        environment = (
            {"type": "local"} if tool.environment == "local" else self._native_container(tool.environment)
        )

        # native environment fields in provider_options, such as skills, are kept; the typed environment wins
        native_environment = tool.provider_options.get("environment") or {}
        environment = {**native_environment, **environment}

        return {**tool.provider_options, "type": "shell", "environment": environment}

    def _extra_fields(self, provider_options: dict[str, Any], canonical_keys: set[str]) -> dict[str, Any]:
        return {key: value for key, value in provider_options.items() if key not in canonical_keys}

    def map_computer_tool_result(self, item: ComputerToolResult) -> dict[str, Any]:
        # item.output is ignored: computer_call_output only carries a screenshot, and OpenAI never
        # sends the cursor_position action that needs text. item.cache is ignored, as on tool definitions.
        if item.is_error:
            raise UnsupportedToolOptionError(
                "OpenAI computer_call_output has no error flag. Send a screenshot of the current "
                "screen instead, so the model can see what happened."
            )

        if item.screenshot is None:
            raise ValueError("OpenAI needs a screenshot in every ComputerToolResult.")

        native: dict[str, Any] = {
            "type": "computer_call_output",
            "call_id": item.call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": "data:image/png;base64," + base64.b64encode(item.screenshot).decode("ascii"),
            },
        }

        if item.acknowledged_safety_checks:
            native["acknowledged_safety_checks"] = [
                check.model_dump(exclude_none=True) for check in item.acknowledged_safety_checks
            ]

        return {**self._extra_fields(item.provider_options, set(native)), **native}

    def map_shell_tool_result(self, item: ShellToolResult) -> dict[str, Any]:
        # item.cache is ignored: shell_call_output has no content parts to put a breakpoint on
        if item.is_error:
            raise UnsupportedToolOptionError(
                "OpenAI shell_call_output has no error flag. Report what happened in stderr, with a non-zero exit_code."
            )

        native: dict[str, Any] = {
            "type": "shell_call_output",
            "call_id": item.call_id,
            "output": [
                {
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                    "outcome": {"type": "timeout"} if output.timed_out else {"type": "exit", "exit_code": output.exit_code},
                }
                for output in item.output
            ],
        }

        if item.max_output_length is not None:
            native["max_output_length"] = item.max_output_length

        return {**self._extra_fields(item.provider_options, set(native)), **native}

    def _is_content_parts(self, result) -> bool:
        return isinstance(result, list) and all(
            isinstance(part, dict) and "type" in part for part in result
        )

    def map_tool_result(self, item: FunctionToolResultProtocol):
        # OpenAI has no error flag on function_call_output, so errors are marked in the output itself
        if item.is_error:
            output = json.dumps({"error": item.result}, default=str)

        elif isinstance(item.result, str) or self._is_content_parts(item.result):
            output = item.result

        else:
            output = json.dumps(item.result, default=str)

        if item.cache:
            output = self._mark_last_part(output)

        canonical_keys = {"type", "call_id", "output"}
        extra_fields = {
            key: value
            for key, value in item.provider_options.items()
            if key not in canonical_keys
        }

        return {
            **extra_fields,
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": output
        }

        
    def _normalize_annotation(self, annotation) -> Annotation:
        provider_type = annotation.type
        return Annotation(
            type="file_path" if provider_type == "file_path" else "citation",
            provider="openai",
            provider_type=provider_type,
            source=getattr(annotation, "url", None) or getattr(annotation, "file_id", None),
            title=getattr(annotation, "title", None) or getattr(annotation, "filename", None),
            cited_text=getattr(annotation, "cited_text", None),
            provider_metadata=annotation.model_dump(mode="json"),
        )

    def _normalize_output_item(self, output_item) -> OutputType:
        if output_item.type == "message" and (
            output_item.content is None or any(
                part.type != "output_text"
                for part in output_item.content
            )
        ):
            return ProviderOutputItem(raw=output_item, provider_type=output_item.type)
        
        match output_item.type:
            case "message":
                return AssistantMessage(
                        raw=output_item,
                        content=[
                            TextOutputItem(
                                raw=i,
                                text=i.text,
                                annotations=[
                                    self._normalize_annotation(annotation)
                                    for annotation in i.annotations
                                ]
                            )
                            for i in output_item.content
                        ]
                    )

            case "reasoning":
                return ReasoningOutputItem(
                    raw=output_item,
                    id=output_item.id,
                    content="".join(part.text for part in output_item.content),
                    text="".join(part.text for part in output_item.summary)
                ) 

            case "function_call":
                return FunctionCallOutputItem(
                    raw=output_item,
                    call_id=output_item.call_id,
                    name=output_item.name,
                    arguments=json.loads(output_item.arguments)
                )

            case "web_search_call":
                return self._normalize_web_search_call(output_item)

            case "computer_call":
                return self._normalize_computer_call(output_item)

            case "shell_call":
                return self._shell_item(output_item, None)

            # only calls in a container have their output in the response; OpenAI ran them
            case "shell_call_output":
                return self._shell_item(None, output_item)

            case _:
                return ProviderOutputItem(raw=output_item, provider_type=output_item.type)


    def _normalize_web_search_call(self, output_item) -> WebSearchCallOutputItem:
        action = output_item.action

        # older responses have a single query; newer ones list every query in queries
        queries = getattr(action, "queries", None) or (
            [action.query] if getattr(action, "query", None) else []
        )

        return WebSearchCallOutputItem(
            raw=output_item,
            provider="openai",
            call_id=output_item.id,
            status=output_item.status,
            action=action.type,
            queries=queries,
            url=getattr(action, "url", None),
            pattern=getattr(action, "pattern", None),
            sources=[
                WebSource(url=source.url)
                for source in getattr(action, "sources", None) or []
            ],
        )

    def _is_local_shell(self, item) -> bool:
        # environment is None or local for calls your code runs, and a container for calls OpenAI runs
        return getattr(getattr(item, "environment", None), "type", "local") == "local"

    def _shell_item(self, call, output) -> ShellCallOutputItem:
        '''Builds a shell call from its shell_call item, its shell_call_output item, or both.

            A local call has no output item: your code runs it. A call in a container that has no output
            item yet is still running.
        '''
        if call is not None and self._is_local_shell(call):
            return ShellCallOutputItem(
                raw=call,
                provider="openai",
                call_id=call.call_id,
                commands=list(call.action.commands),
                timeout_ms=call.action.timeout_ms,
                max_output_length=call.action.max_output_length,
                status=call.status,
            )

        action = getattr(call, "action", None)
        native = call if call is not None else output

        return ShellCallOutputItem(
            raw=[item for item in (call, output) if item is not None],
            provider="openai",
            executed_by="provider",
            call_id=native.call_id,
            commands=list(action.commands) if action is not None else [],
            output=[
                ShellCommandOutput(
                    stdout=chunk.stdout,
                    stderr=chunk.stderr,
                    exit_code=getattr(chunk.outcome, "exit_code", 0),
                    timed_out=chunk.outcome.type == "timeout",
                )
                for chunk in (output.output if output is not None else [])
            ],
            timeout_ms=action.timeout_ms if action is not None else None,
            max_output_length=output.max_output_length if output is not None else (
                action.max_output_length if action is not None else None
            ),
            status=output.status if output is not None else native.status,
        )

    def _normalize_computer_action(self, action) -> actions.ComputerAction:
        keys = list(getattr(action, "keys", None) or [])

        try:
            match action.type:
                case "click":
                    return actions.Click(
                        x=action.x, y=action.y, button=_MOUSE_BUTTONS[action.button], modifiers=keys,
                    )
                case "double_click":
                    return actions.Click(x=action.x, y=action.y, clicks=2, modifiers=keys)
                case "drag":
                    return actions.Drag(path=[actions.Point(x=p.x, y=p.y) for p in action.path], modifiers=keys)
                case "keypress":
                    return actions.KeyPress(keys=keys)
                case "move":
                    return actions.MoveMouse(x=action.x, y=action.y, modifiers=keys)
                case "screenshot":
                    return actions.Screenshot()
                case "scroll":
                    return actions.Scroll(
                        x=action.x, y=action.y, dx=action.scroll_x, dy=action.scroll_y, unit="pixels", modifiers=keys,
                    )
                case "type":
                    return actions.TypeText(text=action.text)
                case "wait":
                    return actions.Wait()
        except (AttributeError, KeyError, ValueError):
            pass

        # a newer action type, or one whose fields changed, is kept with its native payload
        return actions.ProviderComputerAction(provider_type=action.type, payload=action.model_dump(mode="json"))

    def _normalize_computer_call(self, output_item) -> ComputerCallOutputItem:
        native_actions = output_item.actions if output_item.actions is not None else (
            [output_item.action] if output_item.action is not None else []
        )

        return ComputerCallOutputItem(
            raw=output_item,
            provider="openai",
            call_id=output_item.call_id,
            actions=[self._normalize_computer_action(action) for action in native_actions],
            status=output_item.status,
            safety_checks=[
                SafetyCheck(id=check.id, code=check.code, message=check.message)
                for check in output_item.pending_safety_checks or []
            ],
        )

    def map_output_item(self, item: OutputItem) -> Any:
        # a provider shell call's raw is a list: its shell_call and shell_call_output items
        return list(item.raw) if isinstance(item.raw, list) else item.raw

    def _normalize_output(self, output_items) -> list[OutputType]:
        output: list[OutputType] = []
        # provider shell calls by call_id, so an output that isn't next to its call still gets the commands
        shell_calls: dict[str, ShellCallOutputItem] = {}

        for native in output_items:
            item = self._normalize_output_item(native)
            previous = output[-1] if output else None

            if isinstance(item, ShellCallOutputItem) and native.type == "shell_call":
                shell_calls[item.call_id] = item

            # a shell_call_output directly after its shell_call completes that call. Only neighbouring
            # items are merged, so replaying the output keeps the native item order.
            if (
                native.type == "shell_call_output"
                and isinstance(previous, ShellCallOutputItem)
                and isinstance(previous.raw, list)
                and len(previous.raw) == 1
                and previous.raw[0].type == "shell_call"
                and previous.call_id == native.call_id
            ):
                output[-1] = self._shell_item(previous.raw[0], native)

            elif native.type == "shell_call_output" and isinstance(item, ShellCallOutputItem) and item.call_id in shell_calls:
                call = shell_calls[item.call_id]
                item.commands, item.timeout_ms = call.commands, call.timeout_ms
                output.append(item)

            else:
                output.append(item)

        return output

    def _normalize_stream_event(self, event, pending_shell_calls: dict[str, Any] | None = None):
        '''Normalizes one native stream event.

            pending_shell_calls holds finished shell_call items OpenAI is running, by call_id, until their
            shell_call_output arrives. Pass the same dict for every event of one stream.
        '''
        if pending_shell_calls is None:
            pending_shell_calls = {}

        match event.type:
            case "response.created":
                yield ResponseStartedEvent(
                    raw_event=event
                )

            case "response.completed":
                yield ResponseCompletedEvent(
                    response=self._normalize_response(event.response),
                    raw_event=event
                )

            case "response.failed":
                response = self._normalize_response(event.response)
                error = response.error or ResponseError(message="Response failed")
                yield ResponseFailedEvent(
                    message=error.message,
                    error=error,
                    response=response,
                    raw_event=event
                )

            case "error":
                yield ResponseFailedEvent(
                    message=event.message,
                    error=ResponseError(message=event.message, code=event.code),
                    raw_event=event
                )

            case "response.output_item.added" if event.item.type == "function_call":
                yield ToolCallStartedEvent(
                    call_id=event.item.call_id,
                    name=event.item.name,
                    output_index=event.output_index,
                    raw_event=event
                )

            case "response.output_item.added" if event.item.type == "computer_call" or (
                event.item.type == "shell_call" and self._is_local_shell(event.item)
            ):
                yield ToolCallStartedEvent(
                    call_id=event.item.call_id,
                    name="computer" if event.item.type == "computer_call" else "shell",
                    call_type=event.item.type,
                    output_index=event.output_index,
                    raw_event=event
                )

            case "response.output_item.added" if event.item.type == "shell_call":
                # a shell call in a container, which OpenAI runs
                yield ServerToolCallStartedEvent(
                    call_id=event.item.call_id,
                    name="shell",
                    output_index=event.output_index,
                    raw_event=event
                )

            case "response.output_item.added" if event.item.type in _SERVER_TOOL_CALL_ITEMS:
                yield ServerToolCallStartedEvent(
                    call_id=event.item.id,
                    name=_SERVER_TOOL_CALL_ITEMS[event.item.type],
                    output_index=event.output_index,
                    raw_event=event
                )

            case "response.output_text.delta":
                yield TextDeltaEvent(
                    delta=event.delta,
                    provider="openai",
                    output_index=event.output_index,
                    content_index=event.content_index,
                    raw_event=event
                )

            # a shell call OpenAI runs is done when its output arrives; output_item.done then carries both items
            case "response.output_item.done" if event.item.type == "shell_call" and not self._is_local_shell(event.item):
                pending_shell_calls[event.item.call_id] = event.item

            case "response.output_item.done" if event.item.type == "shell_call_output":
                yield OutputItemCompletedEvent(
                    item=self._shell_item(pending_shell_calls.pop(event.item.call_id, None), event.item),
                    raw_event=event
                )

            case "response.output_item.done":
                yield OutputItemCompletedEvent(
                    item=self._normalize_output_item(event.item),
                    raw_event=event
                )

            case "response.reasoning_summary_text.delta":
                yield ReasoningSummaryDeltaEvent(
                    delta=event.delta,
                    output_index=event.output_index,
                    item_id=event.item_id,
                    raw_event=event
                )

            case "response.reasoning_text.delta":
                yield ReasoningTextDeltaEvent(
                    delta=event.delta,
                    output_index=event.output_index,
                    item_id=event.item_id,
                    raw_event=event
                )

            case "response.function_call_arguments.delta":
                yield FunctionCallArgumentDeltaEvent(
                    delta=event.delta,
                    output_index=event.output_index,
                    raw_event=event
                )

            case "response.reasoning_summary_part.added":
                yield ReasoningStartedEvent(
                    item_id=event.item_id,
                    raw_event=event
                )

            case _:
                yield CustomProviderEvent(
                    provider_type=event.type,
                    raw_event=event
                )

    def _normalize_status(self, status) -> ResponseStatus | None:
        try:
            return ResponseStatus(status) if status is not None else None
        except ValueError:
            return None

    def _normalize_incomplete_reason(self, raw_response) -> IncompleteReason | None:
        details = getattr(raw_response, "incomplete_details", None)
        match getattr(details, "reason", None):
            case "max_output_tokens":
                return IncompleteReason.MAX_TOKENS
            case "content_filter":
                return IncompleteReason.CONTENT_FILTER
            case _:
                return None

    def _normalize_response(self, raw_response) -> OpenAIResponse:
        output = self._normalize_output(raw_response.output)

        provider_usage = raw_response.usage
        usage = None
        if provider_usage is not None:
            # OpenAI doesn't report server tool counts, so they are counted from the output
            web_searches = sum(isinstance(item, WebSearchCallOutputItem) for item in output)

            input_details = provider_usage.input_tokens_details
            output_details = provider_usage.output_tokens_details
            usage = Usage(
                input_tokens=provider_usage.input_tokens,
                output_tokens=provider_usage.output_tokens,
                total_tokens=provider_usage.total_tokens,
                cached_input_tokens=input_details.cached_tokens if input_details is not None else None,
                cache_write_input_tokens=input_details.cache_write_tokens if input_details is not None else None,
                reasoning_tokens=output_details.reasoning_tokens if output_details is not None else None,
                server_tool_requests={"web_search": web_searches} if web_searches else None,
            )

        return OpenAIResponse(
            id=raw_response.id,
            model=raw_response.model,
            status=self._normalize_status(raw_response.status),
            incomplete_reason=self._normalize_incomplete_reason(raw_response),
            output=output,
            error=ResponseError(message=raw_response.error.message, code=raw_response.error.code) if raw_response.error else None,
            usage=usage,
            provider_usage=provider_usage,
            raw_response=raw_response,
        )

    def _build_request_kwargs(self, request: Request):
        canonical_params: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
            "reasoning": self.map_reasoning_params(request.reasoning), 
        }

        provider_options = dict(**request.provider_options or {})

        # only sent when set, so native tools passed in request_options are kept
        if request.tools:
            canonical_params["tools"] = self.map_tools(request.tools)

        if any(isinstance(tool, WebSearchTool) for tool in request.tools or []):
            include = list(provider_options.get("include") or [])
            if WEB_SEARCH_SOURCES_INCLUDE not in include:
                include.append(WEB_SEARCH_SOURCES_INCLUDE)
            canonical_params["include"] = include

        if request.max_tokens is not None:
            canonical_params["max_output_tokens"] = request.max_tokens

        if request.instructions is not None:
            canonical_params["instructions"] = request.instructions

        cache_params = self.map_cache_params(request.cache)

        if "prompt_cache_options" in cache_params:
            # keep other native prompt_cache_options fields, such as ttl; the typed mode wins
            cache_params["prompt_cache_options"] = {
                **(provider_options.get("prompt_cache_options") or {}),
                **cache_params["prompt_cache_options"],
            }

        canonical_params.update(cache_params)

        extra_body = provider_options.get("extra_body")

        if extra_body is not None:
            provider_options["extra_body"] = {
                key: value
                for key, value in extra_body.items()
                if key not in canonical_params
            }


        return {**provider_options, **canonical_params}

    def generate_sync(self, request: Request) -> OpenAIResponse:

        kwargs = self._build_request_kwargs(request)

        _raw_response = self._client.responses.create(**kwargs, stream=False)

        return self._normalize_response(_raw_response)

    def generate_stream(self, request: Request) -> Iterator[StreamEventUnion]:
    
        kwargs = self._build_request_kwargs(request)

        pending_shell_calls: dict[str, Any] = {}

        with self._client.responses.stream(**kwargs) as stream:
            for raw_event in stream:
                yield from self._normalize_stream_event(raw_event, pending_shell_calls)

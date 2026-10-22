import base64
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..adapter import (
    Adapter,
    AssistantMessageProtocol,
    FunctionToolResultProtocol,
    FunctionToolSchemaProtocol,
    CacheParamsProtocol,
    ReasoningParamsProtocol,
    UserMessageProtocol,
    SystemPromptProtocol
)

from ...errors import UnsupportedToolOptionError
from ...providers.anthropic_provider import AnthropicProvider

from pydantic import ValidationError

from ...tools.computer import ComputerUseTool
from ...tools.function import InputSchema, Property
from ...tools.shell import ShellTool
from ...tools.web_search import WebSearchTool
from ...types import computer_actions as actions
from ...types.input_types import ComputerToolResult, ProviderToolResult, ShellToolResult
from ...types.output_types import (
    Annotation,
    AssistantMessage,
    ComputerCallOutputItem,
    FunctionCallOutputItem,
    OutputItem,
    OutputType,
    ProviderOutputItem,
    ReasoningOutputItem,
    ShellCallOutputItem,
    TextOutputItem,
    WebSearchCallOutputItem,
    WebSource,
)
from ...types.request import Request
from ...types.response import (
    AnthropicResponse,
    IncompleteReason,
    ResponseError,
    ResponseStatus,
    StopReason,
    Usage,
)
from ...types.streaming.event_types import (
    CustomProviderEvent,
    FunctionCallArgumentDeltaEvent,
    OutputItemCompletedEvent,
    ReasoningStartedEvent,
    ReasoningSummaryDeltaEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseStartedEvent,
    ServerToolCallStartedEvent,
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

# The web search tool version sent when provider_options doesn't pin one. It works on every current model.
DEFAULT_WEB_SEARCH_TOOL_TYPE = "web_search_20250305"

# Client tool versions sent when provider_options doesn't pin one
DEFAULT_COMPUTER_TOOLSET_TYPE = "computer_toolset_20260801"
DEFAULT_BASH_TOOL_TYPE = "bash_20250124"

# Computer action tool_use blocks carry this toolset_name; their name is the member, e.g. "left_click"
COMPUTER_TOOLSET_NAME = "computer"

_CLICK_MEMBERS: dict[str, tuple[actions.MouseButton, int]] = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}

_SCROLL_DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# ServerToolUsage fields, by the tool name used in Usage.server_tool_requests
_SERVER_TOOL_USAGE_FIELDS = {"web_search": "web_search_requests", "web_fetch": "web_fetch_requests"}


@dataclass
class _StreamState:
    '''What one stream has seen so far. Content block deltas don't say which kind of block they belong to.'''
    # indexes of server_tool_use blocks, whose input deltas aren't function arguments
    server_tool_indexes: set[int] = field(default_factory=set)
    # finished web search server_tool_use blocks, by id, waiting for their result block
    web_search_calls: dict[str, Any] = field(default_factory=dict)
    # the request declared a ShellTool, so bash tool_use blocks are shell calls
    shell_tool: bool = False


class AnthropicAdapter(Adapter):
    provider_name = "anthropic"

    def __init__(self, provider: AnthropicProvider):
        super().__init__(provider)
        self._client = provider.client

    def map_input_message(self, item: UserMessageProtocol | SystemPromptProtocol):
        # system messages are moved to the top-level system parameter in _build_request_kwargs
        return {
            "role": item.role,
            "content": self._mark_last_block(item.content) if item.cache else item.content
        }

    def _with_cache_control(self, block: dict) -> dict:
        '''Adds a cache breakpoint to a block. A cache_control the block already has is kept.'''
        if "cache_control" in block:
            return block
        return {**block, "cache_control": {"type": "ephemeral"}}

    def _mark_last_block(self, content) -> list:
        blocks = self._as_blocks(content)
        if not blocks:
            return blocks
        return [*blocks[:-1], self._with_cache_control(blocks[-1])]

    def _apply_ttl(self, blocks: list, ttl: str) -> list:
        '''Sets ttl on every cache breakpoint that doesn't have one.

            Anthropic requires longer-TTL entries to come before shorter ones, so block breakpoints
            must not be shorter than the automatic breakpoint at the end of the prompt.
        '''
        return [
            {**block, "cache_control": {**block["cache_control"], "ttl": ttl}}
            if isinstance(block, dict)
            and isinstance(block.get("cache_control"), dict)
            and "ttl" not in block["cache_control"]
            else block
            for block in blocks
        ]

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
        # thinking signatures included, and merged into one assistant turn in _build_request_kwargs.
        # A web search call's raw is a list: its server_tool_use block and its result block.
        blocks = item.raw if isinstance(item.raw, list) else [item.raw]
        return {
            "role": "assistant",
            "content": [self._block_param(block) for block in blocks]
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

        if item.cache:
            block = self._with_cache_control(block)

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

    def map_cache_params(self, cache: CacheParamsProtocol | None) -> dict[str, Any]:
        '''Maps CacheOptions to Anthropic's top-level cache_control, which turns on automatic caching.

            Mode "explicit" leaves it out, so only block-level breakpoints are cached.
            CacheOptions.key has no Anthropic equivalent and is ignored.
        '''
        if cache is None or cache.mode == "explicit":
            return {}

        cache_control: dict[str, Any] = {"type": "ephemeral"}

        if cache.retention is not None:
            cache_control["ttl"] = "1h" if cache.retention == "long" else "5m"

        return {"cache_control": cache_control}

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

            # a cache_control in provider_options is kept, as on other blocks
            if tool.cache and "cache_control" not in tool.provider_options:
                _native_tool_schema["cache_control"] = {"type": "ephemeral"}

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

    def map_web_search_tool(self, tool: WebSearchTool) -> dict[str, Any]:
        native: dict[str, Any] = {
            "type": tool.provider_options.get("type", DEFAULT_WEB_SEARCH_TOOL_TYPE),
            "name": "web_search",
        }

        for name in ("max_uses", "allowed_domains", "blocked_domains"):
            value = getattr(tool, name)
            if value is not None:
                native[name] = value

        if tool.user_location is not None:
            native["user_location"] = self._native_user_location(tool.user_location)

        # a cache_control in provider_options is kept, as on other blocks
        if tool.cache and "cache_control" not in tool.provider_options:
            native["cache_control"] = {"type": "ephemeral"}

        return {**tool.provider_options, **native}

    def map_computer_use_tool(self, tool: ComputerUseTool) -> dict[str, Any]:
        tool_type = tool.provider_options.get("type", DEFAULT_COMPUTER_TOOLSET_TYPE)

        if not tool_type.startswith("computer_toolset_"):
            raise ValueError(
                f"ComputerUseTool maps to Anthropic's computer toolset, not {tool_type!r}, whose calls have "
                "a different shape. Use a ProviderTool for earlier computer tool versions."
            )

        native: dict[str, Any] = {"type": tool_type}

        # a cache_control in provider_options is kept, as on other blocks
        if tool.cache and "cache_control" not in tool.provider_options:
            native["cache_control"] = {"type": "ephemeral"}

        return {**tool.provider_options, **native}

    def map_shell_tool(self, tool: ShellTool) -> dict[str, Any]:
        if tool.environment != "local":
            raise UnsupportedToolOptionError(
                "Anthropic's bash tool only runs in your environment, so ShellTool(environment=ContainerEnvironment(...)) "
                "is not supported. Anthropic runs commands in its own containers through the code execution tool; "
                "use a ProviderTool for it."
            )

        native: dict[str, Any] = {
            "type": tool.provider_options.get("type", DEFAULT_BASH_TOOL_TYPE),
            "name": "bash",
        }

        if tool.cache and "cache_control" not in tool.provider_options:
            native["cache_control"] = {"type": "ephemeral"}

        return {**tool.provider_options, **native}

    def _tool_result_block(self, call_id: str, content, is_error: bool, cache: bool, provider_options: dict) -> dict:
        canonical_keys = {"type", "tool_use_id", "content", "is_error"}
        block = {
            **{key: value for key, value in provider_options.items() if key not in canonical_keys},
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": content,
        }

        if is_error:
            block["is_error"] = True

        return self._with_cache_control(block) if cache else block

    def map_computer_tool_result(self, item: ComputerToolResult):
        content: list[dict[str, Any]] = []

        if item.screenshot is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(item.screenshot).decode("ascii"),
                },
            })

        if item.output or not content:
            content.append({"type": "text", "text": item.output or "OK"})

        block = self._tool_result_block(item.call_id, content, item.is_error, item.cache, item.provider_options)
        # every computer action result must name the toolset of its tool_use block
        block["toolset_name"] = COMPUTER_TOOLSET_NAME

        return {"role": "user", "content": [block]}

    def _shell_output_text(self, item: ShellToolResult) -> str:
        if not item.output:
            return "tool has been restarted."

        parts = []
        for output in item.output:
            # bash interleaves stderr with stdout; the separate streams are sent one after the other
            text = "\n".join(stream for stream in (output.stdout, output.stderr) if stream)
            if output.timed_out:
                text = f"{text}\nCommand timed out." if text else "Command timed out."
            elif output.exit_code != 0:
                text = f"{text}\nExit code: {output.exit_code}" if text else f"Exit code: {output.exit_code}"
            parts.append(text or "(no output)")

        return "\n".join(parts)

    def map_shell_tool_result(self, item: ShellToolResult):
        block = self._tool_result_block(
            item.call_id, self._shell_output_text(item), item.is_error, item.cache, item.provider_options,
        )
        return {"role": "user", "content": [block]}

    def map_provider_tool_result(self, item: ProviderToolResult):
        native = super().map_provider_tool_result(item)
        # a content block such as a tool_result goes in a user turn; a full message is sent as is
        return native if "role" in native else {"role": "user", "content": [native]}

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
            canonical_params["tools"] = self.map_tools(request.tools)

        reasoning_params = self.map_reasoning_params(request.reasoning, max_tokens=max_tokens)

        if "thinking" in reasoning_params:
            canonical_params["thinking"] = reasoning_params["thinking"]

        if "output_config" in reasoning_params:
            # keep other native output_config fields, such as format; the typed effort wins
            canonical_params["output_config"] = {
                **(provider_options.get("output_config") or {}),
                **reasoning_params["output_config"],
            }

        cache = request.cache

        if cache is not None:
            canonical_params.update(self.map_cache_params(cache))
            # typed cache options decide top-level caching, also when mode "explicit" turns it off
            provider_options.pop("cache_control", None)

            if cache.retention is not None:
                ttl = "1h" if cache.retention == "long" else "5m"
                for key in ("system", "tools"):
                    if key in canonical_params:
                        canonical_params[key] = self._apply_ttl(canonical_params[key], ttl)
                for message in messages:
                    message["content"] = self._apply_ttl(message["content"], ttl)

        extra_body = provider_options.get("extra_body")

        if extra_body is not None:
            provider_options["extra_body"] = {
                key: value
                for key, value in extra_body.items()
                if key not in canonical_params and not (cache is not None and key == "cache_control")
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

    def _normalize_output_item(self, content_block, shell_tool: bool = False) -> OutputType:
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

            case "tool_use" if getattr(content_block, "toolset_name", None) == COMPUTER_TOOLSET_NAME:
                return ComputerCallOutputItem(
                    raw=content_block,
                    provider="anthropic",
                    call_id=content_block.id,
                    actions=[self._normalize_computer_action(content_block.name, dict(content_block.input))],
                )

            case "tool_use" if shell_tool and content_block.name == "bash":
                command = content_block.input.get("command")
                return ShellCallOutputItem(
                    raw=content_block,
                    provider="anthropic",
                    call_id=content_block.id,
                    commands=[command] if isinstance(command, str) else [],
                    restart=bool(content_block.input.get("restart")),
                )

            case "tool_use":
                return FunctionCallOutputItem(
                    raw=content_block,
                    call_id=content_block.id,
                    name=content_block.name,
                    arguments=dict(content_block.input)
                )

            case "server_tool_use" if content_block.name == "web_search":
                return self._web_search_item(content_block, None)

            case "web_search_tool_result":
                return self._web_search_item(None, content_block)

            case _:
                # redacted_thinking, other server tool blocks and anything newer pass through unchanged
                return ProviderOutputItem(
                    provider_type=content_block.type,
                    raw=content_block
                )

    def _keys(self, text) -> list[str]:
        # Anthropic joins keys with "+", e.g. "ctrl+shift"; a lone "+" is the plus key
        if not text:
            return []
        return [text] if text == "+" else text.split("+")

    def _normalize_computer_action(self, member: str, payload: dict[str, Any]) -> actions.ComputerAction:
        coordinate: list[Any] = payload.get("coordinate") or [None, None]
        modifiers = self._keys(payload.get("text"))

        try:
            match member:
                case _ if member in _CLICK_MEMBERS:
                    button, clicks = _CLICK_MEMBERS[member]
                    return actions.Click(
                        x=coordinate[0], y=coordinate[1], button=button, clicks=clicks, modifiers=modifiers,
                    )
                case "left_click_drag":
                    start = payload["start_coordinate"]
                    return actions.Drag(
                        path=[actions.Point(x=start[0], y=start[1]), actions.Point(x=coordinate[0], y=coordinate[1])],
                        modifiers=modifiers,
                    )
                case "mouse_move":
                    return actions.MoveMouse(x=coordinate[0], y=coordinate[1])
                case "left_mouse_down":
                    return actions.MouseDown()
                case "left_mouse_up":
                    return actions.MouseUp()
                case "scroll":
                    step_x, step_y = _SCROLL_DIRECTIONS[payload["scroll_direction"]]
                    amount = payload["scroll_amount"]
                    return actions.Scroll(
                        x=coordinate[0], y=coordinate[1], dx=step_x * amount, dy=step_y * amount,
                        unit="clicks", modifiers=modifiers,
                    )
                case "type":
                    return actions.TypeText(text=payload["text"])
                case "key":
                    return actions.KeyPress(keys=modifiers, repeat=payload.get("repeat", 1))
                case "hold_key":
                    return actions.HoldKey(keys=modifiers, duration=payload["duration"])
                case "wait":
                    return actions.Wait(duration=payload["duration"])
                case "screenshot":
                    return actions.Screenshot()
                case "zoom":
                    return actions.Zoom(region=tuple(payload["region"]))
                case "cursor_position":
                    return actions.CursorPosition()
        except (KeyError, IndexError, TypeError, ValidationError):
            pass

        # a member added in a newer toolset version, or one whose input changed, keeps its native payload
        return actions.ProviderComputerAction(provider_type=member, payload=payload)

    def _web_search_item(self, call_block, result_block) -> WebSearchCallOutputItem:
        '''Builds one web search call from its server_tool_use block, its result block, or both.

            A call without a result is still running, e.g. when the turn paused (stop_reason pause_turn).
        '''
        if call_block is not None:
            call_id = call_block.id
            query = (call_block.input or {}).get("query")
        elif result_block is not None:
            call_id = result_block.tool_use_id
            query = None
        else:
            raise ValueError("a web search call needs its server_tool_use block, its result block, or both")

        status = "in_progress"
        sources: list[WebSource] = []
        error = None

        if result_block is not None:
            content = result_block.content
            if isinstance(content, list):
                status = "completed"
                sources = [
                    WebSource(url=result.url, title=result.title, page_age=result.page_age)
                    for result in content
                ]
            else:
                status = "failed"
                error = content.error_code

        return WebSearchCallOutputItem(
            raw=[block for block in (call_block, result_block) if block is not None],
            provider="anthropic",
            call_id=call_id,
            status=status,
            queries=[query] if isinstance(query, str) else [],
            sources=sources,
            error=error,
        )

    def _text_parts(self, message: AssistantMessage) -> list[TextOutputItem]:
        return message.content if isinstance(message.content, list) else [message.content]

    def _normalize_output(self, output_items, shell_tool: bool = False) -> list[OutputType]:
        output: list[OutputType] = []
        # web search calls by id, so a result that isn't next to its call still gets the queries
        web_search_calls: dict[str, WebSearchCallOutputItem] = {}

        for block in output_items:
            item = self._normalize_output_item(block, shell_tool)
            previous = output[-1] if output else None

            if isinstance(item, WebSearchCallOutputItem) and block.type == "server_tool_use":
                web_search_calls[item.call_id] = item

            # a web search result directly after its server_tool_use block completes that call.
            # Only neighbouring blocks are merged, so replaying the output keeps the native block order.
            if (
                block.type == "web_search_tool_result"
                and isinstance(previous, WebSearchCallOutputItem)
                and previous.call_id == block.tool_use_id
                and previous.status == "in_progress"
            ):
                output[-1] = self._web_search_item(previous.raw[0], block)

            # parallel searches interleave calls and results; each stays its own item, in native order
            elif isinstance(item, WebSearchCallOutputItem) and item.call_id in web_search_calls:
                item.queries = web_search_calls[item.call_id].queries
                output.append(item)

            # Anthropic splits text at every citation; neighbouring text blocks are one message
            elif isinstance(item, AssistantMessage) and output and isinstance(output[-1], AssistantMessage):
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

        server_tool_use = getattr(provider_usage, "server_tool_use", None)
        server_tool_requests = {
            name: getattr(server_tool_use, usage_field, None) or 0
            for name, usage_field in _SERVER_TOOL_USAGE_FIELDS.items()
        } if server_tool_use is not None else {}
        server_tool_requests = {name: count for name, count in server_tool_requests.items() if count}

        return Usage(
            input_tokens=input_tokens,
            output_tokens=provider_usage.output_tokens,
            total_tokens=input_tokens + provider_usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            reasoning_tokens=output_details.thinking_tokens if output_details is not None else None,
            server_tool_requests=server_tool_requests or None,
        )

    def _normalize_stop_reason(self, reason) -> StopReason | str | None:
        if reason is None:
            return None
        try:
            return StopReason(reason)
        except ValueError:
            return reason

    def _normalize_status(
        self, stop_reason: StopReason | str | None
    ) -> tuple[ResponseStatus | None, IncompleteReason | None]:
        '''Maps a stop reason to the canonical status and, for incomplete responses, the reason.

            Anthropic has no response status; stop_reason says how the turn ended. A stop reason
            Tessaract doesn't know yet gives no status.
        '''
        match stop_reason:
            case StopReason.END_TURN | StopReason.STOP_SEQUENCE | StopReason.TOOL_USE:
                return ResponseStatus.COMPLETED, None
            case StopReason.MAX_TOKENS:
                return ResponseStatus.INCOMPLETE, IncompleteReason.MAX_TOKENS
            case StopReason.MODEL_CONTEXT_WINDOW_EXCEEDED:
                return ResponseStatus.INCOMPLETE, IncompleteReason.CONTEXT_WINDOW
            case StopReason.REFUSAL:
                return ResponseStatus.INCOMPLETE, IncompleteReason.CONTENT_FILTER
            # the turn isn't finished: send the output back to let the model continue
            case StopReason.PAUSE_TURN | StopReason.COMPACTION:
                return ResponseStatus.INCOMPLETE, IncompleteReason.PAUSED
            case _:
                return None, None

    def _normalize_response(self, raw_response, shell_tool: bool = False) -> AnthropicResponse:
        stop_reason = self._normalize_stop_reason(raw_response.stop_reason)
        status, incomplete_reason = self._normalize_status(stop_reason)

        return AnthropicResponse(
            id=raw_response.id,
            model=raw_response.model,
            status=status,
            incomplete_reason=incomplete_reason,
            stop_reason=stop_reason,
            output=self._normalize_output(raw_response.content, shell_tool),
            usage=self._normalize_usage(raw_response.usage),
            provider_usage=raw_response.usage,
            raw_response=raw_response
        )

    def _normalize_stream_event(self, event, state: _StreamState | None = None) -> Iterator[StreamEventUnion]:
        if state is None:
            state = _StreamState()

        match event.type:
            case "message_start":
                yield ResponseStartedEvent(raw_event=event)

            case "content_block_start" if event.content_block.type == "tool_use":
                call = self._normalize_output_item(event.content_block, state.shell_tool)
                if isinstance(call, ComputerCallOutputItem | ShellCallOutputItem):
                    name, call_type = ("computer" if call.type == "computer_call" else "shell"), call.type
                else:
                    name, call_type = event.content_block.name, "function_call"
                yield ToolCallStartedEvent(
                    call_id=event.content_block.id,
                    name=name,
                    call_type=call_type,
                    output_index=event.index,
                    raw_event=event
                )

            case "content_block_start" if event.content_block.type == "server_tool_use":
                state.server_tool_indexes.add(event.index)
                yield ServerToolCallStartedEvent(
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

            # server tool input isn't function arguments; it stays a CustomProviderEvent
            case "content_block_delta" if event.delta.type == "input_json_delta" and event.index not in state.server_tool_indexes:
                yield FunctionCallArgumentDeltaEvent(
                    delta=event.delta.partial_json,
                    output_index=event.index,
                    raw_event=event
                )

            # a web search call is done when its result arrives; output_item.done then carries both blocks
            case "content_block_stop" if (
                getattr(event, "content_block", None) is not None
                and event.content_block.type == "server_tool_use"
                and event.content_block.name == "web_search"
            ):
                state.web_search_calls[event.content_block.id] = event.content_block

            case "content_block_stop" if (
                getattr(event, "content_block", None) is not None
                and event.content_block.type == "web_search_tool_result"
            ):
                yield OutputItemCompletedEvent(
                    item=self._web_search_item(
                        state.web_search_calls.pop(event.content_block.tool_use_id, None),
                        event.content_block,
                    ),
                    output_index=event.index,
                    raw_event=event
                )

            # MessageStream adds the finished block and message to these events
            case "content_block_stop" if getattr(event, "content_block", None) is not None:
                yield OutputItemCompletedEvent(
                    item=self._normalize_output_item(event.content_block, state.shell_tool),
                    output_index=event.index,
                    raw_event=event
                )

            case "message_stop" if getattr(event, "message", None) is not None:
                yield ResponseCompletedEvent(
                    response=self._normalize_response(event.message, state.shell_tool),
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

    def _declares_shell_tool(self, request: Request) -> bool:
        return any(isinstance(tool, ShellTool) for tool in request.tools or [])

    def generate_sync(self, request: Request) -> AnthropicResponse:
        messages_api, kwargs = self._messages_api(self._build_request_kwargs(request))

        _raw_response = messages_api.create(**kwargs, stream=False)

        return self._normalize_response(_raw_response, self._declares_shell_tool(request))

    def generate_stream(self, request: Request) -> Iterator[StreamEventUnion]:
        from anthropic import APIStatusError

        messages_api, kwargs = self._messages_api(self._build_request_kwargs(request))

        state = _StreamState(shell_tool=self._declares_shell_tool(request))

        with messages_api.stream(**kwargs) as stream:
            try:
                for raw_event in stream:
                    yield from self._normalize_stream_event(raw_event, state)
            except APIStatusError as exc:
                # error events sent mid-stream; errors before the stream starts are raised
                error = self._response_error(exc)
                yield ResponseFailedEvent(
                    message=error.message,
                    error=error,
                    raw_event=exc
                )

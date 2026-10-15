from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Protocol

from ..types.output_types import OutputItem, TextOutputItem
from ..providers.provider import Provider
from ..tools.function import InputSchema

if TYPE_CHECKING:
    from ..types.request import Request
    from ..types.response import Response
    from ..types.streaming.event_types import StreamEventUnion


class UserMessageProtocol(Protocol):
    role: Literal["user"] = "user"
    content: str | list[dict]

class AssistantMessageProtocol(Protocol):
    type: Literal["assistant_message"] = "assistant_message"
    role: Literal["assistant"] = "assistant"
    content: TextOutputItem | list[TextOutputItem]
    raw: Any

class SystemPromptProtocol(Protocol):
    role: Literal["system"] = "system"
    content: str | list[dict]

class FunctionToolResultProtocol(Protocol):
    type: Literal["function_tool_result"] = "function_tool_result"
    call_id: str
    result: Any
    is_error: bool = False
    provider_options: dict[str, Any]

class FunctionToolSchemaProtocol(Protocol):
    name: str
    description: str
    input_schema: InputSchema | None
    strict: bool | None = None
    provider_options: dict[str, Any]

class ReasoningParamsProtocol(Protocol):
    effort: Literal[
        "none", "minimal", "low", "medium", "high", "extra_high", "max"
    ] | None = None
    summary: Literal["omitted", "concise", "auto", "detailed", "updates"] | None = None
    budget: int | None = None
    mode: Literal["standard", "pro", "enabled", "between_tools", "adaptive", "disabled"] | None = None
    native_options: dict | None = None

class CacheParamsProtocol(Protocol):
    retention: Literal["short", "long"] | None = None
    key: str | None = None

class ReasoningProtocol(Protocol):
    type: Literal["reasoning"] = "reasoning"
    text: str | None = None

class Adapter:
    def __init__(self, provider: Provider):
        self._provider = provider

    def map_input_message(self, item: UserMessageProtocol | SystemPromptProtocol) -> Any:
        raise NotImplementedError("not yet implemented...")

    def map_assistant_message(self, item: AssistantMessageProtocol) -> Any:
        raise NotImplementedError("not yet implemented...")

    def map_output_item(self, item: OutputItem) -> Any:
        '''Replays a previous output item (reasoning, function call, provider item) as native input.'''
        return item.raw

    def map_tool_result(self, item: FunctionToolResultProtocol) -> Any:
        raise NotImplementedError("not yet implemented...")

    def map_function_schema(self, tools: list[FunctionToolSchemaProtocol]) -> list:
        raise NotImplementedError("not yet implemented...")

    def map_reasoning_params(self, reasoning: ReasoningParamsProtocol | None) -> Any:
        raise NotImplementedError("not yet implemented...")
    
    def map_cache_params(self, cache: CacheParamsProtocol | None) -> dict[str, Any]:
        raise NotImplementedError("not yet implemented...")

    def map_reasoning(self, item: ReasoningProtocol) -> Any:
        raise NotImplementedError("not yet implemented")

    def generate_sync(self, request: "Request") -> "Response":
        raise NotImplementedError("not yet implemented")

    def generate_stream(self, request: "Request") -> Iterator["StreamEventUnion"]:
        raise NotImplementedError("not yet implemented")

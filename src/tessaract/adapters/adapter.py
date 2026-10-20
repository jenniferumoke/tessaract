from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Protocol

from ..types.output_types import OutputItem, TextOutputItem
from ..providers.provider import Provider
from ..tools.function import FunctionTool, InputSchema
from ..tools.provider import ProviderTool
from ..tools.web_search import UserLocation, WebSearchTool

if TYPE_CHECKING:
    from ..types.request import Request
    from ..types.response import Response
    from ..types.streaming.event_types import StreamEventUnion


class UserMessageProtocol(Protocol):
    role: Literal["user"] = "user"
    content: str | list[dict]
    cache: bool = False

class AssistantMessageProtocol(Protocol):
    type: Literal["assistant_message"] = "assistant_message"
    role: Literal["assistant"] = "assistant"
    content: TextOutputItem | list[TextOutputItem]
    raw: Any

class SystemPromptProtocol(Protocol):
    role: Literal["system"] = "system"
    content: str | list[dict]
    cache: bool = False

class FunctionToolResultProtocol(Protocol):
    type: Literal["function_tool_result"] = "function_tool_result"
    call_id: str
    result: Any
    is_error: bool = False
    cache: bool = False
    provider_options: dict[str, Any]

class FunctionToolSchemaProtocol(Protocol):
    name: str
    description: str
    input_schema: InputSchema | None
    strict: bool | None = None
    cache: bool = False
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
    mode: Literal["implicit", "explicit"] | None = None
    retention: Literal["short", "long"] | None = None
    key: str | None = None

class ReasoningProtocol(Protocol):
    type: Literal["reasoning"] = "reasoning"
    text: str | None = None

class Adapter:
    # the provider a ProviderTool must be written for to be sent through this adapter
    provider_name: str

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

    def map_tools(self, tools: Sequence[FunctionTool | WebSearchTool | ProviderTool]) -> list:
        '''Maps canonical tool definitions to native ones, keeping their order.'''
        native = []

        for tool in tools:
            if isinstance(tool, FunctionTool):
                native.extend(self.map_function_schema([tool]))
            elif isinstance(tool, WebSearchTool):
                native.append(self.map_web_search_tool(tool))
            elif isinstance(tool, ProviderTool):
                native.append(self.map_provider_tool(tool))
            else:
                raise TypeError(f"Unsupported tool type: {type(tool).__name__}")

        return native

    def map_web_search_tool(self, tool: WebSearchTool) -> dict[str, Any]:
        raise NotImplementedError("not yet implemented...")

    def map_provider_tool(self, tool: ProviderTool) -> dict[str, Any]:
        if tool.provider != self.provider_name:
            raise ValueError(
                f"This ProviderTool was written for {tool.provider!r} and cannot be sent to "
                f"{self.provider_name!r}. Use a canonical tool, or a ProviderTool for {self.provider_name!r}."
            )
        return dict(tool.definition)

    def _native_user_location(self, location: UserLocation) -> dict[str, Any]:
        # both providers take the same approximate location object
        return {"type": "approximate", **location.model_dump(exclude_none=True)}

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

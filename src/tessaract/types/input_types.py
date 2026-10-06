
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel


if TYPE_CHECKING:
    from ..adapters.adapter import Adapter

class InputType(BaseModel):
    def raw(self, adapter: Adapter) -> Any:
        ...

class UserMessage(InputType):
    role: Literal["user"] = "user"
    content: str | list[dict]

    def raw(self, adapter: Adapter):
        return adapter.map_input_message(self)

class SystemPrompt(InputType):
    role: Literal["system"] = "system"
    content: str | list[dict]

    def raw(self, adapter: Adapter):
        return adapter.map_input_message(self)

class FunctionToolResult(InputType):
    type: Literal["function_tool_result"] = "function_tool_result"
    call_id: str
    result: Any
    is_error: bool = False

    def raw(self, adapter: Adapter):
        return adapter.map_tool_result(self)

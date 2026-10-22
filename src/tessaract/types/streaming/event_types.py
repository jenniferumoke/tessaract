from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from ..output_types import OutputType
from ..response import Response, ResponseError


# change these BaseModel classes to pure python dataclasses
class TextDeltaEvent(BaseModel):
    type: Literal["text.delta"] = "text.delta"

    delta: str

    provider: Literal["openai", "anthropic"]
    response_id: str | None = None

    output_index: int | None = None
    content_index: int | None = None
    item_id: str | None = None

    raw_event: object | None = None

class ResponseStartedEvent(BaseModel):
    type: Literal["response_started"] = "response_started"
    message: Literal["starting response..."] = "starting response..."

    raw_event: Any = Field(
        exclude=True,
        repr=False
    )

class ResponseCompletedEvent(BaseModel):
    type: Literal["response.completed"] = "response.completed"
    response: Response
    message: Literal["... finished response"] = "... finished response"

    raw_event: Any = Field(
        exclude=True,
        repr=False
    )

class ReasoningSummaryDeltaEvent(BaseModel):
    type: Literal["reasoning_summary.delta"] = "reasoning_summary.delta"
    item_id: str | None = None
    delta: str
    output_index: int
    content_index: int | None = None

    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False
    )

class ReasoningStartedEvent(BaseModel):
    type: Literal["reasoning.started"] = "reasoning.started"
    item_id: str | None = None
    output_index: int | None = None
    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False
    )



class ReasoningTextDeltaEvent(BaseModel):
    type: Literal["reasoning_text.delta"] = "reasoning_text.delta"
    item_id: str | None = None
    delta: str
    output_index: int
    content_index: int | None = None

    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False
    )

class FunctionCallArgumentDeltaEvent(BaseModel):
    type: Literal["tool_arguments.delta"] = "tool_arguments.delta"
    item_id: str | None = None
    delta: str
    output_index: int
    content_index: int | None = None
    
    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False
    )

class ToolCallStartedEvent(BaseModel):
    '''The model started a call to a tool your code runs.

        call_type is the type of the output item the call becomes. name is the function name for
        function calls, and "computer" or "shell" for those tools.
    '''
    type: Literal["tool_call.started"] = "tool_call.started"

    call_id: str
    name: str
    output_index: int
    call_type: Literal["function_call", "computer_call", "shell_call"] = "function_call"

    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )

class ServerToolCallStartedEvent(BaseModel):
    '''The model started a call to a tool the provider runs, such as web search.

        name is the tool name: "web_search", "shell" for a command in a provider container, or the native
        name of another server tool. The finished call arrives as an
        output_item.done event. You don't send a result back for server tool calls.
    '''
    type: Literal["server_tool_call.started"] = "server_tool_call.started"

    call_id: str
    name: str
    output_index: int

    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )

class ResponseFailedEvent(BaseModel):
    type: Literal["response.failed"] = "response.failed"
    message: str
    error: ResponseError
    response: Response | None = None


    raw_event: Any | None = Field(
        default=None,
        exclude=True,
        repr=False
    )

class OutputItemCompletedEvent(BaseModel):
    type: Literal["output_item.done"] = "output_item.done"
    output_index: int | None = None
    item: OutputType
    raw_event: Any = Field(
        default=None,
        exclude=True,
        repr=False
    )

class CustomProviderEvent(BaseModel):
    '''A native stream event without a canonical equivalent.

        type is always "custom_provider_event", so `event.type == ...` checks narrow the
        StreamEventUnion for type checkers. The native event name is in provider_type.
    '''
    type: Literal["custom_provider_event"] = "custom_provider_event"
    provider_type: str
    raw_event: Any

StreamEventUnion: TypeAlias = (
    CustomProviderEvent | 
    ReasoningStartedEvent |
    ResponseFailedEvent | 
    ToolCallStartedEvent | 
    ServerToolCallStartedEvent |
    FunctionCallArgumentDeltaEvent | 
    ReasoningTextDeltaEvent | 
    ReasoningSummaryDeltaEvent |
    ResponseCompletedEvent | 
    ResponseStartedEvent | 
    TextDeltaEvent |
    OutputItemCompletedEvent
    )


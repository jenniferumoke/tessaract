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
    type: Literal["tool_call.started"] = "tool_call.started"

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
    _type: Literal["custom_provider_event"] = "custom_provider_event"
    type: str
    raw_event: Any

StreamEventUnion: TypeAlias = (
    CustomProviderEvent | 
    ReasoningStartedEvent |
    ResponseFailedEvent | 
    ToolCallStartedEvent | 
    FunctionCallArgumentDeltaEvent | 
    ReasoningTextDeltaEvent | 
    ReasoningSummaryDeltaEvent |
    ResponseCompletedEvent | 
    ResponseStartedEvent | 
    TextDeltaEvent
    )


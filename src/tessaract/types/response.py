from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ..providers.provider import Provider
from .output_types import (
    AssistantMessage,
    OutputType,
    TextOutputItem,
)

if TYPE_CHECKING:
    from openai.types.responses.response_usage import ResponseUsage


@dataclass
class ResponseError:
    message: str
    code: str | None = None

class ResponseStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    COMPACTION =  "compaction"
    REFUSAL = "refusal"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    total_tokens: int | None = None

    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class Response:
    id: str
    model: str
    provider: Literal["anthropic", "openai"] | None = None

    output: list[OutputType] = field(
        default_factory=list,
    )

    usage: Usage | None = None
    error: ResponseError | None = None

    # finish_details: FinishDetails | None = None

    # provider_timestamps: ProviderTimestamps | None = None
    # telemetry: ResponseTelemetry | None = None

    # provider_metadata: dict[str, Any] = Field(
    #     default_factory=dict,
    # )

    raw_response: Any | None = field(
        default=None,
        repr=False,
    )

    @property
    def output_text(self) -> str:
        return "".join(
            block.text
            for item in self.output
            if isinstance(item, AssistantMessage)
            for block in item.content
            if isinstance(block, TextOutputItem)
        )

    @property
    def response_id(self) -> str:
        return self.id


@dataclass
class OpenAIResponse(Response):
    provider: Literal["openai"] = "openai"
    provider_usage: ResponseUsage | None = field(default=None, repr=False)
    status: ResponseStatus = field(kw_only=True)

    @property
    def input_tokens(self) -> int | None:
        return self.usage.input_tokens if self.usage is not None else None

    @property
    def cached_input_tokens(self) -> int | None:
        return self.usage.cached_input_tokens if self.usage is not None else None

@dataclass
class AnthropicResponse(Response):
    provider: Literal["anthropic"] = "anthropic"
    stop_reason: StopReason = field(kw_only=True)

    provider_usage: ResponseUsage | None = field(default=None, repr=False)
    
    @property
    def input_tokens(self) -> int | None:
        return self.usage.input_tokens if self.usage is not None else None

    @property
    def cached_input_tokens(self) -> int | None:
        return self.usage.cached_input_tokens if self.usage is not None else None


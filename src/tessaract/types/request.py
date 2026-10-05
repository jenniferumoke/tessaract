from dataclasses import dataclass
from typing import Any, Literal

from ..tools.function import FunctionTool


@dataclass
class ReasoningOptions:
    effort: Literal[
        "none", "minimal", "low", "medium", "high", "extra_high", "max"
    ] | None = None
    summary: Literal["concise", "auto", "detailed"] | None = None
    mode: Literal["standard", "pro"] | None = None

@dataclass
class Request:
    '''Request model for generating a response.
        Args:
            model: str
            instructions: str
            input: list[role | content]
            tools: list[FunctionTool] | None = None
            reasoning:  ReasoningOptions | None = None
            stream: bool = False
            provider_options: dict[str, Any] | None = None
    '''
    model: str
    input: list
    instructions: str | None = None
    tools: list[FunctionTool] | None = None
    reasoning:  ReasoningOptions | None = None
    stream: bool = False
    provider_options: dict[str, Any] | None = None
    max_tokens: int | None = None
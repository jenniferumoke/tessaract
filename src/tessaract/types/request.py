from dataclasses import dataclass
from typing import Any, Literal

from ..tools.function import FunctionTool

'''
    thinking={
                "type": "enabled", 
                "budget_tokens": 1024, # must not be >= max_tokens and must be >= 1024
                "display": "summarized" or "omitted" or "updates",
                "block_binding": {
                    "prefix_match_behaviour": "drop_block" or "error"
                }
            }
    thinking={
            "type": "disabled", 
        }
    thinking={
        "type": "between_tools"
    }
    thinking={
            "type": "adaptive", 
            "display": "summarized" or "omitted" or "updates",
            the next config params are from output_config and must be converted to the output_config, for recent reasoning models that do not allow thinking enabled
            "effort": ["low", "medium", "high", "xhigh", "max"]
        }
'''

@dataclass
class ReasoningOptions:
    effort: Literal[
        "none", "minimal", "low", "medium", "high", "extra_high", "max" # only low medium high, extra_high converted to xhigh and max should be sent to anthropic, and once sent to anthropic should be converted to output_config. if output_config as provider options in request, let typed param win.
    ] | None = None
    summary: Literal["omitted", "concise", "auto", "detailed", "updates"] | None = None #  concise should map to "summarized" in anthropic, "omitted" and "updates" should only be sent to anthropic, in thinking_enabled and thinking_adaptive
    budget: int | None = None # this should only be sent to anthropic when thinking enabled
    mode: Literal["standard", "pro", "enabled", "between_tools", "adaptive", "disabled"] | None = None # can seperate between openai and anthropic modes in an enum


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
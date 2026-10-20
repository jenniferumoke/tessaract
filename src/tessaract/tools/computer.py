from typing import Any, Literal

from pydantic import BaseModel, Field


class ComputerUseTool(BaseModel):
    '''Lets the model control a computer: your code runs each action and returns a screenshot.

        Maps to OpenAI's computer tool and Anthropic's computer_toolset_20260801. Calls come back as
        ComputerCallOutputItems; answer each one with a ComputerToolResult.

        cache: add a cache breakpoint to the tool definition. Anthropic only; OpenAI caches automatically.
        provider_options: native tool fields, e.g. {"configs": {"zoom": {"enabled": False}}} for Anthropic.

        Anthropic's earlier computer_20251124 tool has a different call shape; use a ProviderTool for it.
    '''
    type: Literal["computer"] = "computer"
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

from typing import Any, Literal

from pydantic import BaseModel, Field


class ShellTool(BaseModel):
    '''Lets the model run shell commands in an environment your code controls.

        Maps to OpenAI's shell tool with a local environment and Anthropic's bash_20250124 tool.
        Calls come back as ShellCallOutputItems; answer each one with a ShellToolResult.

        cache: add a cache breakpoint to the tool definition. Anthropic only; OpenAI caches automatically.
        provider_options: native tool fields. On OpenAI, "environment" replaces the local environment,
            e.g. to add skills.
    '''
    type: Literal["shell"] = "shell"
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

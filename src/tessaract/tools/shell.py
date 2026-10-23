from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ContainerEnvironment(BaseModel):
    '''A container the provider manages and runs shell commands in: OpenAI's hosted shell, or Anthropic's
        code execution tool.

        container_id: reuse an existing container, e.g. Response.container_id. The other fields only apply to a new container.
        file_ids: uploaded files to copy into a new container
        memory_limit: the new container's memory. OpenAI only; Anthropic containers have a fixed size.
        allowed_domains: the only domains the container can reach. OpenAI only; Anthropic containers have no network access.
        network_disabled: block all network access, which is how Anthropic containers always run
        provider_options: native fields, e.g. {"skills": [...]}. Merged into OpenAI's environment, and sent as
            Anthropic's top-level container object.

        An option the target provider doesn't support raises UnsupportedToolOptionError.
    '''
    type: Literal["container"] = "container"
    container_id: str | None = None
    file_ids: list[str] | None = None
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = None
    allowed_domains: list[str] | None = None
    network_disabled: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _model_validator(self):
        if self.allowed_domains is not None and self.network_disabled:
            raise ValueError("allowed_domains and network_disabled cannot be used together")

        new_container_fields = (self.file_ids, self.memory_limit, self.allowed_domains)
        if self.container_id is not None and (any(value is not None for value in new_container_fields) or self.network_disabled):
            raise ValueError(
                "container_id reuses an existing container, so file_ids, memory_limit, allowed_domains and "
                "network_disabled cannot be set with it"
            )

        return self


class ShellTool(BaseModel):
    '''Lets the model run shell commands.

        environment: "local" runs commands in your environment: calls come back as ShellCallOutputItems
            with executed_by="client", and you answer each one with a ShellToolResult. A ContainerEnvironment
            runs them in a container the provider manages: calls come back with
            executed_by="provider" and their output, and need no result.

        Maps to OpenAI's shell tool, and to Anthropic's bash_20250124 tool for a local environment or its
        code_execution_20250825 tool for a container.

        cache: add a cache breakpoint to the tool definition. Anthropic only; OpenAI caches automatically.
        provider_options: native tool fields. On OpenAI, the fields of "environment" are merged into
            the environment, e.g. to add skills to a local environment.
    '''
    type: Literal["shell"] = "shell"
    environment: Literal["local"] | ContainerEnvironment = "local"
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

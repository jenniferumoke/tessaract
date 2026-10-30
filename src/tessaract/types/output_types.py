from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from .computer_actions import ComputerAction

class OutputItem(BaseModel):
    raw: Any

class Annotation(BaseModel):
    type: Literal["citation", "file_path"] = "citation"
    provider_type: str | None = None

    source: str | None = None
    title: str | None = None
    cited_text: str | None = None

    provider: Literal["anthropic", "openai", "openrouter"]

    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

class TextOutputItem(OutputItem):
    type: Literal["text"] = "text"
    text: str
    annotations: list[Annotation] = Field(
        default_factory=list,
    )

class AssistantMessage(BaseModel):
    type: Literal["assistant_message"] = "assistant_message"
    role: Literal["assistant"] = "assistant"
    content: TextOutputItem | list[TextOutputItem]
    raw: Any

class ReasoningOutputItem(OutputItem):
    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    text: str | list | None = None
    content: str | list | None = None

class FunctionCallOutputItem(OutputItem):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class WebSource(BaseModel):
    url: str
    title: str | None = None
    # how old the page is, as the provider reports it (Anthropic only)
    page_age: str | None = None

class WebSearchCallOutputItem(OutputItem):
    '''A web search the provider ran.

        action: "search", or "open_page" / "find_in_page" when OpenAI browses a result
        queries: the search queries; empty for open_page and find_in_page
        url / pattern: the page opened or searched, for open_page and find_in_page
        sources: the pages found. OpenAI returns urls only.
        error: the provider's error code when status is "failed"

        On Anthropic, one call is a server_tool_use block followed by a web_search_tool_result block;
        raw is the list of both, and both are replayed unchanged. On OpenAI, raw is the web_search_call item.
    '''
    type: Literal["web_search_call"] = "web_search_call"
    call_id: str
    status: Literal["in_progress", "searching", "completed", "failed"]
    action: str = "search"
    queries: list[str] = Field(default_factory=list)
    url: str | None = None
    pattern: str | None = None
    sources: list[WebSource] = Field(default_factory=list)
    error: str | None = None
    provider: Literal["anthropic", "openai"]

class SafetyCheck(BaseModel):
    '''A check the provider wants you to confirm before running a computer call (OpenAI only).

        To go ahead, pass the checks back in ComputerToolResult.acknowledged_safety_checks.
    '''
    id: str
    code: str | None = None
    message: str | None = None

class ComputerCallOutputItem(OutputItem):
    '''Computer actions for your code to run, in order. Answer with a ComputerToolResult.

        OpenAI can batch several actions into one call. Anthropic sends one action per call, and
        often several calls per turn; every call needs its own result.
    '''
    type: Literal["computer_call"] = "computer_call"
    call_id: str
    actions: list[ComputerAction]
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    safety_checks: list[SafetyCheck] = Field(default_factory=list)
    provider: Literal["anthropic", "openai"]

class ShellCommandOutput(BaseModel):
    '''What one shell command printed. exit_code is ignored when timed_out is True.'''
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

class ShellCallOutputItem(OutputItem):
    '''Shell commands, in order.

        executed_by: "client" when your code runs them (a local ShellTool): answer with a ShellToolResult.
            "provider" when the provider ran them in its container (OpenAI): output holds what they printed,
            and you send nothing back.
        commands: OpenAI can send several; Anthropic sends one
        output: one ShellCommandOutput per command, for provider calls. Empty while the call is still running.
        file_ids: files the commands left in the container's output directory (Anthropic), for the Files API
        error: the provider's error code when a provider call failed (Anthropic), e.g. "execution_time_exceeded"
        timeout_ms / max_output_length: limits OpenAI asks you to apply
        restart: Anthropic asks you to restart the shell session; commands is empty

        A provider call's raw is the list of its native call and output: OpenAI's shell_call and shell_call_output
        items, or Anthropic's server_tool_use and bash_code_execution_tool_result blocks. Both are replayed unchanged.
    '''
    type: Literal["shell_call"] = "shell_call"
    call_id: str
    executed_by: Literal["client", "provider"] = "client"
    commands: list[str] = Field(default_factory=list)
    output: list[ShellCommandOutput] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    timeout_ms: int | None = None
    max_output_length: int | None = None
    restart: bool = False
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    provider: Literal["anthropic", "openai"]

class ProviderOutputItem(OutputItem):
    type: Literal["provider_output"] = "provider_output"
    provider_type: str

OutputType: TypeAlias = (
    AssistantMessage | 
    ReasoningOutputItem | 
    FunctionCallOutputItem | 
    WebSearchCallOutputItem |
    ComputerCallOutputItem |
    ShellCallOutputItem |
    TextOutputItem | 
    ProviderOutputItem
    )


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .output_types import SafetyCheck

if TYPE_CHECKING:
    from ..adapters.adapter import Adapter

class InputType(BaseModel):
    def raw(self, adapter: Adapter) -> Any:
        ...

class UserMessage(InputType):
    role: Literal["user"] = "user"
    content: str | list[dict]
    cache: bool = False

    def raw(self, adapter: Adapter):
        return adapter.map_input_message(self)

class SystemPrompt(InputType):
    role: Literal["system"] = "system"
    content: str | list[dict]
    cache: bool = False

    def raw(self, adapter: Adapter):
        return adapter.map_input_message(self)

class FunctionToolResult(InputType):
    type: Literal["function_tool_result"] = "function_tool_result"
    call_id: str
    result: Any
    is_error: bool = False
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

    def raw(self, adapter: Adapter):
        return adapter.map_tool_result(self)

class ComputerToolResult(InputType):
    '''The result of a ComputerCallOutputItem.

        screenshot: PNG bytes of the screen after the actions ran. OpenAI needs one for every
            computer call. Anthropic needs one for screenshot and zoom actions; for other actions
            output is enough, and "OK" is sent when both are empty.
        output: text for the model, e.g. the cursor position for a cursor_position action (Anthropic)
        is_error: the actions failed; describe what went wrong in output. Anthropic only:
            OpenAI has no error flag, so send the screenshot of the current screen instead.
        acknowledged_safety_checks: the call's safety_checks you confirmed (OpenAI)
    '''
    type: Literal["computer_tool_result"] = "computer_tool_result"
    call_id: str
    screenshot: bytes | None = None
    output: str | None = None
    is_error: bool = False
    acknowledged_safety_checks: list[SafetyCheck] = Field(default_factory=list)
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

    def raw(self, adapter: Adapter):
        return adapter.map_computer_tool_result(self)

class ShellCommandOutput(BaseModel):
    '''What one shell command printed. exit_code is ignored when timed_out is True.'''
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

class ShellToolResult(InputType):
    '''The result of a ShellCallOutputItem: one ShellCommandOutput per command, in order.

        output: empty after a restart (Anthropic); "tool has been restarted." is sent then
        max_output_length: pass the call's max_output_length back (OpenAI)
        is_error: the commands couldn't run at all. Anthropic only; a command that exits with
            a non-zero code isn't an error, its exit code is in the output.
    '''
    type: Literal["shell_tool_result"] = "shell_tool_result"
    call_id: str
    output: list[ShellCommandOutput] = Field(default_factory=list)
    max_output_length: int | None = None
    is_error: bool = False
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)

    def raw(self, adapter: Adapter):
        return adapter.map_shell_tool_result(self)

class ProviderToolResult(InputType):
    '''A native input item, sent unchanged. Use it to answer calls from a ProviderTool that Tessaract
        has no canonical result type for, e.g. OpenAI's apply_patch_call or Anthropic's text editor.

        item: for OpenAI, a native input item such as {"type": "apply_patch_call_output", ...}. For Anthropic,
            a content block such as {"type": "tool_result", ...}, sent in a user turn; a dict with a "role"
            is sent as that message.
        provider is the provider the item was written for. Sending it to another provider raises.
    '''
    type: Literal["provider_tool_result"] = "provider_tool_result"
    provider: Literal["anthropic", "openai"]
    item: dict[str, Any]

    def raw(self, adapter: Adapter):
        return adapter.map_provider_tool_result(self)

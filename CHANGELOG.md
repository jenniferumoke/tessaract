# Changelog

## Unreleased

### Changed: native items that now have canonical types

These items used to come back as `ProviderOutputItem`. They now come back as canonical types, so code that matched them inside `ProviderOutputItem` (`item.type == "provider_output" and item.provider_type == ...`) must match the new `type` instead. Replaying output into the history is unchanged.

| Native item | Before | Now |
| --- | --- | --- |
| OpenAI `web_search_call` | `ProviderOutputItem` | `WebSearchCallOutputItem` (`"web_search_call"`) |
| Anthropic `server_tool_use` (`web_search`) + `web_search_tool_result` | two `ProviderOutputItem`s | one `WebSearchCallOutputItem` |
| OpenAI `computer_call` | `ProviderOutputItem` | `ComputerCallOutputItem` (`"computer_call"`) |
| OpenAI `shell_call` (local environment) | `ProviderOutputItem` | `ShellCallOutputItem` (`"shell_call"`) |
| Anthropic `tool_use` with `toolset_name: "computer"` | `FunctionCallOutputItem` | `ComputerCallOutputItem` |
| Anthropic `bash` `tool_use`, when the request has a `ShellTool` | `FunctionCallOutputItem` | `ShellCallOutputItem` |

Stream events changed the same way:

- OpenAI `response.output_item.added` for `computer_call` and local `shell_call` items is now `tool_call.started` (was `custom_provider_event`), and for `web_search_call`, `file_search_call`, `code_interpreter_call` and `image_generation_call` items it is `server_tool_call.started`.
- Anthropic `content_block_start` for `server_tool_use` is now `server_tool_call.started`, and that block's `input_json_delta` events stay `custom_provider_event`s instead of `tool_arguments.delta`.
- A web search's `server_tool_use` block no longer has its own `output_item.done`; the call's `output_item.done` arrives with its result block.

### Added

- `WebSearchTool` and `UserLocation`: web search on both providers, returned as `WebSearchCallOutputItem`s.
- `ComputerUseTool` and `ComputerToolResult`: computer use on both providers, with one canonical action set in `tessaract.types.computer_actions`.
- `ShellTool`, `ShellToolResult` and `ShellCommandOutput`: a shell your code runs, on both providers.
- `ProviderTool` and `ProviderToolResult`: any native tool definition and native result, sent unchanged.
- `UnsupportedToolOptionError`, raised when a canonical tool option has no equivalent on the target provider.
- `Usage.server_tool_requests`, server tool calls by tool name.
- `ToolCallStartedEvent.call_type`: `"function_call"`, `"computer_call"` or `"shell_call"`.
- `ServerToolCallStartedEvent` (`server_tool_call.started`).
- `send(tools=...)` accepts any sequence of tools, so a `list[FunctionTool]` still type-checks.

### Fixed

- OpenAI: native tools in `request_options={"tools": [...]}` were overwritten by an empty `tools` list when `tools=` wasn't set.
- Anthropic streaming: `input_json_delta` events of server tool calls were reported as `tool_arguments.delta`.

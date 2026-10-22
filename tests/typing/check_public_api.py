"""Type-checked by pyright in CI, never run: common user code must narrow without casts.

Checking `event.type` or `item.type` should give editors (Pylance) the concrete class, so
attributes such as `event.delta` type-check. If this file reports errors, a public type
has lost its literal discriminator or dropped out of its union.
"""

from tessaract import Tessaract


def stream_events(client: Tessaract) -> None:
    for event in client.send(model="oai/m", input="hi", stream=True):
        if event.type == "text.delta":
            print(event.delta, event.provider)
        elif event.type == "reasoning.started":
            print(event.item_id, event.output_index)
        elif event.type == "reasoning_summary.delta":
            print(event.delta, event.output_index)
        elif event.type == "reasoning_text.delta":
            print(event.delta)
        elif event.type == "tool_call.started":
            print(event.call_id, event.name)
        elif event.type == "server_tool_call.started":
            print(event.call_id, event.name, event.output_index)
        elif event.type == "tool_arguments.delta":
            print(event.delta)
        elif event.type == "output_item.done":
            print(event.item.type)
        elif event.type == "response.completed":
            print(event.response.output_text)
        elif event.type == "response.failed":
            print(event.error.message)
        elif event.type == "custom_provider_event":
            print(event.provider_type, event.raw_event)

    for event in client.send(model="oai/m", input="hi", stream=True):
        match event.type:
            case "text.delta":
                print(event.delta)
            case "output_item.done" if event.item.type == "function_call":
                print(event.item.name, event.item.arguments)


def response_output(client: Tessaract) -> None:
    response = client.send(model="oai/m", input="hi", instructions="Be brief.", max_tokens=100)
    print(response.output_text, response.usage.input_tokens if response.usage else None)
    if response.status == "incomplete":
        print(response.incomplete_reason)

    for item in response.output:
        if item.type == "reasoning":
            print(item.text, item.content)
        elif item.type == "function_call":
            print(item.call_id, item.name, item.arguments)
        elif item.type == "assistant_message":
            print(item.content)
        elif item.type == "web_search_call":
            print(item.call_id, item.status, item.queries, [source.url for source in item.sources])
        elif item.type == "provider_output":
            print(item.provider_type)


def send_with_cache(client: Tessaract) -> None:
    from tessaract import CacheOptions

    response = client.send(model="ant/m", input="hi", cache=CacheOptions(retention="long", key="k"))
    print(response.usage)


def send_with_cache_breakpoints(client: Tessaract) -> None:
    from tessaract import CacheOptions, FunctionTool, SystemPrompt

    client.send(
        model="oai/m",
        input=[SystemPrompt(content="docs", cache=True), "hi"],
        tools=[FunctionTool(name="t", description="t", cache=True)],
        cache=CacheOptions(mode="explicit"),
    )


def send_with_server_tools(client: Tessaract) -> None:
    from tessaract import FunctionTool, ProviderTool, UserLocation, WebSearchTool

    response = client.send(
        model="ant/m",
        input="hi",
        tools=[
            WebSearchTool(max_uses=3, user_location=UserLocation(country="NL")),
            FunctionTool(name="t", description="t"),
            ProviderTool(provider="anthropic", definition={"type": "web_fetch_20250910", "name": "web_fetch"}),
        ],
    )
    if response.usage and response.usage.server_tool_requests:
        print(response.usage.server_tool_requests.get("web_search"))


def run_client_tools(client: Tessaract, history: list) -> None:
    from tessaract import (
        ComputerToolResult,
        ComputerUseTool,
        ShellCommandOutput,
        ShellTool,
        ShellToolResult,
    )

    response = client.send(model="ant/m", input=history, tools=[ComputerUseTool(), ShellTool()])
    history.extend(response.output)

    for item in response.output:
        if item.type == "computer_call":
            for action in item.actions:
                if action.type == "click":
                    print(action.x, action.y, action.button, action.clicks, action.modifiers)
                elif action.type == "scroll":
                    print(action.dx, action.dy, action.unit)
                elif action.type == "type":
                    print(action.text)
                elif action.type == "provider_action":
                    print(action.provider_type, action.payload)
            history.append(ComputerToolResult(
                call_id=item.call_id, screenshot=b"png", acknowledged_safety_checks=item.safety_checks,
            ))
        elif item.type == "shell_call" and item.executed_by == "provider":
            print([output.stdout for output in item.output])
        elif item.type == "shell_call":
            print(item.commands, item.timeout_ms, item.restart)
            history.append(ShellToolResult(
                call_id=item.call_id, output=[ShellCommandOutput(stdout="ok") for _ in item.commands],
                max_output_length=item.max_output_length,
            ))

    for event in client.send(model="ant/m", input=history, tools=[ShellTool()], stream=True):
        if event.type == "tool_call.started" and event.call_type == "shell_call":
            print(event.call_id, event.name)


def send_with_hosted_shell(client: Tessaract) -> None:
    from tessaract import ContainerEnvironment, ShellTool

    client.send(
        model="oai/m",
        input="hi",
        tools=[ShellTool(environment=ContainerEnvironment(memory_limit="4g", allowed_domains=["pypi.org"]))],
    )

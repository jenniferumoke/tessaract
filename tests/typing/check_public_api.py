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

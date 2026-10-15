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
        elif item.type == "provider_output":
            print(item.provider_type)


def send_with_cache(client: Tessaract) -> None:
    from tessaract import CacheOptions

    response = client.send(model="ant/m", input="hi", cache=CacheOptions(retention="long", key="k"))
    print(response.usage)

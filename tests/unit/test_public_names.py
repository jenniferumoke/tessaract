'''The type names user code matches on are public API.

    Renaming one breaks `item.type == ...` checks silently, and giving a native item a canonical
    type moves it out of ProviderOutputItem. When one of these tests fails, the change needs a
    CHANGELOG entry and a docs update, not just a new expected value here.
'''
from typing import get_args

from tessaract.types.input_types import InputType
from tessaract.types.output_types import OutputType
from tessaract.types.streaming.event_types import StreamEventUnion, ToolCallStartedEvent


def _type_names(models) -> set[str]:
    return {model.model_fields["type"].default for model in models}

def test_output_item_types():
    assert _type_names(get_args(OutputType)) == {
        "assistant_message",
        "text",
        "reasoning",
        # calls: <tool>_call
        "function_call",
        "web_search_call",
        "computer_call",
        "shell_call",
        # native items without a canonical type yet
        "provider_output",
    }

def test_input_item_types():
    # UserMessage and SystemPrompt are told apart by role, not type
    results = [model for model in InputType.__subclasses__() if "type" in model.model_fields]

    assert _type_names(results) == {
        # results: <tool>_tool_result
        "function_tool_result",
        "computer_tool_result",
        "shell_tool_result",
        "provider_tool_result",
    }

def test_stream_event_types():
    assert _type_names(get_args(StreamEventUnion)) == {
        "response_started",
        "response.completed",
        "response.failed",
        "text.delta",
        "reasoning.started",
        "reasoning_summary.delta",
        "reasoning_text.delta",
        # calls your code runs; call_type is the output item type the call becomes
        "tool_call.started",
        "tool_arguments.delta",
        # calls the provider runs
        "server_tool_call.started",
        "output_item.done",
        "custom_provider_event",
    }

def test_tool_call_started_call_types_are_client_call_item_types():
    call_types = set(get_args(ToolCallStartedEvent.model_fields["call_type"].annotation))

    assert call_types == {"function_call", "computer_call", "shell_call"}
    assert call_types <= _type_names(get_args(OutputType))

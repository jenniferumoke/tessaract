"""A research agent: OpenAI runs the web searches, your code keeps the notes.

WebSearchTool is provider-executed: OpenAI searches on its own servers and the results come back
in the response, so there is nothing to send back. save_finding is a function tool your code runs.
Both go in the same tools list.

Run with:
    export OPENAI_API_KEY=sk-...
    uv run --extra openai python examples/research_agent.py "How do heat pumps perform below -20C?"
"""

import os
import sys

from tessaract import (
    FunctionTool,
    FunctionToolResult,
    InputSchema,
    OpenAIProvider,
    Property,
    ReasoningOptions,
    Tessaract,
    UserMessage,
    WebSearchTool,
)

MODEL = os.environ.get("TESSARACT_MODEL", "oai/gpt-5.6-luna")
MAX_STEPS = 15
INSTRUCTIONS = (
    "You are a research assistant. Search the web, and call save_finding once for every fact you "
    "will rely on, with the URL it came from. When you have enough, write a short report that "
    "cites its sources."
)

client = Tessaract(providers={"oai": OpenAIProvider()})  # reads OPENAI_API_KEY

TOOLS = [
    # provider-executed: OpenAI runs it. allowed_domains=[...] limits the search to trusted sites.
    WebSearchTool(provider_options={"search_context_size": "high"}),
    # client-executed: your code runs it
    FunctionTool(
        name="save_finding",
        description="Save one fact for the final report, with the URL of the page it came from.",
        input_schema=InputSchema(
            properties={
                "claim": Property(type="string", description="The fact, in one sentence."),
                "url": Property(type="string", description="The page the fact came from."),
            },
            required=["claim", "url"],
            additionalProperties=False,
        ),
    ),
]


def research(question: str) -> None:
    history: list = [UserMessage(content=question)]
    findings: list[dict] = []
    searches = 0

    for _ in range(MAX_STEPS):
        response = client.send(
            model=MODEL,
            input=history,
            tools=TOOLS,
            instructions=INSTRUCTIONS,
            reasoning=ReasoningOptions(effort="medium"),
        )
        history.extend(response.output)

        for item in response.output:
            if item.type == "web_search_call":
                if item.action == "search":
                    print(f"  [search] {', '.join(item.queries)}  ({len(item.sources)} sources)")
                else:                                      # open_page / find_in_page
                    print(f"  [{item.action}] {item.url}")

        if response.usage and response.usage.server_tool_requests:
            searches += response.usage.server_tool_requests.get("web_search", 0)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            print_report(response, findings, searches)
            return

        for call in calls:
            findings.append(call.arguments)
            print(f"  [finding] {call.arguments['claim']}")
            history.append(FunctionToolResult(call_id=call.call_id, result="saved"))

    print("Stopped: too many steps.")


def print_report(response, findings: list[dict], searches: int) -> None:
    print(f"\n{response.output_text}\n")

    print("Cited in the report:")
    for item in response.output:
        if item.type == "assistant_message":
            for part in item.content if isinstance(item.content, list) else [item.content]:
                for citation in part.annotations:
                    print(f"  - {citation.title or citation.source}: {citation.source}")

    print(f"\n{len(findings)} findings saved, {searches} searches run.")


if __name__ == "__main__":
    research(" ".join(sys.argv[1:]) or "What are the main differences between Python 3.14 and 3.15?")

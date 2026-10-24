"""Ask OpenAI and Anthropic the same search question in parallel, then reconcile the answers.

Each provider runs its own web search, so the two answers can rest on different sources. A third
call gets both answers and their sources as text, and writes one answer that says where they agree
and where they don't.

Run with:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra openai --extra anthropic python examples/fan_out_compare.py "Is the EU AI Act in force for general-purpose models yet?"
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from tessaract import AnthropicProvider, OpenAIProvider, Tessaract, UserMessage, WebSearchTool

MODELS = ["oai/gpt-5.6-luna", "ant/claude-opus-5-5"]
RECONCILER = os.environ.get("RECONCILER_MODEL", "ant/claude-opus-5-5")
RECONCILER_INSTRUCTIONS = (
    "You compare answers from two research assistants. Write one answer. Say where they agree, "
    "point out every disagreement and which sources support each side, and cite URLs."
)

client = Tessaract(providers={"oai": OpenAIProvider(), "ant": AnthropicProvider()})


@dataclass
class Answer:
    model: str
    text: str
    sources: dict[str, str] = field(default_factory=dict)   # url -> title


def ask(model: str, question: str) -> Answer:
    history: list = [UserMessage(content=question)]
    answer = Answer(model=model, text="")

    while True:
        response = client.send(model=model, input=history, tools=[WebSearchTool()], max_tokens=8_000)
        history.extend(response.output)
        answer.text += response.output_text

        for item in response.output:
            if item.type == "web_search_call":
                for source in item.sources:
                    answer.sources.setdefault(source.url, source.title or "")
            elif item.type == "assistant_message":
                for part in item.content if isinstance(item.content, list) else [item.content]:
                    for citation in part.annotations:
                        if citation.source:
                            answer.sources[citation.source] = citation.title or answer.sources.get(citation.source, "")

        if response.incomplete_reason != "paused":   # Anthropic can pause a long search turn
            return answer


def as_text(answer: Answer) -> str:
    sources = "\n".join(f"- {title} {url}".strip() for url, title in answer.sources.items())
    return f"## Answer from {answer.model}\n\n{answer.text}\n\nSources:\n{sources or '(none)'}"


def main(question: str) -> None:
    with ThreadPoolExecutor() as pool:
        answers = list(pool.map(lambda model: ask(model, question), MODELS))

    for answer in answers:
        print(f"{answer.model}: {len(answer.sources)} sources, {len(answer.text)} characters")

    # the answers come from different providers, so they are passed on as text
    response = client.send(
        model=RECONCILER,
        input=f"Question: {question}\n\n" + "\n\n".join(as_text(answer) for answer in answers),
        instructions=RECONCILER_INSTRUCTIONS,
    )
    print(f"\n{response.output_text}")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "What is the current state of solid-state battery production for cars?")

"""A streaming reasoning + tool-calling agent built on Tessaract.

Reuses the tools from examples/agent.py and streams reasoning summaries and
answer text as they are generated.

Run with OpenAI, or set TESSARACT_MODEL=ant/claude-opus-5-5 for Anthropic:
    export OPENAI_API_KEY=sk-...
    uv run --extra openai python examples/streaming_agent.py
"""

from agent import INSTRUCTIONS, MAX_STEPS, MAX_TOKENS, MODEL, REASONING, TOOLS, client, execute

from tessaract import UserMessage


def run_turn(history: list, user_text: str) -> None:
    history.append(UserMessage(content=user_text))

    for _ in range(MAX_STEPS):
        completed = None

        for event in client.send(
            model=MODEL,
            input=history,
            tools=TOOLS,
            reasoning=REASONING,
            instructions=INSTRUCTIONS,
            max_tokens=MAX_TOKENS,
            stream=True,
        ):
            match event.type:
                case "reasoning.started":
                    print("\n  [thinking] ", end="", flush=True)
                case "reasoning_summary.delta":
                    print(event.delta, end="", flush=True)
                case "text.delta":
                    print(event.delta, end="", flush=True)
                case "tool_call.started":
                    print(f"\n  [tool] calling {event.name}...", flush=True)
                case "output_item.done" if event.item.type == "function_call":
                    print(f"  [tool] {event.item.name}({event.item.arguments})")
                case "response.failed":
                    raise RuntimeError(f"Response failed: {event.message}")
                case "response.completed":
                    completed = event.response

        if completed is None:
            raise RuntimeError("Stream ended without a response.completed event")

        history.extend(completed.output)

        calls = [item for item in completed.output if item.type == "function_call"]
        if not calls:
            print()
            return

        for call in calls:
            history.append(execute(call))

    print("\nStopped: too many tool-calling steps.")


def main() -> None:
    history: list = []
    print("Type 'q' to quit.")
    while True:
        user_text = input("\nyou > ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        if user_text:
            print("agent > ", end="", flush=True)
            run_turn(history, user_text)


if __name__ == "__main__":
    main()

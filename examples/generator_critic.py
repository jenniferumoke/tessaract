"""Generator and critic on different providers.

Claude writes code and tests it in Anthropic's code execution container. GPT reviews the code and
the test run, and returns a verdict through a function tool. The loop repeats until the critic
approves or MAX_ROUNDS runs out.

Output items replay from their native form, so the generator's history can't be sent to the
critic. The critic gets plain text instead: the code and the commands the container ran.

Run with:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra openai --extra anthropic python examples/generator_critic.py
"""

import os
import re

from tessaract import (
    AnthropicProvider,
    ContainerEnvironment,
    FunctionTool,
    InputSchema,
    OpenAIProvider,
    Property,
    ShellTool,
    Tessaract,
    UserMessage,
)

GENERATOR = os.environ.get("GENERATOR_MODEL", "ant/claude-opus-5-5")
CRITIC = os.environ.get("CRITIC_MODEL", "oai/gpt-6.1-sol")
MAX_ROUNDS = 4

TASK = (
    "Write a Python function parse_duration(text: str) -> int that converts strings such as "
    "'1h30m', '45s', '2h' and '1h 5m 10s' to seconds, and raises ValueError on anything else. "
    "Write pytest tests for it and run them in your container."
)
GENERATOR_INSTRUCTIONS = (
    "You write production-quality Python. Test your code in the container before answering. "
    "End your answer with the complete final module in a single ```python block."
)
CRITIC_INSTRUCTIONS = (
    "You are a strict code reviewer. Review the code and its test run for correctness, edge cases "
    "and clarity. Approve only code you would merge. Call submit_review with your verdict."
)

client = Tessaract(providers={"ant": AnthropicProvider(), "oai": OpenAIProvider()})

SUBMIT_REVIEW = FunctionTool(
    name="submit_review",
    description="Submit the review verdict.",
    input_schema=InputSchema(
        properties={
            "approved": Property(type="boolean", description="True if the code can be merged as is."),
            "issues": Property(
                type="array",
                items=Property(type="string"),
                description="What must change before approval. Empty when approved.",
            ),
        },
        required=["approved", "issues"],
        additionalProperties=False,
    ),
)


def generate(history: list, container_id: str | None) -> tuple[str, list, str | None]:
    '''Runs one generator turn. Returns its text, the shell calls it made, and the container id.'''
    text, shell_calls = "", []
    while True:
        response = client.send(
            model=GENERATOR,
            input=history,
            tools=[ShellTool(environment=ContainerEnvironment(container_id=container_id))],
            instructions=GENERATOR_INSTRUCTIONS,
            max_tokens=16_000,
        )
        history.extend(response.output)
        text += response.output_text
        shell_calls += [item for item in response.output if item.type == "shell_call"]
        container_id = response.container_id or container_id

        if response.incomplete_reason != "paused":   # a long code execution turn can pause
            return text, shell_calls, container_id


def test_log(shell_calls: list) -> str:
    '''The commands the container ran and their output, as text another provider can read.'''
    lines = []
    for call in shell_calls:
        for command, output in zip(call.commands, call.output):
            lines += [f"$ {command}", output.stdout[-2000:], output.stderr[-2000:], f"(exit {output.exit_code})"]
    return "\n".join(line for line in lines if line)


def review(code: str, log: str) -> dict:
    response = client.send(
        model=CRITIC,
        input=f"Task:\n{TASK}\n\nCode:\n```python\n{code}\n```\n\nTest run:\n{log or '(no commands run)'}",
        tools=[SUBMIT_REVIEW],
        instructions=CRITIC_INSTRUCTIONS,
        request_options={"tool_choice": {"type": "function", "name": "submit_review"}},
    )
    call = next(item for item in response.output if item.type == "function_call")
    return call.arguments


def main() -> None:
    history: list = [UserMessage(content=TASK)]
    container_id = None

    for round_number in range(1, MAX_ROUNDS + 1):
        print(f"\n--- round {round_number}: {GENERATOR} drafts ---")
        text, shell_calls, container_id = generate(history, container_id)
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        code = blocks[-1] if blocks else text
        print(f"  {len(shell_calls)} commands run in container {container_id}")

        print(f"--- round {round_number}: {CRITIC} reviews ---")
        verdict = review(code, test_log(shell_calls))
        if verdict["approved"]:
            print(f"Approved in round {round_number}.\n\n{code}")
            return

        for issue in verdict["issues"]:
            print(f"  - {issue}")
        # same provider, so the generator keeps its full history; the feedback is plain text
        history.append(UserMessage(content="A reviewer asked for these changes:\n- " + "\n- ".join(verdict["issues"])))

    print(f"Not approved after {MAX_ROUNDS} rounds.")


if __name__ == "__main__":
    main()

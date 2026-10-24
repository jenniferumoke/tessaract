"""A data analyst agent: Claude runs Python in a container Anthropic manages.

ShellTool with a ContainerEnvironment is provider-executed: the commands run in Anthropic's code
execution container and come back with their output, so there is nothing to send back. The CSV is
uploaded with the Files API and copied into the container; the charts the model saves come back as
file ids you can download.

Run with your own CSV, or leave the path out to use a generated one:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run --extra anthropic python examples/data_analyst.py [sales.csv]
"""

import csv
import os
import random
import sys
import tempfile
from pathlib import Path

from tessaract import AnthropicProvider, ContainerEnvironment, ShellTool, Tessaract, UserMessage

MODEL = os.environ.get("TESSARACT_MODEL", "ant/claude-opus-5-5")
OUTPUT_DIR = Path("analyst_output")
INSTRUCTIONS = (
    "You are a data analyst. The user's CSV is in your container. Use Python and pandas to answer. "
    "Save every chart you make as a PNG file, and say which file shows what."
)
QUESTIONS = [
    "Which region grew fastest month over month? Make a line chart of revenue by region.",
    "Now compare the average order value per product, and chart it.",   # reuses the same container
]

provider = AnthropicProvider()  # reads ANTHROPIC_API_KEY
client = Tessaract(providers={"ant": provider})


def sample_csv() -> Path:
    path = Path(tempfile.mkdtemp()) / "sales.csv"
    random.seed(7)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "region", "product", "orders", "revenue"])
        for month in range(1, 13):
            for region, growth in (("EMEA", 1.02), ("APAC", 1.06), ("AMER", 1.01)):
                for product in ("basic", "pro", "enterprise"):
                    orders = int(random.randint(40, 60) * growth ** month)
                    price = {"basic": 20, "pro": 75, "enterprise": 400}[product]
                    writer.writerow([f"2026-{month:02d}", region, product, orders, orders * price])
    return path


def send(history: list, environment: ContainerEnvironment):
    '''Sends one request, and resends while a long code execution turn pauses.'''
    while True:
        response = client.send(
            model=MODEL,
            input=history,
            tools=[ShellTool(environment=environment)],
            instructions=INSTRUCTIONS,
            max_tokens=16_000,
        )
        history.extend(response.output)

        for call in (item for item in response.output if item.type == "shell_call"):
            for command, output in zip(call.commands, call.output):
                print(f"  [{call.executed_by}] $ {command.splitlines()[0][:80]}  -> exit {output.exit_code}")
            download(call.file_ids)

        if response.incomplete_reason != "paused":
            return response
        # keep the paused turn in the same container
        environment = ContainerEnvironment(container_id=response.container_id)


def download(file_ids: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for file_id in file_ids:
        name = provider.client.files.retrieve_metadata(file_id).filename
        provider.client.files.download(file_id).write_to_file(OUTPUT_DIR / name)
        print(f"  [file] saved {OUTPUT_DIR / name}")


def main(csv_path: Path) -> None:
    with csv_path.open("rb") as f:
        uploaded = provider.client.files.upload(file=(csv_path.name, f, "text/csv"))

    history: list = []
    # the first request starts a container with the CSV in it
    environment = ContainerEnvironment(file_ids=[uploaded.id])

    for question in QUESTIONS:
        print(f"\nyou > {question}")
        history.append(UserMessage(content=question))
        response = send(history, environment)
        print(f"analyst > {response.output_text}")

        # later questions reuse the container, with the CSV and anything the model computed so far
        environment = ContainerEnvironment(container_id=response.container_id)

    provider.client.files.delete(uploaded.id)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else sample_csv())

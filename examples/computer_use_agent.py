"""A computer use agent: OpenAI decides what to do, your code does it in a Playwright browser.

ComputerUseTool is client-executed: each computer_call comes back with a list of actions, your
code runs them and answers with a ComputerToolResult holding a screenshot of the result.
The browser has no address bar, so the agent starts on START_URL.

Run with:
    export OPENAI_API_KEY=sk-...
    uv run --extra openai --with playwright playwright install chromium
    uv run --extra openai --with playwright python examples/computer_use_agent.py
"""

import os
import sys

from playwright.sync_api import Page, sync_playwright  # pyright: ignore[reportMissingImports]

from tessaract import ComputerToolResult, ComputerUseTool, OpenAIProvider, Tessaract, UserMessage

MODEL = os.environ.get("TESSARACT_MODEL", "oai/gpt-5.6-luna")
START_URL = os.environ.get("START_URL", "https://news.ycombinator.com")
TASK = "What are the titles of the top three stories, and how many points does each have?"
WIDTH, HEIGHT = 1280, 800
MAX_STEPS = 30

# OpenAI sends key names such as "CTRL" and "ENTER"; Playwright wants "Control" and "Enter"
KEYS = {
    "CTRL": "Control", "CMD": "Meta", "META": "Meta", "ALT": "Alt", "OPTION": "Alt", "SHIFT": "Shift",
    "ENTER": "Enter", "RETURN": "Enter", "ESC": "Escape", "ESCAPE": "Escape", "TAB": "Tab",
    "SPACE": " ", "BACKSPACE": "Backspace", "DELETE": "Delete", "HOME": "Home", "END": "End",
    "PAGEUP": "PageUp", "PAGEDOWN": "PageDown", "UP": "ArrowUp", "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft", "RIGHT": "ArrowRight", "ARROWUP": "ArrowUp", "ARROWDOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft", "ARROWRIGHT": "ArrowRight",
}

client = Tessaract(providers={"oai": OpenAIProvider()})  # reads OPENAI_API_KEY


def key(name: str) -> str:
    return KEYS.get(name.upper(), name)


def hold(page: Page, modifiers: list[str], down: bool) -> None:
    for modifier in modifiers:
        (page.keyboard.down if down else page.keyboard.up)(key(modifier))


def run_action(page: Page, action) -> None:
    match action.type:
        case "click":
            hold(page, action.modifiers, down=True)
            if action.x is not None and action.y is not None:
                button = action.button if action.button in ("left", "right", "middle") else "left"
                page.mouse.click(action.x, action.y, button=button, click_count=action.clicks)
            hold(page, action.modifiers, down=False)
        case "scroll":
            if action.x is not None and action.y is not None:
                page.mouse.move(action.x, action.y)
            page.mouse.wheel(action.dx, action.dy)              # OpenAI scrolls in pixels
        case "type":
            page.keyboard.type(action.text)
        case "keypress":
            for _ in range(action.repeat):
                page.keyboard.press("+".join(key(k) for k in action.keys))
        case "move":
            page.mouse.move(action.x, action.y)
        case "drag":
            start, *rest = action.path
            page.mouse.move(start.x, start.y)
            page.mouse.down()
            for point in rest:
                page.mouse.move(point.x, point.y)
            page.mouse.up()
        case "wait":
            page.wait_for_timeout((action.duration or 1) * 1000)
        case "screenshot":
            pass                                                # every result carries one anyway
        case _:
            print(f"  [skipped] {action.type} is not supported by this agent")


def confirm(call) -> bool:
    '''OpenAI flags risky steps with safety checks. A person confirms them before the agent goes on.'''
    for check in call.safety_checks:
        print(f"  [safety check] {check.code}: {check.message}")
    return input("  Continue? [y/N] ").strip().lower() == "y"


def main(task: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(START_URL)

        history: list = [UserMessage(content=f"The browser is open on {START_URL}. {task}")]

        for _ in range(MAX_STEPS):
            response = client.send(model=MODEL, input=history, tools=[ComputerUseTool()])
            history.extend(response.output)

            calls = [item for item in response.output if item.type == "computer_call"]
            if not calls:
                print(f"\nagent > {response.output_text}")
                break

            for call in calls:
                if call.safety_checks and not confirm(call):
                    print("Stopped at a safety check.")
                    return

                for action in call.actions:                     # OpenAI can batch several actions
                    print(f"  [{action.type}] {action.model_dump(exclude={'type'}, exclude_defaults=True)}")
                    run_action(page, action)

                history.append(ComputerToolResult(
                    call_id=call.call_id,
                    screenshot=page.screenshot(),               # OpenAI needs one in every result
                    acknowledged_safety_checks=call.safety_checks,
                ))
        else:
            print("Stopped: too many steps.")

        browser.close()


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or TASK)

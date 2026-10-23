from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field

# Coordinates are in the pixel space of the screenshots you return, with the origin at the top left.
# Key names are passed through as the provider sends them (e.g. Anthropic "Return", OpenAI "ENTER").

MouseButton: TypeAlias = Literal["left", "right", "middle", "back", "forward"]


class Point(BaseModel):
    x: int
    y: int

class Click(BaseModel):
    '''Click a mouse button. x and y are None when the model clicks at the current cursor position.

        clicks: 1, 2 for a double click, or 3 for a triple click
        modifiers: keys held during the click, e.g. ["shift"]
    '''
    type: Literal["click"] = "click"
    x: int | None = None
    y: int | None = None
    button: MouseButton = "left"
    clicks: int = 1
    modifiers: list[str] = Field(default_factory=list)

class Scroll(BaseModel):
    '''Scroll at (x, y), or at the current cursor position when they are None.

        dx / dy: positive scrolls right / down. unit says what they count:
        OpenAI sends "pixels", Anthropic sends scroll-wheel "clicks".
    '''
    type: Literal["scroll"] = "scroll"
    x: int | None = None
    y: int | None = None
    dx: int = 0
    dy: int = 0
    unit: Literal["pixels", "clicks"]
    modifiers: list[str] = Field(default_factory=list)

class TypeText(BaseModel):
    type: Literal["type"] = "type"
    text: str

class KeyPress(BaseModel):
    '''Press keys together, e.g. ["ctrl", "s"], repeat times.'''
    type: Literal["keypress"] = "keypress"
    keys: list[str]
    repeat: int = 1

class HoldKey(BaseModel):
    '''Hold keys down for duration seconds. Anthropic only.'''
    type: Literal["hold_key"] = "hold_key"
    keys: list[str]
    duration: float

class MoveMouse(BaseModel):
    type: Literal["move"] = "move"
    x: int
    y: int
    modifiers: list[str] = Field(default_factory=list)

class Drag(BaseModel):
    '''Press at the first point, move through the path, and release at the last point.'''
    type: Literal["drag"] = "drag"
    path: list[Point]
    modifiers: list[str] = Field(default_factory=list)

class MouseDown(BaseModel):
    type: Literal["mouse_down"] = "mouse_down"
    button: MouseButton = "left"

class MouseUp(BaseModel):
    type: Literal["mouse_up"] = "mouse_up"
    button: MouseButton = "left"

class Screenshot(BaseModel):
    type: Literal["screenshot"] = "screenshot"

class Zoom(BaseModel):
    '''Return a full-resolution screenshot of a region: (x0, y0) top left, (x1, y1) bottom right. Anthropic only.'''
    type: Literal["zoom"] = "zoom"
    region: tuple[int, int, int, int]

class Wait(BaseModel):
    '''Pause. duration is in seconds; None (OpenAI) leaves the length to you.'''
    type: Literal["wait"] = "wait"
    duration: float | None = None

class CursorPosition(BaseModel):
    '''Report the cursor position. Anthropic only; return it as text in ComputerToolResult.output.'''
    type: Literal["cursor_position"] = "cursor_position"

class ProviderComputerAction(BaseModel):
    '''A native action without a canonical equivalent yet, e.g. a member added in a newer toolset version.'''
    type: Literal["provider_action"] = "provider_action"
    provider_type: str
    payload: dict[str, Any] = Field(default_factory=dict)

ComputerAction: TypeAlias = Annotated[
    Click | Scroll | TypeText | KeyPress | HoldKey | MoveMouse | Drag | MouseDown | MouseUp
    | Screenshot | Zoom | Wait | CursorPosition | ProviderComputerAction,
    Field(discriminator="type"),
]

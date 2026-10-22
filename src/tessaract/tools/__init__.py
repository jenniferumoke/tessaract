from typing import TypeAlias

from .computer import ComputerUseTool
from .function import FunctionTool, InputSchema, Property
from .provider import ProviderTool
from .shell import ContainerEnvironment, ShellTool
from .web_search import UserLocation, WebSearchTool

Tool: TypeAlias = FunctionTool | WebSearchTool | ComputerUseTool | ShellTool | ProviderTool

__all__ = [
    "ComputerUseTool",
    "ContainerEnvironment",
    "FunctionTool",
    "InputSchema",
    "Property",
    "ProviderTool",
    "ShellTool",
    "Tool",
    "UserLocation",
    "WebSearchTool"
]

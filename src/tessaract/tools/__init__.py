from typing import TypeAlias

from .function import FunctionTool, InputSchema, Property
from .provider import ProviderTool
from .web_search import UserLocation, WebSearchTool

Tool: TypeAlias = FunctionTool | WebSearchTool | ProviderTool

__all__ = [
    "FunctionTool",
    "InputSchema",
    "Property",
    "ProviderTool",
    "Tool",
    "UserLocation",
    "WebSearchTool"
]

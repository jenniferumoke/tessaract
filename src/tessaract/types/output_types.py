from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

class OutputItem(BaseModel):
    raw: Any

class Annotation(BaseModel):
    type: Literal["citation", "file_path"] = "citation"
    provider_type: str | None = None

    source: str | None = None
    title: str | None = None
    cited_text: str | None = None

    provider: Literal["anthropic", "openai"]

    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

class TextOutputItem(OutputItem):
    type: Literal["text"] = "text"
    text: str
    annotations: list[Annotation] = Field(
        default_factory=list,
    )

class AssistantMessage(BaseModel):
    type: Literal["assistant_message"] = "assistant_message"
    role: Literal["assistant"] = "assistant"
    content: TextOutputItem | list[TextOutputItem]
    raw: Any

class ReasoningOutputItem(OutputItem):
    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    text: str | list | None = None
    content: str | list | None = None

class FunctionCallOutputItem(OutputItem):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class WebSource(BaseModel):
    url: str
    title: str | None = None
    # how old the page is, as the provider reports it (Anthropic only)
    page_age: str | None = None

class WebSearchCallOutputItem(OutputItem):
    '''A web search the provider ran.

        action: "search", or "open_page" / "find_in_page" when OpenAI browses a result
        queries: the search queries; empty for open_page and find_in_page
        url / pattern: the page opened or searched, for open_page and find_in_page
        sources: the pages found. OpenAI returns urls only.
        error: the provider's error code when status is "failed"

        On Anthropic, one call is a server_tool_use block followed by a web_search_tool_result block;
        raw is the list of both, and both are replayed unchanged. On OpenAI, raw is the web_search_call item.
    '''
    type: Literal["web_search_call"] = "web_search_call"
    call_id: str
    status: Literal["in_progress", "searching", "completed", "failed"]
    action: str = "search"
    queries: list[str] = Field(default_factory=list)
    url: str | None = None
    pattern: str | None = None
    sources: list[WebSource] = Field(default_factory=list)
    error: str | None = None
    provider: Literal["anthropic", "openai"]

class ProviderOutputItem(OutputItem):
    type: Literal["provider_output"] = "provider_output"
    provider_type: str

OutputType: TypeAlias = (
    AssistantMessage | 
    ReasoningOutputItem | 
    FunctionCallOutputItem | 
    WebSearchCallOutputItem |
    TextOutputItem | 
    ProviderOutputItem
    )

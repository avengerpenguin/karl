import re
from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

SEARCH_TOOL_NAME = "search_tools"


class ToolSearchMiddleware(AgentMiddleware):
    """Hide searchable tools from the model until `search_tools` reveals them."""

    def __init__(
        self, searchable_tools: Sequence[BaseTool], max_results: int = 8
    ) -> None:
        super().__init__()
        self.catalog: dict[str, BaseTool] = {t.name: t for t in searchable_tools}
        self.max_results = max_results
        catalog = self.catalog

        @tool(SEARCH_TOOL_NAME, response_format="content_and_artifact")
        def search_tools(query: str) -> tuple[str, list[str]]:
            """Find and enable additional tools (email, Jira, GitLab, Slack, Confluence,
            calendar, databases, Google, web, etc.). Call this with a few keywords
            describing what you need; matching tools become callable on your next step."""
            terms = set(re.findall(r"\w+", query.lower()))
            scored = []
            for name, t in catalog.items():
                haystack = set(re.findall(r"\w+", f"{name} {t.description}".lower()))
                score = len(terms & haystack)
                if score:
                    scored.append((score, name))

            names = [n for _, n in sorted(scored, reverse=True)[: self.max_results]]
            if not names:
                return "No matching tools. Try different keywords.", []

            lines = [
                f"- {n}: {catalog[n].description.splitlines()[0][:160]}" for n in names
            ]
            return "Enabled tools:\n" + "\n".join(lines), names

        self.tools = [search_tools]

    def _revealed(self, request: ModelRequest) -> set[str]:
        revealed: set[str] = set()
        for message in request.messages:
            if (
                isinstance(message, ToolMessage)
                and message.name == SEARCH_TOOL_NAME
                and isinstance(message.artifact, list)
            ):
                revealed.update(message.artifact)
        return revealed

    def _filter(self, request: ModelRequest) -> ModelRequest:
        revealed = self._revealed(request)
        tools = [
            t
            for t in request.tools
            if isinstance(t, dict)  # provider-native tool dicts: leave alone
            or t.name not in self.catalog
            or t.name in revealed
        ]
        return request.override(tools=tools)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._filter(request))

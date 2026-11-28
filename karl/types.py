from dataclasses import dataclass
from typing import Callable, Awaitable, Protocol

from langgraph.graph.state import CompiledStateGraph


class Elicitor(Protocol):
    async def request_url(
        self,
        *,
        message: str,
        url: str,
    ) -> str:
        """Present a URL and return an opaque elicitation ID."""

    async def complete(self, elicitation_id: str) -> None:
        """Mark an outstanding elicitation as completed."""


@dataclass(frozen=True)
class AgentSessionContext:
    session_id: str
    elicitor: Elicitor


AgentFactory = Callable[[AgentSessionContext], Awaitable[CompiledStateGraph]]

from typing import Callable, Awaitable

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

AgentFactory = Callable[[BaseChatModel], Awaitable[CompiledStateGraph]]

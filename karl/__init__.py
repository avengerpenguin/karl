import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, AnyMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama


BEEPER_MCP = {
    "messages": {
        "transport": "http",
        "url": "http://localhost:23373/v0/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('BEEPER_TOKEN')}",
        },
    },
}


async def create(model="deepseek-r1:32b", tools=None):
    from karl.tools.custom_mcp import load_mcp_tools_isolated

    llm = ChatOllama(model=model)
    if not tools:
        tools = await load_mcp_tools_isolated(BEEPER_MCP)
    return create_agent(model=llm, tools=tools)


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)
    # N.B. all content is available through token.content_blocks


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

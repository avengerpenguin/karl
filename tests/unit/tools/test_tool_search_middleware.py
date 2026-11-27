import pytest
from deepagents import create_deep_agent
from karl.tools.tool_search_middleware import SEARCH_TOOL_NAME, ToolSearchMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph


@pytest.fixture
def tool_search_agent(chat_model, get_weather, get_time):
    tools = [get_weather, get_time]
    return create_deep_agent(
        model=chat_model,
        tools=tools,
        middleware=[ToolSearchMiddleware(tools)],
    )


def requested_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Names of every tool the model asked to call, in order."""
    return [
        call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def executed_tools(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Every tool result that came back, in order."""
    return [message for message in messages if isinstance(message, ToolMessage)]


def test_no_tools_searched_by_default(tool_search_agent: CompiledStateGraph):
    response = tool_search_agent.invoke(
        {"messages": [{"role": "user", "content": "What is two plus two?"}]},
    )
    messages = response["messages"]

    assert requested_tool_calls(messages) == []
    assert executed_tools(messages) == []

    answer = messages[-1].content
    assert "4" in answer or "four" in answer.lower()


def test_tool_searched_for_when_needed(tool_search_agent: CompiledStateGraph):
    response = tool_search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current time? Use the get_time tool to look it up.",
                }
            ]
        },
    )
    messages = response["messages"]

    # Exactly these calls, in exactly this order, and nothing else
    assert requested_tool_calls(messages) == [SEARCH_TOOL_NAME, "get_time"]

    search_result, time_result = executed_tools(messages)

    # The search revealed get_time (the middleware reads the artifact to unhide tools)
    assert search_result.name == SEARCH_TOOL_NAME
    assert "get_time" in search_result.artifact

    # get_time actually ran and returned its value
    assert time_result.name == "get_time"
    assert time_result.content == "12:00"

    assert "12:00" in messages[-1].content

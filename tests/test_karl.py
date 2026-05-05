from textwrap import dedent

import karl
import pytest
from langchain_core.messages import SystemMessage, AIMessageChunk, AnyMessage, AIMessage, ToolMessage, HumanMessage


@pytest.mark.asyncio
async def test_response():
    agent = await karl.create()
    messages = [
        SystemMessage(dedent("""\
        You are a personal assistant focused on managing the user's various inboxes.
        Currently, you are advising the user through direct messages received on LinkedIn.
        The user has Beeper Desktop installed so you can call local APIs to list accounts, find
        any with LinkedIn as the network and search chats that are not replied to.
        
        Follow what is asked, but generally you are to employ the strategy of:
        
        1. Favour oldest messages to clear the backlog
        2. Advise messages that are time-wasting and can be archived within Beeper
        3. Flag messages that look like real job opportunities and suggest replies to send
        """)),
        HumanMessage(dedent("""\
        Find the oldest LinkedIn direct message I've not replied to.
        """))
    ]
    async for chunk in agent.astream(
            input={"messages": messages},
            stream_mode=["messages", "updates"],
            version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk):
                karl._render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):
                    karl._render_completed_message(update["messages"][-1])

if __name__ == "__main__":
    pytest.main([__file__])

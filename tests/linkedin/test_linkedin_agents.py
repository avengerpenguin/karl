import os

import pytest
from karl.linkedin.agents import create
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

pytestmark = [pytest.mark.vcr]
os.environ["BEEPER_TOKEN"] = "tokeymctokeface"


@pytest.mark.asyncio
async def test_linkedin_agent(chat_model: str | BaseChatModel):
    agent = create(chat_model)
    response: dict = await agent.ainvoke(
        input=dict(
            messages="Please check for messages on LinkedIn and suggest a draft response to the most recent not-replied-to message. Use past replies I have done to help shape the response."
        )
    )
    messages: list[BaseMessage] = response["messages"]

    ai_message = messages[-1]
    print(ai_message.text)
    assert isinstance(ai_message, AIMessage)

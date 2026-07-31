import os

import pytest

from karl.linkedin.agents import create
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI

pytestmark = [pytest.mark.vcr]
os.environ["BEEPER_TOKEN"] = "tokeymctokeface"


MODEL = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy",
    model="mlx-community/Qwen3.6-27B-4bit",
    temperature=0.3,
    streaming=True,
    stream_chunk_timeout=600,
    timeout=900,
)


@pytest.mark.parametrize(
    "model",
    [MODEL, "ollama:gemma4:12b-mlx"],
)
@pytest.mark.asyncio
async def test_linkedin_agent(model: str | BaseChatModel):
    agent = create(model)
    response: dict = await agent.ainvoke(
        input=dict(
            messages="Please check for messages on LinkedIn and suggest a draft response to the most recent not-replied-to message. Use past replies I have done to help shape the response."
        )
    )
    messages: list[BaseMessage] = response["messages"]

    ai_message = messages[-1]
    print(ai_message.text)
    assert isinstance(ai_message, AIMessage)

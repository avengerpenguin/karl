import pytest

from linkedin.tools import find_latest_non_replied_chat, LinkedInChat, find_past_reply_examples


pytestmark = [pytest.mark.vcr(ignore_localhost=False)]


@pytest.mark.asyncio
async def test_find_latest_non_replied_chat():
    # os.environ["BEEPER_TOKEN"] = "tokeymctokeface"
    chat: LinkedInChat = await find_latest_non_replied_chat.ainvoke({})
    assert chat is not None
    assert "Bob Recruiter" in chat.messages[0].content


@pytest.mark.asyncio
async def test_find_past_reply_examples():
    # os.environ["BEEPER_TOKEN"] = "tokeymctokeface"
    examples: list[LinkedInChat] = [
        c async for c in await find_past_reply_examples.ainvoke({})
    ]

    assert examples is not None

    first_chat =  examples[0]

    assert first_chat.messages[0].sender == "recruiter"
    assert "Alice Hirer" in first_chat.messages[0].content

    assert first_chat.messages[1].sender == "user"
    assert "thank you for reaching out" in first_chat.messages[1].content

import os
from typing import AsyncGenerator

from beeper_desktop_api import AsyncBeeperDesktop
from beeper_desktop_api.types import Message
from langchain_core.tools import tool
from pydantic import BaseModel


class ChatMessage(BaseModel):
    sender: str
    content: str


class LinkedInChat(BaseModel):
    messages: list[ChatMessage]


@tool
async def find_latest_non_replied_chat() -> LinkedInChat | None:
    """Find the latest chat on the user's LinkedIn account that has not been replied to."""

    client = AsyncBeeperDesktop(
        access_token=os.getenv("BEEPER_TOKEN"),
    )

    async for chat in client.chats.list(
            account_ids=["linkedin"],
    ):
        if chat.is_archived:
            continue

        messages: list[Message] = [m async for m in client.messages.list(chat_id=chat.id)]
        most_recent_message: Message = messages[0]

        if not most_recent_message.is_sender:
            return LinkedInChat(
                messages=[
                    ChatMessage(
                        sender=m.is_sender and "user" or "recruiter",
                        content=m.text,
                    )
                    for m in messages
                ])

    return None


@tool
async def find_past_reply_examples() -> AsyncGenerator[LinkedInChat]:
    """
    Find past examples of chats where the user did reply and engage.
    Useful to learn how the user has responded in the past to help match style or to include details
    they would normally include.
    Returns a series of chat instances that comprise a series of messages where the sender is either
    the user or a recruiter.
    """
    client = AsyncBeeperDesktop(
        access_token=os.getenv("BEEPER_TOKEN"),
    )
    examples_found = 0

    async for chat in client.chats.list(
            account_ids=["linkedin"],
    ):
        if chat.is_archived:
            continue

        messages: list[Message] = [
            m async for m in client.messages.list(chat_id=chat.id)
        ]
        for message in messages:
            if message.is_sender:
                yield LinkedInChat(
                    messages=[
                        ChatMessage(
                            sender=m.is_sender and "user" or "recruiter",
                            content=m.text,
                        )
                        for m in reversed(messages)
                    ])
                examples_found += 1
                break

        if examples_found >= 5:
            break

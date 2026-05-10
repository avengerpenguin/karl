import os
from datetime import datetime, timezone
from typing import AsyncGenerator

try:
    from beeper_desktop_api import AsyncBeeperDesktop
    from beeper_desktop_api.types import Message
except ImportError:
    raise ImportError("Please install karl[beeper] to use LinkedIn tools")
from langchain_core.tools import tool
from pydantic import BaseModel


class ChatMessage(BaseModel):
    sender: str
    content: str
    message_date: str = ""
    message_age: str = ""


class LinkedInChat(BaseModel):
    messages: list[ChatMessage]


def _create_human_readable_age(timestamp: datetime):
    """
    Converts a timestamp to a human-readable age based on current time.
    Examples:
    - "1 day ago"
    - "2 hours ago"
    - "2 weeks ago"
    - "3 months ago"
    """
    now = datetime.now(tz=timezone.utc)
    delta = now - timestamp
    years = delta.days // 365
    months = (delta.days % 365) // 30
    days = delta.days
    hours = delta.seconds // 3600
    if years > 0:
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif months > 0:
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif days > 0:
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif hours > 0:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        return "Within the last hour"


@tool
async def find_latest_non_replied_chat() -> LinkedInChat | None:
    """
    Find the latest chat on the user's LinkedIn account that has not been replied to.
    Response will include the date and time and age of the message to help understand how fresh it is.
    """

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
                        message_date=m.timestamp.isoformat(),
                        message_age=_create_human_readable_age(m.timestamp),
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
                            message_date=m.timestamp.isoformat(),
                            message_age=_create_human_readable_age(m.timestamp),
                        )
                        for m in reversed(messages)
                    ])
                examples_found += 1
                break

        if examples_found >= 10:
            break


@tool
async def save_draft_message(draft_linkedin_message: str) -> str:
    """
    Save a draft message to send on LinkedIn.
    Once the user is happy with the draft message, save it and they will send it.
    """
    with open("linkedin_draft.txt", "w") as f:
        f.write(draft_linkedin_message.strip() + "\n")

    return "Draft saved"

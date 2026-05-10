from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from textwrap import dedent

try:
    from imapclient import IMAPClient
except ImportError:
    raise ImportError("Please install karl[imap] to use Email tools")

import os
from imapclient.exceptions import InvalidCriteriaError
from langchain_core.tools import tool
from pydantic import BaseModel

IMAP_HOST = "mail.rossfenning.co.uk"
IMAP_PORT = 993
IMAP_USER = "post@rossfenning.co.uk"
IMAP_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

@contextmanager
def imap_connection():
    server = IMAPClient(IMAP_HOST, port=IMAP_PORT, use_uid=True)
    server.login(IMAP_USER, IMAP_PASSWORD)
    try:
        yield server
    finally:
        server.logout()


@tool
def list_folders() -> list[str]:
    """
    List all email folders available on the server.
    """
    with imap_connection() as server:
        return [
            name
            for _, _, name in server.list_folders()
        ]


class Email(BaseModel):
    message_id: int
    subject: str
    sender: str
    date_sent: str
    body: str


class EmailError(BaseModel):
    error_message: str


def _extract_date_sent(message) -> str:
    date_header = message.get("date", "")
    if not date_header:
        return ""

    try:
        return parsedate_to_datetime(date_header).isoformat()
    except (TypeError, ValueError):
        return date_header


@tool
def search_emails(folder: str, criteria: str = "ALL") -> list[int] | EmailError:
    """
    Return a list of messages ids from the given folder matching criteria.

    criteria should be a string that is valid IMAP syntax to pass to SEARCH. Example values:

    'UNSEEN'
    'SMALLER 500'
    'NOT DELETED'
    'TEXT "foo bar" FLAGGED SUBJECT "baz"'
    'SINCE 03-Apr-2005'
    """
    with imap_connection() as server:
        server.select_folder(folder)
        try:
            return server.search(criteria, charset="utf-8")
        except InvalidCriteriaError as e:
            return EmailError(error_message=str(e))


def _extract_text_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = part.get_content_disposition()

            if content_type == "text/plain" and content_disposition != "attachment":
                return part.get_content()

        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = part.get_content_disposition()

            if content_type == "text/html" and content_disposition != "attachment":
                return part.get_content()

        return ""

    return message.get_content()


@tool
def fetch_email(folder: str, message_ids: list[int]):
    """
    Fetch a specific email from a mailbox.

    message_ids should contain at least one numeric id, but can contain multiple ids to fetch multiple emails.

    message_ids are likely to be returned by search_emails.
    """
    with imap_connection() as server:
        server.select_folder(folder)

        emails = []
        for message_id, raw_email in server.fetch(message_ids, ["RFC822"]).items():
            message = BytesParser(policy=policy.default).parsebytes(raw_email[b"RFC822"])

            emails.append(
                Email(
                    message_id=message_id,
                    subject=message.get("subject", ""),
                    sender=message.get("from", ""),
                    date_sent=_extract_date_sent(message),
                    body=_extract_text_body(message),
                )
            )

        return emails


@tool
def draft_email(sender: str, recipient: str, subject: str, body: str) -> str:
    """
    Used to suggest a draft email to the user instead of being able to send emails directly.
    This gives the user the final say on what they want to send.

    sender must be set as it's typically it's the recipient of the email being replied to since multiple emails
    might be pointing to this inbox

    return value is the response from the server. If successful, you should notify the user to check their drafts folder
    """
    with imap_connection() as server:
        imap_response = server.append(
            "Drafts", dedent(f"""\
            From: {sender}
            To: {recipient}
            Subject: {subject}

            {body}
            """), [r"\Draft"])
        return f"Response from server while saving draft email: {imap_response}"

import imaplib
import os
import ssl
from contextlib import contextmanager
from email import message_from_bytes
from email.header import decode_header
from email.policy import default
from typing import Iterator, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


IMAP_HOST = "mail.example.com"
IMAP_PORT = 993
IMAP_USER = "post@example.com"
IMAP_PASSWORD = os.getenv("EMAIL_PASSWORD")


def _decode_header(value: str) -> str:
    parts = []
    for part, encoding in decode_header(value or ""):
        if isinstance(part, bytes):
            parts.append(part.decode(encoding or "utf-8", errors="ignore"))
        else:
            parts.append(part)
    return "".join(parts)


@contextmanager
def imap_connection() -> Iterator[imaplib.IMAP4_SSL]:
    ssl_context = ssl.create_default_context()
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, ssl_context=ssl_context)

    try:
        imap.login(IMAP_USER, IMAP_PASSWORD)
        yield imap
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def _format_email_summary(message_id: str, raw_email: bytes) -> dict[str, Any]:
    msg = message_from_bytes(raw_email, policy=default)

    return {
        "message_id": message_id,
        "from": _decode_header(str(msg.get("From", ""))),
        "to": _decode_header(str(msg.get("To", ""))),
        "subject": _decode_header(str(msg.get("Subject", ""))),
        "date": str(msg.get("Date", "")),
    }


def _format_email_full(message_id: str, raw_email: bytes) -> dict[str, Any]:
    msg = message_from_bytes(raw_email, policy=default)

    body = ""
    body_part = msg.get_body(preferencelist=("plain", "html"))
    if body_part:
        payload = body_part.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = body_part.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="ignore")

    attachments = []
    for attachment in msg.iter_attachments():
        filename = attachment.get_filename()
        if filename:
            attachments.append(filename)

    return {
        "message_id": message_id,
        "from": _decode_header(str(msg.get("From", ""))),
        "to": _decode_header(str(msg.get("To", ""))),
        "cc": _decode_header(str(msg.get("CC", ""))),
        "subject": _decode_header(str(msg.get("Subject", ""))),
        "date": str(msg.get("Date", "")),
        "body": body,
        "attachments": attachments,
    }


class SearchEmailsInput(BaseModel):
    mailbox: str = Field(
        default="INBOX",
        description='Mailbox/folder to search, for example "INBOX" or "INBOX.jobs".',
    )
    criteria: str = Field(
        default="ALL",
        description=(
            "IMAP SEARCH criteria only. Examples: "
            'ALL, FROM "alice@example.com", SUBJECT "invoice", '
            'SENTSINCE "01-Apr-2026". Do not include SELECT or LIST.'
        ),
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of emails to return.",
    )


class FetchEmailInput(BaseModel):
    mailbox: str = Field(
        default="INBOX",
        description='Mailbox/folder containing the email, for example "INBOX.jobs".',
    )
    message_id: str = Field(
        description="IMAP message sequence ID returned by search_emails.",
    )


@tool
def list_mail_folders() -> list[str]:
    """List available IMAP mail folders/mailboxes."""
    with imap_connection() as imap:
        typ, data = imap.list()

        if typ != "OK" or not data:
            return []

        folders = []
        for item in data:
            if not isinstance(item, bytes):
                continue

            line = item.decode(errors="ignore")

            # Typical line:
            # '(\\HasNoChildren) "." "INBOX.jobs"'
            if '"' in line:
                folder = line.split('"')[-2]
            else:
                folder = line.split()[-1]

            folders.append(folder)

        return folders


@tool(args_schema=SearchEmailsInput)
def search_emails(mailbox: str = "INBOX", criteria: str = "ALL", limit: int = 10) -> list[dict[str, Any]]:
    """
    Search emails in a mailbox.

    Use this when the user asks to find, list, triage, or summarise emails.
    The mailbox is selected separately from the search criteria.
    Do not put SELECT or LIST in the criteria.
    """
    forbidden_terms = ["SELECT", "LIST", "STORE", "COPY", "EXPUNGE", "DELETE"]
    upper_criteria = criteria.upper()

    if any(term in upper_criteria for term in forbidden_terms):
        raise ValueError(
            "Invalid search criteria. Do not include IMAP commands such as "
            "SELECT, LIST, STORE, COPY, or EXPUNGE. Use mailbox separately."
        )

    with imap_connection() as imap:
        typ, _ = imap.select(f'"{mailbox}"', readonly=True)
        if typ != "OK":
            raise ValueError(f"Could not select mailbox: {mailbox}")

        typ, data = imap.search(None, criteria)
        if typ != "OK" or not data or not isinstance(data[0], bytes):
            return []

        message_ids = data[0].split()
        message_ids = message_ids[-limit:]

        results = []

        # Most recent first
        for message_id in reversed(message_ids):
            message_id_str = message_id.decode()
            typ, msg_data = imap.fetch(message_id_str, "(RFC822)")

            if (
                typ == "OK"
                and isinstance(msg_data, list)
                and msg_data
                and isinstance(msg_data[0], tuple)
            ):
                raw_email = msg_data[0][1]
                results.append(_format_email_summary(message_id_str, raw_email))

        return results


@tool(args_schema=FetchEmailInput)
def fetch_email(mailbox: str, message_id: str) -> dict[str, Any]:
    """
    Fetch the full content of a specific email.

    Use this after search_emails when the user asks about a specific result,
    asks for detail, or wants a draft reply.
    """
    with imap_connection() as imap:
        typ, _ = imap.select(f'"{mailbox}"', readonly=True)
        if typ != "OK":
            raise ValueError(f"Could not select mailbox: {mailbox}")

        typ, msg_data = imap.fetch(message_id, "(RFC822)")

        if (
            typ == "OK"
            and isinstance(msg_data, list)
            and msg_data
            and isinstance(msg_data[0], tuple)
        ):
            raw_email = msg_data[0][1]
            return _format_email_full(message_id, raw_email)

        raise ValueError(f"Could not fetch email {message_id} from {mailbox}")


email_tools = [
    list_mail_folders,
    search_emails,
    fetch_email,
]
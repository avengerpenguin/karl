import asyncio
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import build_calendar_service

from karl.oauth import in_container

SCOPES = ["https://www.googleapis.com/auth/calendar"]

_AUTH_PROMPT = """
=== Google Calendar authorisation required ===

Open this URL in your browser and approve access:

{url}

Waiting for the OAuth callback...
"""


def _callback_port() -> int:
    # Must sit inside the range published in compose.yml (18000-18015).
    return int(os.getenv("KARL_GOOGLE_OAUTH_PORT", "18010"))


def _google_credentials() -> Credentials:
    token_file = Path(os.getenv("KARL_GOOGLE_TOKEN_FILE", "token.json"))
    client_secrets_file = os.getenv("KARL_GOOGLE_CLIENT_SECRETS", "credentials.json")

    creds: Credentials | None = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
        creds = flow.run_local_server(
            host="localhost",  # goes into the redirect URI
            bind_addr="0.0.0.0" if in_container() else None,  # what we listen on
            port=_callback_port(),
            open_browser=not in_container(),
            authorization_prompt_message=_AUTH_PROMPT,
        )

    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json(), encoding="utf-8")
    token_file.chmod(0o600)
    return creds


async def get_google_calendar_tools():
    credentials = await asyncio.to_thread(_google_credentials)
    toolkit = CalendarToolkit(
        api_resource=build_calendar_service(credentials=credentials)
    )
    return toolkit.get_tools()

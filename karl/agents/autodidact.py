import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import parse_qs, urlparse

from langchain.agents import AgentState
from langchain.agents.middleware import (
    after_model,
    ToolErrorMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage, AnyMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from mcp.client.auth import TokenStorage, OAuthClientProvider
from mcp.shared.auth import OAuthToken, OAuthClientInformationFull, OAuthClientMetadata
from pydantic import AnyUrl
from ..linkedin.tools import (
    find_latest_non_replied_chat,
    find_past_reply_examples,
    save_draft_message,
)
from ..tools import http, search, cv
from ..obsidian.tools import (
    list_obsidian_vaults,
    list_obsidian_notes_opened_recently,
    search_obsidian_notes,
    read_obsidian_note,
    append_to_obsidian_note,
    view_obsidian_base,
    get_daily_note_path,
    append_to_daily_note,
    read_daily_note,
)
from ..obsidian.backends import ObsidianBackend
from ..gitlab.tools import (
    get_gitlab_merge_requests_created_by_user,
    get_gitlab_reviews_requested_for_user,
    get_gitlab_merge_requests_assigned_to_user,
    get_gitlab_merge_request_diff,
    list_gitlab_ci_pipelines,
    get_gitlab_ci_pipeline,
    get_gitlab_ci_job_log,
    get_gitlab_merge_request,
)
from ..jira.tools import (
    get_assigned_jira_tickets,
    get_specific_jira_ticket,
    get_jira_issue,
    get_jira_issue_field_value,
    update_jira_issue_fields,
    jira_issue_exists,
    assign_jira_issue,
    create_jira_issue,
    create_or_update_jira_issue,
    get_jira_issue_transitions,
    get_jira_issue_status_changelog,
    get_jira_status_id_from_name,
    get_jira_transition_id_to_status_name,
    transition_jira_issue,
    create_jira_issue_with_sdk_create_issue,
    get_jira_issue_comments,
    get_jira_issue_comment,
    get_jira_issue_changelog,
    get_jira_issue_property,
)
from ..confluence.tools import confluence
from ..email.tools import list_folders, search_emails, fetch_email, archive_email
from ..slack.tools import get_tools as get_slack_tools
from deepagents import create_deep_agent
from langchain_google_community import CalendarToolkit

toolkit = CalendarToolkit()


def _message_created_at(message: AnyMessage) -> datetime | None:
    """Return the message creation time if one is present in metadata."""
    value = (
        message.additional_kwargs.get("created_at")
        or message.response_metadata.get("created_at")
        or message.additional_kwargs.get("timestamp")
        or message.response_metadata.get("timestamp")
    )

    if value is None:
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    return None


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages older than 24 hours to keep context window focused on current session."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    messages: list[AnyMessage] = state["messages"]
    messages_to_remove = [
        RemoveMessage(id=m.id)
        for m in messages
        if m.id
        and (created_at := _message_created_at(m)) is not None
        and created_at < cutoff
    ]

    if messages_to_remove:
        return {"messages": messages_to_remove}

    return None


async def handle_redirect(auth_url: str) -> None:
    print(f"Visit: {auth_url}")


def _preferred_oauth_callback_port(service: str) -> int:
    base = int(os.getenv("KARL_OAUTH_CALLBACK_PORT_BASE", "18000"))
    span = int(os.getenv("KARL_OAUTH_CALLBACK_PORT_SPAN", "20000"))
    digest = hashlib.sha256(service.encode("utf-8")).hexdigest()
    return base + (int(digest[:8], 16) % span)


def _allocate_oauth_callback_ports(services: list[str]) -> dict[str, int]:
    used_ports: set[int] = set()
    ports: dict[str, int] = {}

    for service in sorted(services):
        port = _preferred_oauth_callback_port(service)

        while port in used_ports:
            port += 1

        used_ports.add(port)
        ports[service] = port

    return ports


def make_handle_callback(port: int):
    async def handle_callback() -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        callback_received: asyncio.Future[tuple[str, str | None]] = loop.create_future()

        async def handle_request(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            request_line = await reader.readline()
            request = request_line.decode("utf-8", errors="replace").strip()

            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            try:
                _method, target, _version = request.split(" ", 2)
                params = parse_qs(urlparse(target).query)
                code = params["code"][0]
                state = params.get("state", [None])[0]

                if not callback_received.done():
                    callback_received.set_result((code, state))

                body = "OAuth callback received. You can close this tab."
                status = "200 OK"
            except Exception as exc:
                if not callback_received.done():
                    callback_received.set_exception(exc)

                body = "OAuth callback failed. Check the terminal for details."
                status = "400 Bad Request"

            response = (
                f"HTTP/1.1 {status}\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                "Connection: close\r\n"
                "\r\n"
                f"{body}"
            )
            writer.write(response.encode("utf-8"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle_request, "localhost", port)

        async with server:
            return await callback_received

    return handle_callback


async def handle_callback() -> tuple[str, str | None]:
    loop = asyncio.get_running_loop()
    callback_received: asyncio.Future[tuple[str, str | None]] = loop.create_future()

    async def handle_request(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = await reader.readline()
        request = request_line.decode("utf-8", errors="replace").strip()

        while True:
            line = await reader.readline()
            if line in {b"\r\n", b"\n", b""}:
                break

        try:
            _method, target, _version = request.split(" ", 2)
            params = parse_qs(urlparse(target).query)
            code = params["code"][0]
            state = params.get("state", [None])[0]

            if not callback_received.done():
                callback_received.set_result((code, state))

            body = "OAuth callback received. You can close this tab."
            status = "200 OK"
        except Exception as exc:
            if not callback_received.done():
                callback_received.set_exception(exc)

            body = "OAuth callback failed. Check the terminal for details."
            status = "400 Bad Request"

        response = (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            "Connection: close\r\n"
            "\r\n"
            f"{body}"
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_request, "localhost", 8000)

    async with server:
        return await callback_received


class FileTokenStorage(TokenStorage):
    """Local JSON-file token storage for single-instance MCP OAuth clients."""

    def __init__(
        self,
        storage_key: str,
        token_file: Path | None = None,
    ):
        self.storage_key = storage_key
        self.token_file = token_file or Path(
            os.getenv("KARL_OAUTH_TOKEN_FILE", ".karl/oauth-tokens.json")
        )

    def _read_all(self) -> dict[str, Any]:
        if not self.token_file.exists():
            return {}

        with self.token_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_all(self, data: dict[str, Any]) -> None:
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

        tmp_file = self.token_file.with_suffix(f"{self.token_file.suffix}.tmp")
        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")

        os.replace(tmp_file, self.token_file)
        self.token_file.chmod(0o600)

    def _read_service_data(self) -> dict[str, Any]:
        return self._read_all().get(self.storage_key, {})

    def _write_service_data(self, service_data: dict[str, Any]) -> None:
        data = self._read_all()
        data[self.storage_key] = service_data
        self._write_all(data)

    async def get_tokens(self) -> OAuthToken | None:
        """Get stored tokens."""
        tokens = self._read_service_data().get("tokens")
        if tokens is None:
            return None

        return OAuthToken.model_validate(tokens)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        service_data = self._read_service_data()
        service_data["tokens"] = tokens.model_dump(mode="json")
        self._write_service_data(service_data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get stored client information."""
        client_info = self._read_service_data().get("client_info")
        if client_info is None:
            return None

        return OAuthClientInformationFull.model_validate(client_info)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        service_data = self._read_service_data()
        service_data["client_info"] = client_info.model_dump(mode="json")
        self._write_service_data(service_data)


class InMemoryTokenStorage(TokenStorage):
    """Demo In-memory token storage implementation."""

    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        """Get stored tokens."""
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get stored client information."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        self.client_info = client_info


mcp_config = json.loads(os.getenv("CUSTOM_MCP_URLS", "{}"))
oauth_services = [
    service for service, mcp in mcp_config.items() if mcp.get("auth", None) == "oauth"
]
oauth_callback_ports = _allocate_oauth_callback_ports(oauth_services)

for service, mcp in mcp_config.items():
    if mcp.get("auth", None) == "oauth":
        storage = FileTokenStorage(service)
        callback_port = oauth_callback_ports[service]
        redirect_uri = f"http://localhost:{callback_port}/callback"

        oauth_auth = OAuthClientProvider(
            server_url=mcp["url"],
            client_metadata=OAuthClientMetadata(
                client_name="Karl MCP Client",
                redirect_uris=[AnyUrl(redirect_uri)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="user",
            ),
            storage=storage,
            redirect_handler=handle_redirect,
            callback_handler=make_handle_callback(callback_port),
        )
        mcp["auth"] = oauth_auth

CUSTOM_MCP_TOOLS = MultiServerMCPClient(mcp_config, tool_name_prefix=True)

_CHECKPOINTER_CONTEXT = None
_CHECKPOINTER = None


async def get_checkpointer() -> AsyncSqliteSaver:
    global _CHECKPOINTER_CONTEXT, _CHECKPOINTER

    if _CHECKPOINTER is None:
        checkpoint_path = Path(
            os.getenv("KARL_CHECKPOINT_DB", ".karl/checkpoints.sqlite")
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        _CHECKPOINTER_CONTEXT = AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
        _CHECKPOINTER = await _CHECKPOINTER_CONTEXT.__aenter__()

    return _CHECKPOINTER


def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
    error_message = f"`{request.tool_call['name']}` failed with {type(exc).__name__}. Full info: {exc}"
    print(f"WARNING: Tool failure: {error_message}", file=sys.stderr)
    return error_message


async def create(model: BaseChatModel | str):
    checkpointer = await get_checkpointer()

    return create_deep_agent(
        model=model,
        tools=[
            list_folders,
            search_emails,
            fetch_email,
            archive_email,
            cv.fetch_cv,
            # list_todoist_projects,
            # list_todoist_tasks,
            find_latest_non_replied_chat,
            find_past_reply_examples,
            save_draft_message,
            cv.fetch_cv,
            get_gitlab_merge_requests_created_by_user,
            get_gitlab_reviews_requested_for_user,
            get_gitlab_merge_requests_assigned_to_user,
            get_gitlab_merge_request,
            get_gitlab_merge_request_diff,
            list_gitlab_ci_pipelines,
            get_gitlab_ci_pipeline,
            get_gitlab_ci_job_log,
            list_obsidian_vaults,
            list_obsidian_notes_opened_recently,
            search_obsidian_notes,
            read_obsidian_note,
            append_to_obsidian_note,
            view_obsidian_base,
            get_daily_note_path,
            read_daily_note,
            append_to_daily_note,
            search.web_search,
            http.fetch_url,
            # Jira
            get_assigned_jira_tickets,
            get_specific_jira_ticket,
            get_jira_issue,
            get_jira_issue_field_value,
            update_jira_issue_fields,
            jira_issue_exists,
            assign_jira_issue,
            create_jira_issue,
            create_or_update_jira_issue,
            get_jira_issue_transitions,
            get_jira_issue_status_changelog,
            get_jira_status_id_from_name,
            get_jira_transition_id_to_status_name,
            transition_jira_issue,
            create_jira_issue_with_sdk_create_issue,
            get_jira_issue_comments,
            get_jira_issue_comment,
            get_jira_issue_changelog,
            get_jira_issue_property,
        ]
        + await get_slack_tools()
        + await CUSTOM_MCP_TOOLS.get_tools()
        + toolkit.get_tools()
        + [
            StructuredTool.from_function(f)
            for f in [
                confluence.get_page_by_title,
                confluence.get_page_by_id,
                confluence.get_all_spaces,
                confluence.get_all_pages_from_space,
                confluence.get_child_pages,
                confluence.update_page,
                confluence.get_page_comments,
            ]
        ],
        system_prompt=dedent("""\
        You are a personal assistant helping a user with any task they need help with.
        All context and knowledge are persisted via filesystem tools which should be consulted for information on any task.
        The filesystem knowledge base should be maintained as you execute tasks.
        Previous AI agents will have left information for you and you should be mindful of future AI agents that won't have access to conversation history
        so after you learn anything new, update the knowledge base via the filesystem tools given.
        Also be eager in trimming useless or updating out of date information to keep the knowledge base growing with more noise than signal.
        Files in the knowledge base are markdown format using wikilinks to link notes. Yaml frontmatter can be used for metadata too.
        Note all tools have been built by the user who is capable of building more tools to need.
        If you encounter a task for which there is no clear tool to use, you may request new tools be built to expand your capability.
        """),
        middleware=[
            ToolErrorMiddleware(on_error),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "archive_email": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    # -------------------------------------------------------------------------
                    # Obsidian
                    # -------------------------------------------------------------------------
                    # "append_to_obsidian_note": {
                    #     "allowed_decisions": ["approve", "reject"],
                    # },
                    # "append_to_daily_note": {
                    #     "allowed_decisions": ["approve", "reject"],
                    # },
                    # -------------------------------------------------------------------------
                    # Native Jira tools
                    # -------------------------------------------------------------------------
                    "create_jira_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "create_jira_issue_with_sdk_create_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "create_or_update_jira_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "update_jira_issue_fields": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "assign_jira_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "transition_jira_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "update_content": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    # Todoist tasks
                    "todoist_complete-tasks": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_uncomplete-tasks": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    # -------------------------------------------------------------------------
                    # Todoist projects and sections
                    # -------------------------------------------------------------------------
                    "todoist_add-projects": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_update-projects": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_project-management": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_project-move": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_add-sections": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_update-sections": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    # -------------------------------------------------------------------------
                    # Todoist comments and reminders
                    # -------------------------------------------------------------------------
                    "todoist_add-comments": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_update-comments": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_add-reminders": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_update-reminders": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    # -------------------------------------------------------------------------
                    # Todoist labels and filters
                    # -------------------------------------------------------------------------
                    "todoist_add-labels": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_update-labels": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_add-filters": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_update-filters": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    # -------------------------------------------------------------------------
                    # Other Todoist mutations
                    # -------------------------------------------------------------------------
                    "todoist_analyze-project-health": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_delete-object": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_reorder-objects": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_manage-assignments": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "todoist_import-project-template": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                },
                description_prefix="The agent wants to call a tool that requires approval.",
            ),
            # FilesystemMiddleware(
            #     backend=ObsidianBackend(vault="AI Vault"),
            #     system_prompt=dedent("""\
            #     All context and knowledge are persisted via filesystem tools which should be consulted for information on any task.
            #     The filesystem knowledge base should be maintained as you execute tasks.
            #     Previous AI agents will have left information for you and you should be mindful of future AI agents that won't have access to conversation history
            #     so after you learn anything new, update the knowledge base via the filesystem tools given.
            #     Also be eager in trimming useless or updating out of date information to keep the knowledge base growing with more noise than signal.
            #     Files in the knowledge base are markdown format using wikilinks to link notes. Yaml frontmatter can be used for metadata too.
            #     """)
            # ),
            # ToolRetryMiddleware(
            #     max_retries=5,
            #     backoff_factor=2.0,
            #     initial_delay=2.0,
            # ),
            # ContextEditingMiddleware(
            #     edits=[
            #         ClearToolUsesEdit(
            #             trigger=20000,
            #             keep=2,
            #         ),
            #     ],
            # ),
        ],
        checkpointer=checkpointer,
        backend=ObsidianBackend(vault="AI Vault"),
        memory=["/AGENTS.md"],
        skills=["/skills/"],
    )

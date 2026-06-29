from datetime import datetime, timedelta, timezone
from textwrap import dedent

from langchain.agents import AgentState
from langchain.agents.middleware import (
    after_model,
)
from langchain_core.messages import RemoveMessage, AnyMessage
from langgraph.runtime import Runtime
from ..tools import http, search, cv
from ..obsidian.tools import (
    list_obsidian_vaults,
    list_obsdian_notes_opened_recently,
    search_obsidian_notes,
    read_obsidian_note,
    append_to_obsidian_note,
)
from ..obsidian.backends import ObsidianBackend
from ..gitlab.tools import (
    get_gitlab_merge_requests_created_by_user,
    get_gitlab_reviews_requested_for_user,
    get_gitlab_merge_requests_assigned_to_user,
)
from ..jira.tools import get_assigned_jira_tickets, get_specific_jira_ticket
from ..email.tools import list_folders, search_emails, fetch_email
from ..slack.tools import get_tools as get_slack_tools
from ..todoist.tools import list_todoist_projects, list_todoist_tasks
from deepagents import create_deep_agent


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


async def create(model):
    return create_deep_agent(
        model=model,
        tools=[
            list_folders,
            search_emails,
            fetch_email,
            cv.fetch_cv,
            get_assigned_jira_tickets,
            get_specific_jira_ticket,
            list_todoist_projects,
            list_todoist_tasks,
            get_gitlab_merge_requests_created_by_user,
            get_gitlab_reviews_requested_for_user,
            get_gitlab_merge_requests_assigned_to_user,
            list_obsidian_vaults,
            list_obsdian_notes_opened_recently,
            search_obsidian_notes,
            read_obsidian_note,
            append_to_obsidian_note,
            search.web_search,
            http.fetch_url,
        ]
        + await get_slack_tools(),
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
            delete_old_messages,
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
        backend=ObsidianBackend(vault="AI Vault"),
        memory=["/AGENTS.md"],
        skills=["/skills/"],
    )

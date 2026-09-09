import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

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


CUSTOM_MCP_TOOLS = MultiServerMCPClient(json.loads(os.getenv("CUSTOM_MCP_URLS", "{}")))

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
    error_message = f"`{request.tool_call['name']}` failed with {type(exc).__name__}."
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
            cv.fetch_cv,
            list_todoist_projects,
            list_todoist_tasks,
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
        + [
            StructuredTool.from_function(f)
            for f in [
                confluence.get_page_by_title,
                confluence.get_page_id_by_url,
                confluence.get_all_spaces,
                confluence.get_all_pages_from_space,
                confluence.get_comments,
                confluence.get_child_pages,
                confluence.update_content,
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
                    "append_to_obsidian_note": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "append_to_daily_note": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "create_jira_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "create_or_update_jira_issue": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                    "update_content": {
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

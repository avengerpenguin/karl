import os
import sys
from pathlib import Path
from textwrap import dedent

import tiktoken

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware import (
    ToolErrorMiddleware,
    HumanInTheLoopMiddleware,
    wrap_tool_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from ..confluence.tools import confluence
from ..email.tools import list_folders, search_emails, fetch_email, archive_email
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
from ..google.adsense import get_google_adsense_tools
from ..google.calendar import get_google_calendar_tools
from ..google.search_console import get_google_search_console_tools
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
from ..linkedin.tools import (
    find_latest_non_replied_chat,
    find_past_reply_examples,
    save_draft_message,
)
from ..slack.tools import get_tools as get_slack_tools
from ..sql.tools import get_database_tools
from ..tools import http, search, cv
from ..tools.custom_mcp import get_custom_mcp_tools


def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
    error_message = f"`{request.tool_call['name']}` failed with {type(exc).__name__}. Full info: {exc}"
    print(f"WARNING: Tool failure: {error_message}", file=sys.stderr)
    return error_message


enc = tiktoken.get_encoding("cl100k_base")


@wrap_tool_call
async def track_tool_payloads(request: ToolCallRequest, handler):
    tool_name = request.tool.name

    # 1. Await the async tool handler execution
    result = await handler(request)

    # 2. Measure the token payload size
    result_str = str(result)
    token_count = len(enc.encode(result_str))

    # 3. Alert if a tool returns a massive payload (> 10,000 tokens)
    if token_count > 10000:
        print(
            f"\n⚠️ [CRITICAL TOOL FLOOD] Tool '{tool_name}' just dumped {token_count} tokens into context!"
        )
        print(f"Snippet: {result_str[:20]}...\n")
    else:
        print(f"[Tool Log] {tool_name}: {token_count} tokens.")

    return result


_CHECKPOINTER_CONTEXT = None
_CHECKPOINTER = None


async def get_checkpointer() -> AsyncSqliteSaver:
    global _CHECKPOINTER_CONTEXT, _CHECKPOINTER

    if _CHECKPOINTER is None:
        checkpoint_path = Path(
            os.getenv("KARL_CHECKPOINT_DB", ".karl/checkpoints-general.sqlite")
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        _CHECKPOINTER_CONTEXT = AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
        _CHECKPOINTER = await _CHECKPOINTER_CONTEXT.__aenter__()

    return _CHECKPOINTER


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
        + await get_custom_mcp_tools()
        + await get_google_calendar_tools()
        + get_google_search_console_tools()
        + get_google_adsense_tools()
        + await get_database_tools()
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
            track_tool_payloads,
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
        ],
        checkpointer=checkpointer,
        backend=LocalShellBackend(root_dir="/workspace"),
        memory=["/AGENTS.md"],
        skills=["/skills/"],
    )

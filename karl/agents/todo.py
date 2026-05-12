from textwrap import dedent

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    SummarizationMiddleware,
)
from ..tools import http
from ..obsidian.tools import (
    list_obsidian_vaults,
    list_obsdian_notes_opened_recently,
    search_obsidian_notes,
    read_obsidian_note,
    append_to_obsidian_note,
)
from ..gitlab.tools import (
    get_gitlab_merge_requests_created_by_user,
    get_gitlab_reviews_requested_for_user,
    get_gitlab_merge_requests_assigned_to_user,
)
from ..jira.tools import get_assigned_jira_tickets, get_specific_jira_ticket
from ..email.tools import list_folders, search_emails, fetch_email
from ..tools.cv import fetch_cv
from ..slack.tools import get_tools as get_slack_tools
from ..todoist.tools import list_todoist_projects, list_todoist_tasks


async def create(model):
    return create_agent(
        model,
        [
            list_folders,
            search_emails,
            fetch_email,
            fetch_cv,
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
            http.fetch_url,
        ]
        + await get_slack_tools(),
        system_prompt=dedent("""\
        You are a personal assistant who helps users check their various inboxes and plan their day.

        You have access to all relevant inboxes on services the user uses for work.
        """),
        middleware=[
            ToolRetryMiddleware(
                max_retries=5,
                backoff_factor=2.0,
                initial_delay=2.0,
            ),
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=50000,
                        keep=5,
                    ),
                ],
            ),
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 100000),
                keep=("tokens", 50000),
            ),
        ],
    )

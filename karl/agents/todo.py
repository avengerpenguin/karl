from textwrap import dedent

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

from ..tools.jira import get_assigned_jira_tickets
from ..tools.email_tools import list_folders, search_emails, fetch_email
from ..tools.cv import fetch_cv
from ..tools.slack import get_tools as get_slack_tools
from ..tools.todoist import list_todoist_projects, list_todoist_tasks


async def create(model):
    return create_agent(
        model,
        [
            list_folders,
            search_emails,
            fetch_email,
            fetch_cv,
            get_assigned_jira_tickets,
            list_todoist_projects,
            list_todoist_tasks,
        ] + await get_slack_tools(),
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
        ],
    )

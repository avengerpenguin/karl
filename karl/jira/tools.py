import os

import requests
from langchain_core.tools import tool
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

try:
    from jira import JIRA, Issue
except ImportError:
    raise ImportError("Please install karl[jira] to use Jira tools")

http = requests.Session()


class JiraTicket(BaseModel):
    key: str
    status: str
    sprint: str
    summary: str
    url: str
    priority: str
    description: str | None = None


@tool
def get_assigned_jira_tickets() -> list[JiraTicket]:
    """Get all Jira tickets assigned to the current user."""
    response = http.get(
        "https://roku.atlassian.net/rest/api/3/search/jql",
        params={
            "jql": "assignee = currentUser() AND statusCategory != Done",
            "fields": "summary,key,status,customfield_10020,description,priority",
        },
        auth=HTTPBasicAuth(
            os.getenv("ATLASSION_USER", ""), os.getenv("ATLASSIAN_API_TOKEN", "")
        ),
    )

    response.raise_for_status()

    def ticket_generator():
        for issue in sorted(
            response.json().get("issues", []),
            key=lambda x: x["fields"]["status"]["id"],
            reverse=True,
        ):
            sprint = (
                ",".join(s["name"] for s in issue["fields"]["customfield_10020"])
                if issue["fields"]["customfield_10020"]
                else "No Sprint"
            )

            description_data = issue["fields"].get("description")
            description_str = str(description_data) if description_data else None

            yield JiraTicket(
                key=issue["key"],
                status=issue["fields"]["status"]["name"],
                sprint=sprint,
                summary=issue["fields"]["summary"],
                url=f"https://roku.atlassian.net/browse/{issue['key']}",
                priority=issue["fields"]["priority"]["name"],
                description=description_str,
            )

    return list(ticket_generator())


@tool
def get_specific_jira_ticket(ticket_ref: str) -> JiraTicket:
    """Retrieve a specific Jira ticket by its reference."""
    jira = JIRA(
        server="https://roku.atlassian.net",
        basic_auth=(os.environ["ATLASSION_USER"], os.environ["ATLASSIAN_API_TOKEN"]),
    )
    issue: Issue = jira.issue(ticket_ref)
    return JiraTicket(
        key=issue.key,
        status=issue.fields.status.name,
        sprint=",".join(s["name"] for s in issue.fields.customfield_10020)
        if issue.fields.customfield_10020
        else "No Sprint",
        summary=issue.fields.summary,
        url=issue.permalink(),
        priority=issue.fields.priority.name,
        description=issue.fields.description,
    )

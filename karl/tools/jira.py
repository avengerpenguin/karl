import os
from typing import Iterable

import requests
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

http = requests.Session()

class JiraTicket(BaseModel):
    key: str
    status: str
    sprint: str
    summary: str
    url: str
    priority: str
    description: str | None = None

def get_assigned_jira_tickets() -> list[JiraTicket]:
    """Get all Jira tickets assigned to the current user."""
    response = http.get(
        "https://roku.atlassian.net/rest/api/3/search/jql",
        params={
            'jql': 'assignee = currentUser() AND statusCategory != Done',
            'fields': 'summary,key,status,customfield_10020,description,priority'
        },
        auth=HTTPBasicAuth("rfenning@roku.com", os.getenv('ATLASSIAN_API_TOKEN')),
    )

    response.raise_for_status()

    def ticket_generator():
        for issue in sorted(response.json().get("issues", []), key=lambda x: x['fields']['status']['id'], reverse=True):
            sprint = ",".join(s["name"] for s in issue['fields']['customfield_10020']) if issue['fields'][
                'customfield_10020'] else "No Sprint"

            description_data = issue['fields'].get('description')
            description_str = str(description_data) if description_data else None

            yield JiraTicket(
                key=issue['key'],
                status=issue['fields']['status']['name'],
                sprint=sprint,
                summary=issue['fields']['summary'],
                url=f"https://roku.atlassian.net/browse/{issue['key']}",
                priority=issue['fields']['priority']['name'],
                description=description_str,
            )

    return list(ticket_generator())

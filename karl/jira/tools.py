import os
from typing import Any

import requests
from langchain_core.tools import tool
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

try:
    from jira import JIRA, Issue
    from atlassian import Jira
except ImportError:
    raise ImportError("Please install karl[jira] to use Jira tools")

http = requests.Session()


jira = Jira(
    url=os.getenv("ATLASSIAN_BASE_URL"),
    username=os.getenv("ATLASSIAN_USER"),
    password=os.getenv("ATLASSIAN_API_TOKEN"),
    session=http,
)


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
        f"{os.getenv('ATLASSIAN_BASE_URL')}/rest/api/3/search/jql",
        params={
            "jql": "assignee = currentUser() AND statusCategory != Done",
            "fields": "summary,key,status,customfield_10020,description,priority",
        },
        auth=HTTPBasicAuth(
            os.getenv("ATLASSIAN_USER", ""), os.getenv("ATLASSIAN_API_TOKEN", "")
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
                url=f"{os.getenv('ATLASSIAN_BASE_URL')}/browse/{issue['key']}",
                priority=issue["fields"]["priority"]["name"],
                description=description_str,
            )

    return list(ticket_generator())


@tool
def get_specific_jira_ticket(ticket_ref: str) -> JiraTicket:
    """Retrieve a specific Jira ticket by its reference."""
    jira = JIRA(
        server=os.getenv("ATLASSIAN_BASE_URL"),
        basic_auth=(os.environ["ATLASSIAN_USER"], os.environ["ATLASSIAN_API_TOKEN"]),
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


@tool
def get_jira_issue(
    issue_key: str,
    fields: str = "*all",
    expand: str | None = None,
) -> dict[str, Any] | None:
    """
    Retrieve a Jira issue by issue key or id.

    Use this when you need the raw Jira issue payload, including fields that are
    not represented by the simpler get_specific_jira_ticket tool.

    issue_key should be a Jira issue key such as "ABC-123" or a Jira issue id.
    fields controls which Jira fields are returned. Use "*all" for all fields,
    or a comma-separated list such as "summary,status,assignee,priority".
    expand can request additional Jira expansions such as "changelog" when needed.
    """
    return jira.issue(issue_key, fields=fields, expand=expand)


@tool
def get_jira_issue_field_value(issue_key: str, field: str) -> Any | None:
    """
    Retrieve a single field value from a Jira issue.

    Use this when you only need one field from an issue instead of the complete
    issue payload.

    issue_key should be a Jira issue key such as "ABC-123".
    field should be the Jira field name or id to read, such as "summary",
    "status", "assignee", "description", or a custom field id like
    "customfield_10020".
    """
    return jira.issue_field_value(issue_key, field)


@tool
def update_jira_issue_fields(
    issue_key: str,
    fields: dict[str, Any],
    notify_users: bool = True,
) -> dict[str, Any] | None:
    """
    Update one or more fields on an existing Jira issue.

    This changes Jira state. Use it only when the user has clearly requested an
    update and the target issue and field values are known.

    issue_key should be a Jira issue key such as "ABC-123".
    fields should map Jira field names or ids to their new values, for example
    {"summary": "New summary"} or {"customfield_10020": [{"id": 123}]}.
    notify_users controls whether Jira should send notifications according to
    the project's notification scheme.
    """
    return jira.update_issue_field(
        issue_key,
        fields=fields,
        notify_users=notify_users,
    )


@tool
def jira_issue_exists(issue_key: str) -> bool:
    """
    Check whether a Jira issue exists.

    Use this before reading, updating, assigning, or transitioning an issue when
    there is uncertainty about whether the issue key is valid.

    issue_key should be a Jira issue key such as "ABC-123".
    """
    return jira.issue_exists(issue_key)


@tool
def assign_jira_issue(issue_key: str, account_id: str) -> dict[str, Any] | None:
    """
    Assign a Jira issue to a user.

    This changes Jira state. Use it only when the user has clearly requested an
    assignment or reassignment.

    issue_key should be a Jira issue key such as "ABC-123".
    account_id should be the Jira Cloud account id of the user to assign the
    issue to.
    """
    return jira.assign_issue(issue_key, account_id)


@tool
def create_jira_issue(fields: dict[str, Any]) -> dict[str, Any] | None:
    """
    Create a new Jira issue.

    This changes Jira state. Use it only when the user has explicitly asked to
    create a Jira issue and the required fields are known.

    fields should be the Jira issue fields payload. It typically includes
    project, issuetype, summary, and optionally description, priority,
    assignee, labels, or custom fields.

    Example shape:
    {
        "project": {"key": "ABC"},
        "issuetype": {"name": "Task"},
        "summary": "Implement the thing",
        "description": "Details of the work to do"
    }
    """
    return jira.issue_create(fields)


@tool
def create_or_update_jira_issue(fields: dict[str, Any]) -> dict[str, Any] | None:
    """
    Create or update a Jira issue using the Atlassian SDK's combined operation.

    This changes Jira state. Prefer create_jira_issue for ordinary issue
    creation and update_jira_issue_fields for ordinary issue updates. Use this
    tool only when the caller specifically needs the SDK create-or-update
    behavior.

    fields should be the Jira fields payload expected by the Atlassian SDK.
    """
    return jira.issue_create_or_update(fields)


@tool
def get_jira_issue_transitions(
    issue_key: str,
    transition_id: str | None = None,
    expand: str | None = None,
) -> dict[str, Any] | None:
    """
    List available workflow transitions for a Jira issue.

    Use this before transitioning an issue when you need to know which
    transitions are currently available.

    issue_key should be a Jira issue key such as "ABC-123".
    transition_id can be provided to retrieve details for a specific transition.
    expand can request additional transition metadata when needed.
    """
    return jira.get_issue_transitions(
        issue_key,
        transition_id=transition_id,
        expand=expand,
    )


@tool
def get_jira_issue_status_changelog(issue_id: str) -> list[dict[str, Any]] | None:
    """
    Retrieve the status-change history for a Jira issue.

    Use this to answer questions about when an issue moved between workflow
    statuses, who changed the status, or how long it spent in previous states.

    issue_id should be the Jira issue id expected by the Atlassian SDK. If you
    only have an issue key, retrieve the issue first and use its id if needed.
    """
    return jira.get_issue_status_changelog(issue_id)


@tool
def get_jira_status_id_from_name(status_name: str) -> str | None:
    """
    Look up a Jira status id from a human-readable status name.

    Use this when an operation needs a Jira status id but the user has provided
    a status name such as "In Progress", "Done", or "Blocked".
    """
    return jira.get_status_id_from_name(status_name)


@tool
def get_jira_transition_id_to_status_name(
    issue_key: str,
    status_name: str,
) -> str | None:
    """
    Find the transition id that would move an issue to the named status.

    Use this before transitioning an issue when the user has described the
    target status by name rather than giving a transition id.

    issue_key should be a Jira issue key such as "ABC-123".
    status_name should be the desired target status, such as "In Progress",
    "Done", or "Blocked".
    """
    return jira.get_transition_id_to_status_name(issue_key, status_name)


@tool
def transition_jira_issue(
    issue_key: str,
    transition_id: str,
    fields: dict[str, Any] | None = None,
    update: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Transition a Jira issue through its workflow.

    This changes Jira state. Use it only when the user has clearly requested a
    status change and the correct transition id is known.

    issue_key should be a Jira issue key such as "ABC-123".
    transition_id should be a valid transition id for the issue's current
    workflow state. If only the target status name is known, first call
    get_jira_transition_id_to_status_name.
    fields can provide fields required by the transition screen.
    update can provide Jira update operations to apply during the transition.
    """
    return jira.issue_transition(
        issue_key,
        transition_id,
        fields=fields,
        update=update,
    )


@tool
def create_jira_issue_with_sdk_create_issue(
    fields: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Create a Jira issue using the Atlassian SDK create_issue method.

    This changes Jira state. Prefer create_jira_issue unless you specifically
    want to use the SDK's create_issue method rather than issue_create.

    fields should be the Jira issue fields payload. It typically includes
    project, issuetype, summary, and optionally description, priority,
    assignee, labels, or custom fields.
    """
    return jira.create_issue(fields)


@tool
def get_jira_issue_comments(issue_key: str) -> dict[str, Any] | None:
    """
    Retrieve comments for a Jira issue.

    Use this when the user asks for discussion, notes, updates, or comments on
    a Jira ticket.

    issue_key should be a Jira issue key such as "ABC-123".
    """
    return jira.issue_get_comments(issue_key)


@tool
def get_jira_issue_comment(issue_key: str, comment_id: str) -> dict[str, Any] | None:
    """
    Retrieve a single comment from a Jira issue.

    Use this when a specific Jira comment id is known and the user needs that
    comment's body, author, timestamps, or metadata.

    issue_key should be a Jira issue key such as "ABC-123".
    comment_id should be the Jira comment id.
    """
    return jira.issue_get_comment(issue_key, comment_id)


@tool
def get_jira_issue_changelog(issue_key: str) -> dict[str, Any] | None:
    """
    Retrieve the changelog for a Jira issue.

    Use this to answer questions about the history of edits to an issue,
    including changes to status, assignee, priority, summary, description, or
    other fields.

    issue_key should be a Jira issue key such as "ABC-123".
    """
    return jira.get_issue_changelog(issue_key)


@tool
def get_jira_issue_property(issue_key: str, property_key: str) -> dict[str, Any] | None:
    """
    Retrieve an entity property from a Jira issue.

    Use this only when you need a specific Jira issue property by key. This is
    different from ordinary issue fields such as summary, status, or assignee.

    issue_key should be a Jira issue key such as "ABC-123".
    property_key should be the Jira issue property key to retrieve.
    """
    return jira.get_issue_property(issue_key, property_key)

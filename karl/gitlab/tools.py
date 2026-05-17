import os

from gitlab.v4.objects import MergeRequest
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Literal

try:
    import gitlab
except ImportError:
    raise ImportError("Please install karl[gitlab] to use GitLab tools")


gl = gitlab.Gitlab(
    url=os.getenv("GITLAB_URL", "https://gitlab.com"),
    private_token=os.getenv("GITLAB_TOKEN"),
)


class GitlabApprovalStatus(BaseModel):
    state: Literal["approved", "awaiting_approval"]
    approved: bool
    approvals_required: int | None = None
    approvals_left: int | None = None
    approved_by: list[str] = []


class GitlabMergeRequest(BaseModel):
    mr_id: int
    mr_iid: int
    web_url: str
    title: str
    description: str
    state: str


@tool
def get_gitlab_merge_requests_created_by_user():
    """Get Merge Requests created by the current user"""
    mrs: list[MergeRequest] = gl.mergerequests.list(
        scope="created_by_me", state="opened"
    )
    return [
        GitlabMergeRequest(
            mr_id=mr.id,
            mr_iid=mr.iid,
            web_url=mr.web_url,
            title=mr.title,
            description=mr.description,
            state=mr.state,
        )
        for mr in mrs
    ]


@tool
def get_gitlab_merge_requests_assigned_to_user():
    """Get Merge Requests assigned to the current user"""
    mrs: list[MergeRequest] = gl.mergerequests.list(
        scope="assigned_to_me", state="opened"
    )
    return [
        GitlabMergeRequest(
            mr_id=mr.id,
            mr_iid=mr.iid,
            web_url=mr.web_url,
            title=mr.title,
            description=mr.description,
            state=mr.state,
        )
        for mr in mrs
    ]


@tool
def get_gitlab_reviews_requested_for_user():
    """Get Merge Requests where the current user is assigned as a reviewers"""
    mrs = gl.mergerequests.list(scope="reviews_for_me", state="opened")
    return [
        GitlabMergeRequest(
            mr_id=mr.id,
            mr_iid=mr.iid,
            web_url=mr.web_url,
            title=mr.title,
            description=mr.description,
            state=mr.state,
        )
        for mr in mrs
    ]

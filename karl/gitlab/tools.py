import os
import subprocess

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


class CommandResult(BaseModel):
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False
    stdout_bytes: int = 0


def truncate_text(text: str, max_chars: int = 100_000) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    head_size = max_chars // 2
    tail_size = max_chars - head_size

    truncated = (
        text[:head_size] + "\n\n...[output truncated]...\n\n" + text[-tail_size:]
    )
    return truncated, True


@tool
def get_gitlab_merge_request(
    repo: str, merge_request_id: str | int, comments: bool = False
) -> CommandResult:
    """
    Get details about a Gitlab merge request.

    """
    command = f"mr view {merge_request_id} --repo {repo}"
    if comments:
        command += " --comments"
    return _cli(command)


@tool
def get_gitlab_merge_request_diff(
    repo: str, merge_request_id: str | int
) -> CommandResult:
    """
    See a diff of a merge request.
    Set repo to OWNER/REPO or GROUP/NAMESPACE/REPO, The full URL or Git URL is also accepted.
    Set merge_request_id to the ID of the merge request, i.e. the number at the end of full URLs.
    """
    return _cli(f"mr diff {merge_request_id} --repo {repo}")


@tool
def list_gitlab_ci_pipelines(repo: str, ref: str) -> CommandResult:
    """
    List CI/CD pipelines.
    Set repo to OWNER/REPO or GROUP/NAMESPACE/REPO, The full URL or Git URL is also accepted.
    Set ref to a valid Gitlab ref value, i.e. branch, tag or MR ID in the form refs/merge-requests/123/head
    """
    return _cli(f"ci list --repo='{repo}' --ref='{ref}'")


@tool
def get_gitlab_ci_pipeline(repo: str, pipeline_id: str | int) -> CommandResult:
    """
    Get the details of a CI/CD pipeline.
    Set repo to OWNER/REPO or GROUP/NAMESPACE/REPO, The full URL or Git URL is also accepted.
    Set pipeline_id to a valid ID number. Use list_gitlab_ci_pipelines to get most recent for a given ref.
    """
    return _cli(f"ci get --repo='{repo}' --pipeline-id='{pipeline_id}'")


@tool
def get_gitlab_ci_job_log(
    repo: str, pipeline_id: str | int, job_id_or_name: str | int, branch: str
) -> CommandResult:
    """
    See job output for a given job and pipeline.
    Set repo to OWNER/REPO or GROUP/NAMESPACE/REPO, The full URL or Git URL is also accepted.
    Set job_id_or_name to a valid Gitlab job ID or name e.g. "224356863" or "lint".
    Use get_gitlab_ci_pipeline to get job names within a given pipeline first.
    Set branch to a valid branch or tag or MR ID in the form refs/merge-requests/123/head
    """
    return _cli(
        f"ci trace '{job_id_or_name}' --pipeline-id='{pipeline_id}' --repo='{repo}' --branch='{branch}'"
    )


def _cli(command: str) -> CommandResult:
    full_commnd = f"glab {command}"
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            full_commnd, shell=True, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired as e:
        return CommandResult(
            command=full_commnd,
            exit_code=None,
            timed_out=True,
            stdout=str(e.stdout),
            stderr="Timeout talking to Gitlab\n\n" + str(e.stderr),
            stdout_bytes=len(str(e.stdout)),
        )

    stdout, truncated = truncate_text(result.stdout)
    return CommandResult(
        command=full_commnd,
        exit_code=result.returncode,
        stdout=stdout,
        stderr=result.stderr,
        stdout_bytes=len(result.stdout),
        truncated=truncated,
    )

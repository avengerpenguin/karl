import os
from datetime import datetime
from typing import Iterator

from langchain.tools import tool
from pydantic import BaseModel

try:
    from todoist_api_python.api import TodoistAPI, Task
except ImportError:
    raise ImportError("Please install karl[todoist] to use Todoist tools")


class TodoistProject(BaseModel):
    project_id: str
    name: str


class TodoistTask(BaseModel):
    task_id: str
    project_id: str
    section_id: str
    content: str
    created_date: datetime | None = None


class TodoistSection(BaseModel):
    section_id: str
    project_id: str
    name: str


@tool
def list_todoist_projects() -> list[TodoistProject]:
    """
    List the projects in the user's Todoist account. Returns the project name and also ID for further lookups.
    """
    api = TodoistAPI(os.getenv("TODOIST_TOKEN"))
    projects_iterator = api.get_projects()
    for p in projects_iterator:
        return [
            TodoistProject(project_id=project.id, name=project.name) for project in p
        ]
    return []


@tool
def list_todoist_sections_within_a_project(project_id: str) -> list[TodoistSection]:
    """
    List the sections within a project identified by project ID. Returns the section name and also ID for further lookups.
    """
    api = TodoistAPI(os.getenv("TODOIST_TOKEN"))
    sections_iterator = api.get_sections(project_id=project_id)
    for s in sections_iterator:
        return [
            TodoistSection(
                section_id=section.id, project_id=section.project_id, name=section.name
            )
            for section in s
        ]
    return []


@tool
def list_todoist_tasks(project_id: str) -> list[TodoistTask]:
    """
    List the tasks in the given project identified by project ID. Returns the task content and also ID for further lookups.
    """
    api = TodoistAPI(os.getenv("TODOIST_TOKEN"))
    tasks_iterator: Iterator[list[Task]] = api.get_tasks(project_id=project_id)
    for tasks in tasks_iterator:
        return [
            TodoistTask(
                task_id=task.id,
                project_id=task.project_id,
                content=task.content,
                created_date=task.created_at,
                section_id=task.section_id,
            )
            for task in tasks
        ]
    return []

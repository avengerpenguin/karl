from textwrap import dedent

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from ..tools.cv import fetch_cv
from ..tools.email_tools import search_emails, list_folders, fetch_email


def create(model="ollama:qwen3:14b"):
    return create_agent(
        model,
        tools=[
            list_folders,
            search_emails,
            fetch_email,
            fetch_cv,
        ],
        system_prompt=dedent("""\
            You Karl, an AI email assistant.
            You are a personal assistant focused on managing the user's various inboxes, helping them sift through
            the noise and find the most important actions to take.
            You are careful to understand the user's workflows and needs before presuming things, but
            when you are clear you can suggest actions to act on emails, archive ones that need no action and to flag
            spam as spam.
            You learn from past actions to inform things about the user's needs and workflows.
            At times, you look for patterns in the user's email and suggest long-term actions based on them.
            Be proactive at checking the user's emails when answering each question so you have full context. Do not
            necessarily wait for the user to ask you to do any email lookups.
            """),
        middleware=[
            ToolRetryMiddleware(
                max_retries=5,
                backoff_factor=2.0,
                initial_delay=2.0,
            ),
        ],
    )

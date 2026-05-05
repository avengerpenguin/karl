from textwrap import dedent

import os
import yaml
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, messages_from_dict, SystemMessage, HumanMessage, messages_to_dict
from langchain_core.runnables import Runnable
from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware, ClearToolUsesEdit
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from .tools.cv import fetch_cv
from .tools.email_tools import search_emails, list_folders, fetch_email

tools = [
    list_folders,
    search_emails,
    fetch_email,
    fetch_cv,
]

llm = ChatOllama(model="qwen3:14b")

agent: Runnable = create_agent(
    llm,
    tools=tools,
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,
                    keep=3,
                ),
            ],
        ),
        SummarizationMiddleware(
            model="ollama:qwen3.5:9b",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)



async def run(message: str):
    console = Console()

    memory_file = os.path.join(os.getcwd(), "memory.yaml")
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            messages: list[BaseMessage] = messages_from_dict(yaml.load(f, Loader=yaml.FullLoader) or [])
    else:
        messages: list[BaseMessage] = [
            SystemMessage(content=dedent("""\
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
            """)),
        ]

    new_message = message

    while new_message:
        console.print(
            Panel(Markdown(new_message), title="You", title_align="left", border_style="green")
        )

        messages.append(HumanMessage(content=new_message))

        async for chunk in agent.astream({"messages": messages}, stream_mode="updates", version="v2"):
            if chunk["type"] == "updates":
                for step, data in chunk["data"].items():
                    panel_title, panel_colour = _create_panel_title(step)

                    if data:
                        messages.append(data['messages'][-1])
                        blocks = data['messages'][-1].content_blocks
                        for block in blocks:
                            text = _generate_panel_text(step, block)
                            markdown = Markdown(text)
                            console.print(
                                Panel(markdown, title=panel_title, title_align="left", border_style=panel_colour)
                            )


        with open(memory_file, "w") as f:
            yaml.dump(messages_to_dict(messages), f)

        new_message = input("You: ")
        print("\x1b[1A\x1b[2K", end="")


def _create_panel_title(step):
    return {
        "model": ("Karl", "red"),
        "tools": ("Data", "white"),
        "SummarizationMiddleware.before_model": ("Karl's inner monologue", "yellow"),
    }.get(step, (f"Unknown: {step}", "gray"))


def _generate_panel_text(step: str, block: dict) -> str:
    if step == "tools":
        return block["text"] if len(block["text"]) < 100 else block["text"][:100] + "..."

    if step == "SummarizationMiddleware.before_model":
        return block["text"] if len(block["text"]) < 100 else block["text"][:100] + "..."

    if 'text' in block:
        return block['text']

    if block['type'] == "tool_call":
        return f"Calling `{block['name']}` with arguments: {block.get('args', '')}"

    return f"Unknown: {block}"

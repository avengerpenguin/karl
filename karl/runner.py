import os

import yaml
from langchain_core.messages import BaseMessage, messages_from_dict, HumanMessage, messages_to_dict
from langchain_core.runnables import Runnable
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


async def run(agent: Runnable, message: str, memory_path: str = "memory.yaml"):
    console = Console()

    memory_file = os.path.join(os.getcwd(), memory_path)
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            messages: list[BaseMessage] = messages_from_dict(yaml.load(f, Loader=yaml.FullLoader) or [])
    else:
        messages: list[BaseMessage] = []

    new_message = message

    while new_message:
        console.print(
            Panel(Markdown(new_message), title="You", title_align="left", border_style="green")
        )

        messages.append(HumanMessage(content=new_message))

        async for chunk in agent.astream(
                {"messages": messages},
                stream_mode="updates",
                version="v2",
        ):
            if chunk["type"] == "updates":
                for step, data in chunk["data"].items():
                    panel_title, panel_colour = _create_panel_title(step)

                    if data and 'messages' in data:
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
    }.get(step, (f"Unknown: {step}", "grey0"))


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

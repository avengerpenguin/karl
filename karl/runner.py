import os
import subprocess
import asyncio
import yaml
from typing import AsyncIterator
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    HumanMessage,
    messages_to_dict,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.schema import StreamEvent
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


async def run(agent: Runnable, message: str, memory_path: str = "memory.yaml"):
    console = Console()

    memory_file = os.path.join(os.getcwd(), memory_path)
    working_memory_file = memory_file.replace(".yaml", ".working.yaml")
    if os.path.exists(memory_file):
        with open(memory_file) as f:
            messages: list[BaseMessage] = messages_from_dict(
                yaml.load(f, Loader=yaml.FullLoader) or []
            )
    else:
        messages: list[BaseMessage] = []

    new_message = message

    while new_message:
        console.print(
            Panel(
                Markdown(new_message),
                title="You",
                title_align="left",
                border_style="green",
            )
        )

        messages.append(HumanMessage(content=new_message))

        stream: AsyncIterator[StreamEvent] = await agent.astream_events(
            dict(messages=messages), version="v3"
        )

        async def consume_messages():
            async for message in stream.messages:
                async for delta in message.reasoning:
                    print(f"[thinking] {delta}", end="", flush=True)

                async for chunk in message.tool_calls:
                    tool_name = chunk["name"]
                    tool_args = chunk["args"]
                    console.print(
                        Panel(
                            str(tool_args)[:200]
                            if len(str(tool_args)) > 200
                            else str(tool_args),
                            title=tool_name + ">",
                            title_align="left",
                            border_style="white",
                        )
                    )

                text = ""
                block = "█ "
                with Live(
                    console=console,
                ) as live:
                    title, colour = _create_panel_title(message.node)

                    async for delta in message.text:
                        text += delta
                        markdown = Markdown(text + block)
                        live.update(
                            Panel(
                                markdown,
                                title=title,
                                title_align="left",
                                border_style=colour,
                            ),
                            refresh=True,
                        )

                    if text.strip():
                        if message.node == "tools" and len(text) > 200:
                            text = text[:200]
                        live.update(
                            Panel(
                                Markdown(text),
                                title=title,
                                title_align="left",
                                border_style=colour,
                            ),
                            refresh=True,
                        )

                full_message = await message.output
                usage = full_message.usage_metadata
                if usage:
                    console.print(
                        f"[dim]tokens — in: {usage.get('input_tokens')}, "
                        f"out: {usage.get('output_tokens')}, "
                        f"total: {usage.get('total_tokens')}[/dim]"
                    )

        async def consume_values():
            async for value in stream.values:
                all_messages: list[BaseMessage] = value["messages"]
                with open(working_memory_file, "w") as f:
                    yaml.dump(messages_to_dict(all_messages), f)

        await asyncio.gather(consume_messages(), consume_values())

        with open(memory_file, "w") as f:
            yaml.dump(messages_to_dict(messages), f)

        subprocess.run(
            ["terminal-notifier", "-message", "Karl is ready", "-sound", "Heroine"]
        )
        new_message = input("You: ")


def _create_panel_title(step):
    return {
        "model": ("Karl", "red"),
        "tools": ("Data", "white"),
        "SummarizationMiddleware.before_model": ("Karl's inner monologue", "yellow"),
    }.get(step, (f"Unknown: {step}", "grey0"))

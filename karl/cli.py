import asyncio
import os
from functools import wraps, partial

import requests
import typer
from langchain_openai import ChatOpenAI

from . import runner
from .bot_runner import KarlBot
from .linkedin.agents import create as create_linkedin_agent
from .email.agents import create as create_email_agent
from .agents.todo import create as create_todo_agent
from .agents.general import create as create_general_agent
from .job import review_job_ad

DEFAULT_MODEL = "openai:gpt-5.6-sol"
app = typer.Typer()


def syncify(func):
    @wraps(func)
    def f(*args, **kwargs):
        asyncio.run(func(*args, **kwargs))

    return f


@app.command()
@syncify
async def job(url: str, model: str = DEFAULT_MODEL):
    await review_job_ad(url, model)


@app.command()
@syncify
async def email(message: str, model: str = DEFAULT_MODEL):
    await runner.run(
        create_email_agent(model), message, memory_path="email_memory_v3.yaml"
    )


@app.command()
@syncify
async def linkedin(message: str, model: str = DEFAULT_MODEL):
    await runner.run(
        create_linkedin_agent(model), message, memory_path="linkedin_memory_7.yaml"
    )


@app.command()
@syncify
async def todo(message: str, model: str = DEFAULT_MODEL):
    await runner.run(
        await create_todo_agent(model), message, memory_path="todo_memory.yaml"
    )


# @app.command()
# @syncify
# async def auto(message: str, model: str = DEFAULT_MODEL):
#     await runner.run(
#         await create_autodidact_agent(model), message, memory_path="auto_memory_2.yaml"
#     )


@app.command()
@syncify
async def bot(model: str = DEFAULT_MODEL):
    model = ChatOpenAI(
        model=os.getenv("KARL_MODEL", "gpt-5.6-sol-bedrock"),
        temperature=0.3,
        streaming=True,
        stream_chunk_timeout=900,
        timeout=900,
        use_responses_api=True,
    )

    avatar_url = "https://static.wikia.nocookie.net/simpsons/images/1/18/Karl2.png/revision/latest"
    avatar_bytes = requests.get(avatar_url).content
    await KarlBot(
        partial(create_general_agent, model),
        os.getenv("MATRIX_HOMESERVER"),
        os.getenv("BOT_USER_ID"),
        "karl",
        os.getenv("MY_MATRIX_ID"),
        allow_room_creation=True,
    ).start("Karl", avatar_bytes)

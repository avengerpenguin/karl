from functools import wraps

import asyncio
import typer
from langchain.agents import create_agent
from . import agent
from .job import review_job_ad

DEFAULT_MODEL = "gemma4:latest"

syncify = lambda f: wraps(f)(lambda *args, **kwargs: asyncio.run(f(*args, **kwargs)))
app = typer.Typer()


@app.command()
@syncify
async def job(url: str, model: str = DEFAULT_MODEL):
    await review_job_ad(url, model)


@app.command()
@syncify
async def email(message: str, model: str = DEFAULT_MODEL, interactive: bool = False):
    await agent.run(message)

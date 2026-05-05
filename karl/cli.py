import asyncio
from functools import wraps

import typer

from . import runner
from .agents.linkedin import create as create_linkedin_agent
from .agents.email_agent import create as create_email_agent
from .job import review_job_ad


DEFAULT_MODEL = "ollama:gemma4:latest"

syncify = lambda f: wraps(f)(lambda *args, **kwargs: asyncio.run(f(*args, **kwargs)))
app = typer.Typer()


@app.command()
@syncify
async def job(url: str, model: str = DEFAULT_MODEL):
    await review_job_ad(url, model)


@app.command()
@syncify
async def email(message: str, model: str = DEFAULT_MODEL):
    await runner.run(create_email_agent(model), message, memory_path="email_memory.yaml")


@app.command()
@syncify
async def linkedin(message: str, model: str = DEFAULT_MODEL):
    await runner.run(create_linkedin_agent(model), message, memory_path="linkedin_memory_6.yaml")

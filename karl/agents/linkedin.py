"""
Dedicated-AI LinkedIn agent.
Can be plugged into the Karl framework or be called as a subagent to need.

TODO:

- Behaviour tests for the cold start currently tested by hand
- Explicit contract (BotEx) tests to check it's happy with the tool interface
- Guardrail tests? Anything to limit yet?
- Vary behaviour tests on model to compare
- Give access to CV (via tool)
- Test a 2nd message on top of the cold start -- does it learn from the previous interactions (i.e. context window)
- Test edge case of no messages
- Test variations of messages: good salary, low ball, london hybrid for user in Manchester etc.
- Explore Judgements for quality testing
"""

from textwrap import dedent

from langchain.agents import create_agent

from ..tools.cv import fetch_cv
from ..tools.linkedin import find_latest_non_replied_chat, find_past_reply_examples, save_draft_message


def create(model = "ollama:qwen3:14b"):
    return create_agent(
        model,
        [
            find_latest_non_replied_chat,
            find_past_reply_examples,
            save_draft_message,
            fetch_cv,
        ],
        system_prompt=dedent("""\
        You are a personal assistant to help users with their LinkedIn interactions.

        You have access to the user's LinkedIn instant messages. In particular you have tools available to do two things:

        The user will ask you to check for any messages not replied to. If there are none, simply say so and stop.
        If there are messages not replied to, do the following:

        1. Read the latest message that is not replied to
        2. Check some past examples of how the user has responded to other recruiters
        3. Optionally, fetch the user's CV to understand the user's background and experience better
        4. Ask the user for clarification if you are unsure how to respond
        6. Optionally, you may include some conclusions and judgements on the role offered given the CV
        5. Once happy, respond with a suggested draft message that the user should send in reply to this latest message

        Note sometimes on LinkedIn the user just gets spam messages trying to sell things. If the most recent message is
        clearly not a recruiter, just flag it as such and prompt the user to archive it.

        Finally, try to keep responses short and brief as this is LinkedIn messaging not formal email and the person
        we're talking to is likely a recruiter not the company itself so there's no pressing need to go into a lot of
        detail or "sell" the candidate.

        Recruiters tend to want to go to a call pretty quickly anyway so it can be sufficient just to reply quickly
        to express interest.
        """),
    )

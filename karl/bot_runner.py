import asyncio
import hashlib
import json
import logging
import re

from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langgraph.prebuilt._tool_call_stream import ToolCallStream
from langgraph.stream import AsyncGraphRunStream, StreamChannel
from langgraph.types import Command
from typing_extensions import override

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    raise ImportError("Please install karl[matrix] to use Matrix")

from .bot import PersonalBot
from .oauth import redirect_presenter
from .types import AgentFactory, AgentSessionContext, Elicitor
from markdown_it import MarkdownIt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_INTERRUPT_CHARS = 12_000
MAX_MATRIX_NOTICE_CHARS = 20_000
LARGE_STRING_ARG_CHARS = 200
PREVIEW_ARG_NAMES = {
    "content",
    "body",
    "message",
    "text",
    "markdown",
    "description",
}


def _get_thread_root_event_id(event: RoomMessageText) -> str:
    relates_to = getattr(event, "source", {}).get("content", {}).get("m.relates_to", {})

    if relates_to.get("rel_type") == "m.thread" and relates_to.get("event_id"):
        return relates_to["event_id"]

    return event.event_id


def _markdown_code_fence(value: str, language: str = "") -> str:
    backtick_runs = re.findall(r"`+", value)
    longest_run = max((len(run) for run in backtick_runs), default=0)
    fence = "`" * max(3, longest_run + 1)

    if language:
        return f"{fence}{language}\n{value}\n{fence}"

    return f"{fence}\n{value}\n{fence}"


def _truncate_review_text(value: str) -> str:
    if len(value) <= MAX_INTERRUPT_CHARS:
        return value

    return value[:MAX_INTERRUPT_CHARS] + "\n... <truncated>"


def _human_review_resume_command(
    decision_type: str,
    message: str | None,
    decision_count: int,
) -> Command:
    if decision_count < 1:
        decision_count = 1

    if decision_type == "approve":
        decisions = [
            {
                "type": "approve",
            }
            for _ in range(decision_count)
        ]

    else:
        decisions = [
            {
                "type": "reject",
                "message": message or "The Matrix user rejected this tool call.",
            }
            for _ in range(decision_count)
        ]

    return Command(
        resume={
            "decisions": decisions,
        }
    )


def _format_tool_args_for_review(args: object) -> str:
    if not isinstance(args, dict):
        try:
            pretty_args = json.dumps(
                args,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            pretty_args = str(args)

        pretty_args = _truncate_review_text(pretty_args)

        return "Arguments:\n\n" + _markdown_code_fence(pretty_args, "json")

    display_args = dict(args)
    rendered_sections: list[str] = []

    for key, value in args.items():
        if not isinstance(value, str):
            continue

        should_render_separately = (
            key in PREVIEW_ARG_NAMES
            or "\n" in value
            or len(value) >= LARGE_STRING_ARG_CHARS
        )

        if not should_render_separately:
            continue

        display_args[key] = "<rendered below>"

        preview = value.strip() if value.strip() else "<empty>"
        preview = _truncate_review_text(preview)

        rendered_sections.append(f"{key} preview:\n\n---\n\n{preview}\n\n---")

    pretty_args = json.dumps(
        display_args,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    pretty_args = _truncate_review_text(pretty_args)

    sections = [
        "Arguments:",
        _markdown_code_fence(pretty_args, "json"),
    ]
    sections.extend(rendered_sections)

    return _truncate_review_text("\n\n".join(sections))


def _format_interrupt(interrupt: object) -> str:
    value = getattr(interrupt, "value", interrupt)

    if isinstance(value, dict):
        action_requests = value.get("action_requests") or []

        if action_requests:
            sections = ["Approval needed before continuing."]

            for index, action in enumerate(action_requests, start=1):
                name = action.get("name", "unknown_tool")
                args = action.get("args", {})

                if len(action_requests) == 1:
                    sections.append(
                        f"Tool: `{name}`\n\n{_format_tool_args_for_review(args)}"
                    )
                else:
                    sections.append(
                        f"Tool {index}: `{name}`\n\n"
                        f"{_format_tool_args_for_review(args)}"
                    )

            if len(action_requests) == 1:
                sections.append(
                    "Reply in this thread with one of:\n\n"
                    "- `approve`\n"
                    "- `deny`\n"
                    "- `deny: <reason>`"
                )
            else:
                sections.append(
                    f"This approval contains **{len(action_requests)} tool calls**.\n\n"
                    "Reply in this thread with one of:\n\n"
                    "- `approve` — approve all listed tool calls\n"
                    "- `deny` — deny all listed tool calls\n"
                    "- `deny: <reason>` — deny all listed tool calls with feedback"
                )

            return "\n\n".join(sections)

    try:
        pretty_value = json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        pretty_value = str(value)

    if len(pretty_value) > MAX_INTERRUPT_CHARS:
        pretty_value = pretty_value[:MAX_INTERRUPT_CHARS] + "\n... <truncated>"

    return (
        "Approval needed before continuing.\n\n"
        f"{_markdown_code_fence(pretty_value, 'json')}\n\n"
        "Reply in this thread with one of:\n\n"
        "- `approve`\n"
        "- `deny`\n"
        "- `deny: <reason>`"
    )


def _thread_relates_to(
    thread_root_event_id: str,
    reply_to_event_id: str | None = None,
) -> dict:
    return {
        "rel_type": "m.thread",
        "event_id": thread_root_event_id,
        "is_falling_back": True,
        "m.in_reply_to": {
            "event_id": reply_to_event_id or thread_root_event_id,
        },
    }


def _pending_human_decision_count(state: object) -> int:
    count = 0

    tasks = getattr(state, "tasks", None) or []
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()

        for interrupt in interrupts:
            value = getattr(interrupt, "value", interrupt)

            if isinstance(value, dict):
                action_requests = value.get("action_requests") or []
                count += len(action_requests)

    values = getattr(state, "values", None)
    if isinstance(values, dict):
        interrupts = values.get("__interrupt__") or []

        if not isinstance(interrupts, list | tuple):
            interrupts = [interrupts]

        for interrupt in interrupts:
            value = getattr(interrupt, "value", interrupt)

            if isinstance(value, dict):
                action_requests = value.get("action_requests") or []
                count += len(action_requests)

    return count


def _human_review_decision(text: str) -> tuple[str, str | None] | None:
    stripped = text.strip()
    lowered = stripped.lower()

    if lowered == "approve":
        return "approve", None

    if lowered in {"deny", "reject"}:
        return (
            "reject",
            (
                "The Matrix user rejected this tool call. "
                "Do not retry the same tool call unchanged."
            ),
        )

    for prefix in ("deny because ", "reject because ", "deny: ", "reject: "):
        if lowered.startswith(prefix):
            reason = stripped[len(prefix) :].strip()
            return (
                "reject",
                (
                    "The Matrix user rejected this tool call with this feedback:\n\n"
                    f"{reason}\n\n"
                    "Revise your plan accordingly. Do not retry the same tool call unchanged."
                ),
            )

    return None


def _agent_thread_id(room_id: str, thread_root_event_id: str) -> str:
    raw = f"{room_id}:{thread_root_event_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _split_matrix_notice(text: str) -> list[str]:
    if len(text) <= MAX_MATRIX_NOTICE_CHARS:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        chunk = remaining[:MAX_MATRIX_NOTICE_CHARS]

        if len(remaining) > MAX_MATRIX_NOTICE_CHARS:
            split_at = max(chunk.rfind("\n\n"), chunk.rfind("\n"))

            if split_at > MAX_MATRIX_NOTICE_CHARS // 2:
                chunk = remaining[:split_at].rstrip()
                remaining = remaining[split_at:].lstrip()
            else:
                remaining = remaining[MAX_MATRIX_NOTICE_CHARS:].lstrip()
        else:
            remaining = ""

        chunks.append(chunk)

    total = len(chunks)
    return [
        f"(part {index}/{total})\n\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    ]


class _MatrixThreadElicitor(Elicitor):
    def __init__(
        self,
        bot: "KarlBot",
        room_id: str,
        thread_root_event_id: str,
        reply_to_event_id: str,
    ):
        self._bot = bot
        self._room_id = room_id
        self._thread_root_event_id = thread_root_event_id
        self._reply_to_event_id = reply_to_event_id

    async def request_url(self, *, message: str, url: str) -> str:
        event_id = await self._bot._send_notice(
            self._room_id,
            f"{message}\n\n{url}",
            thread_root_event_id=self._thread_root_event_id,
            reply_to_event_id=self._reply_to_event_id,
        )
        return event_id or f"matrix-elicitation:{self._thread_root_event_id}"

    async def complete(self, elicitation_id: str) -> None:
        # Matrix currently has no elicitation lifecycle event to update.
        return None


class KarlBot(PersonalBot):
    def __init__(
        self,
        agent_factory: AgentFactory,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agent_factory = agent_factory

    async def _react_to_event(self, room_id: str, event_id: str, key: str) -> None:
        await self.client.room_send(
            room_id=room_id,
            message_type="m.reaction",
            content={
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": event_id,
                    "key": key,
                },
            },
        )

    async def _send_text_message(
        self,
        room_id: str,
        text: str,
        thread_root_event_id: str | None = None,
        reply_to_event_id: str | None = None,
    ) -> str | None:
        md = MarkdownIt("commonmark", {"html": False, "breaks": False})
        content = {
            "msgtype": "m.text",
            "format": "org.matrix.custom.html",
            "body": text,
            "formatted_body": md.render(text).strip(),
        }

        if thread_root_event_id is not None:
            content["m.relates_to"] = _thread_relates_to(
                thread_root_event_id,
                reply_to_event_id,
            )

        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content,
        )
        return getattr(response, "event_id", None)

    async def _send_notice(
        self,
        room_id: str,
        text: str,
        thread_root_event_id: str | None = None,
        reply_to_event_id: str | None = None,
    ) -> str | None:
        first_event_id: str | None = None

        for chunk in _split_matrix_notice(text):
            content = {
                "msgtype": "m.notice",
                "body": chunk,
            }

            if thread_root_event_id is not None:
                content["m.relates_to"] = _thread_relates_to(
                    thread_root_event_id,
                    reply_to_event_id,
                )

            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
            )

            event_id = getattr(response, "event_id", None)
            if event_id is None:
                logging.warning(
                    "Matrix notice send did not return an event_id: %r", response
                )

            if first_event_id is None:
                first_event_id = event_id

        return first_event_id

    async def _edit_notice(self, room_id: str, event_id: str, text: str) -> None:
        await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": f"* {text}",
                "m.new_content": {
                    "msgtype": "m.notice",
                    "body": text,
                },
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": event_id,
                },
            },
        )

    async def _edit_text_message(self, room_id: str, event_id: str, text: str) -> None:
        md = MarkdownIt("commonmark", {"html": False, "breaks": False})
        formatted_body = md.render(text).strip()
        # formatted_body = markdown.markdown(text).strip()

        await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": f"* {text}",
                "format": "org.matrix.custom.html",
                "formatted_body": f"* {formatted_body}",
                "m.new_content": {
                    "msgtype": "m.text",
                    "format": "org.matrix.custom.html",
                    "body": text,
                    "formatted_body": formatted_body,
                },
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": event_id,
                },
            },
        )

    @override
    async def generate_reply(self, room: MatrixRoom, event: RoomMessageText) -> None:
        thread_root_event_id = _get_thread_root_event_id(event)
        agent_thread_id = _agent_thread_id(room.room_id, thread_root_event_id)
        session = AgentSessionContext(
            session_id=agent_thread_id,
            elicitor=_MatrixThreadElicitor(
                self,
                room.room_id,
                thread_root_event_id,
                event.event_id,
            ),
        )

        async def present_oauth_url(url: str) -> None:
            await session.elicitor.request_url(
                message="Sign in to continue",
                url=url,
            )

        async with redirect_presenter(present_oauth_url):
            agent = await self.agent_factory(session)

            agent_config = {
                "configurable": {
                    "thread_id": agent_thread_id,
                    "matrix_room_id": room.room_id,
                    "matrix_thread_root_event_id": thread_root_event_id,
                    "profile_override": {
                        "max_input_tokens": 1000000  # Sets the trigger to ~850k tokens
                    },
                },
                # "callbacks": [ConsoleTraceHandler()],
                "tags": ["matrix", "karlbot"],
                "metadata": {
                    "room_id": room.room_id,
                    "thread_root_event_id": thread_root_event_id,
                },
            }

            review_decision = _human_review_decision(event.body)
            resume_command: Command | None = None

            if review_decision is not None:
                decision_type, message = review_decision
                state = await agent.aget_state(agent_config)
                decision_count = _pending_human_decision_count(state)

                resume_command = _human_review_resume_command(
                    decision_type=decision_type,
                    message=message,
                    decision_count=decision_count,
                )

            agent_input = (
                resume_command
                if resume_command is not None
                else {"messages": [HumanMessage(content=event.body)]}
            )

            await self.client.room_typing(room.room_id, True)
            await self._send_notice(
                room.room_id,
                "Resuming…" if resume_command is not None else "Thinking…",
                thread_root_event_id=thread_root_event_id,
                reply_to_event_id=event.event_id,
            )

            reaction = "✅"
            interrupt_was_reported = False
            consumer_tasks: set[asyncio.Task] = set()

            try:
                stream: AsyncGraphRunStream = await agent.astream_events(
                    agent_input,
                    version="v3",
                    config=agent_config,
                )

                async def consume_messages() -> None:
                    messages_out: StreamChannel = stream.messages
                    async for message in messages_out:
                        message: AsyncChatModelStream

                        async def consume_reasoning() -> None:
                            reasoning = ""
                            reasoning_event_id: str | None = None
                            last_edit_at = 0.0
                            min_edit_interval_seconds = 5.0

                            async for delta in message.reasoning:
                                if not delta:
                                    continue

                                reasoning += delta

                                if not reasoning.strip():
                                    continue

                                now = asyncio.get_running_loop().time()
                                preview_text = f"[thinking] {reasoning.strip()} ▌"

                                if reasoning_event_id is None:
                                    reasoning_event_id = await self._send_notice(
                                        room.room_id,
                                        preview_text,
                                        thread_root_event_id=thread_root_event_id,
                                        reply_to_event_id=event.event_id,
                                    )
                                    last_edit_at = now
                                elif now - last_edit_at >= min_edit_interval_seconds:
                                    await self._edit_notice(
                                        room.room_id,
                                        reasoning_event_id,
                                        preview_text,
                                    )
                                    last_edit_at = now

                            final_reasoning = reasoning.strip()
                            if final_reasoning and reasoning_event_id is not None:
                                await self._edit_notice(
                                    room.room_id,
                                    reasoning_event_id,
                                    f"[thinking] {final_reasoning}",
                                )

                        async def consume_text() -> tuple[str, str | None]:
                            text = ""
                            message_event_id: str | None = None
                            last_edit_at = 0.0
                            min_edit_interval_seconds = 5.0

                            async for delta in message.text:
                                if not delta:
                                    continue

                                text += delta

                                if not text.strip():
                                    continue

                                now = asyncio.get_running_loop().time()
                                preview_text = text.strip() + " ▌"

                                if message_event_id is None:
                                    message_event_id = await self._send_text_message(
                                        room.room_id,
                                        preview_text,
                                        thread_root_event_id=thread_root_event_id,
                                        reply_to_event_id=event.event_id,
                                    )
                                    last_edit_at = now
                                elif now - last_edit_at >= min_edit_interval_seconds:
                                    await self._edit_text_message(
                                        room.room_id,
                                        message_event_id,
                                        preview_text,
                                    )
                                    last_edit_at = now

                            return text, message_event_id

                        reasoning_task = asyncio.create_task(consume_reasoning())
                        text_task = asyncio.create_task(consume_text())

                        _, text_result = await asyncio.gather(
                            reasoning_task,
                            text_task,
                        )
                        text, message_event_id = text_result

                        full_message: AIMessage = await message.output
                        final_text = (full_message.text or text).strip()
                        if not final_text:
                            continue

                        usage = full_message.usage_metadata
                        if usage:
                            model_name = (full_message.response_metadata or {}).get(
                                "model_name"
                            )
                            model_prefix = (
                                f"model: {model_name} — " if model_name else ""
                            )
                            await self._send_notice(
                                room.room_id,
                                f"{model_prefix}tokens — in: {usage.get('input_tokens')}, "
                                f"out: {usage.get('output_tokens')}, "
                                f"total: {usage.get('total_tokens')}",
                                thread_root_event_id=thread_root_event_id,
                                reply_to_event_id=event.event_id,
                            )

                        if message_event_id is None:
                            await self._send_text_message(
                                room.room_id,
                                final_text,
                                thread_root_event_id=thread_root_event_id,
                                reply_to_event_id=event.event_id,
                            )
                        else:
                            await self._edit_text_message(
                                room.room_id,
                                message_event_id,
                                final_text,
                            )

                async def consume_tool_calls() -> None:
                    tool_calls: StreamChannel = stream.tool_calls
                    async for call in tool_calls:
                        call: ToolCallStream
                        tool_name = call.tool_name
                        tool_args = call.input
                        tool_args_snip = (
                            str(tool_args)[:200]
                            if len(str(tool_args)) > 200
                            else str(tool_args)
                        )

                        await self._send_notice(
                            room.room_id,
                            f"Calling tool `{tool_name}` with input `{tool_args_snip}`",
                            thread_root_event_id=thread_root_event_id,
                            reply_to_event_id=event.event_id,
                        )

                async def consume_interrupts() -> None:
                    nonlocal interrupt_was_reported

                    for interrupt in await stream.interrupts():
                        interrupt_was_reported = True
                        await self._send_notice(
                            room.room_id,
                            _format_interrupt(interrupt),
                            thread_root_event_id=thread_root_event_id,
                            reply_to_event_id=event.event_id,
                        )
                        return

                consumer_tasks = {
                    asyncio.create_task(consume_messages()),
                    asyncio.create_task(consume_tool_calls()),
                    asyncio.create_task(consume_interrupts()),
                }

                while consumer_tasks:
                    done, consumer_tasks = await asyncio.wait(
                        consumer_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    await asyncio.gather(*done)

                    if interrupt_was_reported:
                        for task in consumer_tasks:
                            task.cancel()

                        await asyncio.gather(*consumer_tasks, return_exceptions=True)
                        break

            except Exception:
                for task in consumer_tasks:
                    task.cancel()

                await asyncio.gather(*consumer_tasks, return_exceptions=True)
                logger.exception("Failed to generate reply")
                await self._send_text_message(
                    room.room_id,
                    "Sorry, something went wrong.",
                    thread_root_event_id=thread_root_event_id,
                    reply_to_event_id=event.event_id,
                )
                reaction = "❌"
            finally:
                await self.client.room_typing(room.room_id, False)
                await self._react_to_event(room.room_id, event.event_id, reaction)

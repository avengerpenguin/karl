import asyncio
import hashlib
import json
import re

from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt._tool_call_stream import ToolCallStream
from langgraph.stream import AsyncGraphRunStream, StreamChannel
from langgraph.types import Command
from typing_extensions import override
from .agents.autodidact import create as create_autodidact_agent

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    raise ImportError("Please install karl[matrix] to use Matrix")

from .bot import PersonalBot
from markdown_it import MarkdownIt


# MODEL = "ollama:qwen3.6:27b-coding-nvfp4"
# MODEL = "ollama:gemma4:12b-mlx"
# MODEL = "openai:mlx-community/Qwen3.6-35B-A3B-4bit"
MODEL = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy",
    model="mlx-community/Qwen3.6-27B-4bit",
    temperature=0.3,
    streaming=True,
    stream_chunk_timeout=600,
    timeout=900,
)
MODEL = "openai:gpt-5.6-sol"
MAX_INTERRUPT_CHARS = 12_000
LARGE_STRING_ARG_CHARS = 200
PREVIEW_ARG_NAMES = {
    "content",
    "body",
    "message",
    "text",
    "markdown",
    "description",
}


class KarlBot(PersonalBot):
    def _get_thread_root_event_id(self, event: RoomMessageText) -> str:
        relates_to = (
            getattr(event, "source", {}).get("content", {}).get("m.relates_to", {})
        )

        if relates_to.get("rel_type") == "m.thread" and relates_to.get("event_id"):
            return relates_to["event_id"]

        return event.event_id

    def _agent_thread_id(self, room_id: str, thread_root_event_id: str) -> str:
        raw = f"{room_id}:{thread_root_event_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _human_review_decision(self, text: str) -> tuple[str, str | None] | None:
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

    def _pending_human_decision_count(self, state: object) -> int:
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

    def _human_review_resume_command(
        self,
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

    def _markdown_code_fence(self, value: str, language: str = "") -> str:
        backtick_runs = re.findall(r"`+", value)
        longest_run = max((len(run) for run in backtick_runs), default=0)
        fence = "`" * max(3, longest_run + 1)

        if language:
            return f"{fence}{language}\n{value}\n{fence}"

        return f"{fence}\n{value}\n{fence}"

    def _format_tool_args_for_review(self, args: object) -> str:
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

            if len(pretty_args) > MAX_INTERRUPT_CHARS:
                pretty_args = pretty_args[:MAX_INTERRUPT_CHARS] + "\n... <truncated>"

            return "Arguments:\n\n" + self._markdown_code_fence(pretty_args, "json")

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
            if len(preview) > MAX_INTERRUPT_CHARS:
                preview = preview[:MAX_INTERRUPT_CHARS] + "\n... <truncated>"

            rendered_sections.append(f"{key} preview:\n\n---\n\n{preview}\n\n---")

        pretty_args = json.dumps(
            display_args,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        sections = [
            "Arguments:",
            self._markdown_code_fence(pretty_args, "json"),
        ]
        sections.extend(rendered_sections)

        return "\n\n".join(sections)

    def _format_interrupt(self, interrupt: object) -> str:
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
                            f"Tool: `{name}`\n\n"
                            f"{self._format_tool_args_for_review(args)}"
                        )
                    else:
                        sections.append(
                            f"Tool {index}: `{name}`\n\n"
                            f"{self._format_tool_args_for_review(args)}"
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
            f"{self._markdown_code_fence(pretty_value, 'json')}\n\n"
            "Reply in this thread with one of:\n\n"
            "- `approve`\n"
            "- `deny`\n"
            "- `deny: <reason>`"
        )

    def _thread_relates_to(
        self,
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
            content["m.relates_to"] = self._thread_relates_to(
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
        content = {
            "msgtype": "m.notice",
            "body": text,
        }

        if thread_root_event_id is not None:
            content["m.relates_to"] = self._thread_relates_to(
                thread_root_event_id,
                reply_to_event_id,
            )

        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content,
        )
        return getattr(response, "event_id", None)

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
        agent = await create_autodidact_agent(MODEL)

        thread_root_event_id = self._get_thread_root_event_id(event)
        agent_thread_id = self._agent_thread_id(room.room_id, thread_root_event_id)
        agent_config = {
            "configurable": {
                "thread_id": agent_thread_id,
                "matrix_room_id": room.room_id,
                "matrix_thread_root_event_id": thread_root_event_id,
            }
        }

        review_decision = self._human_review_decision(event.body)
        resume_command: Command | None = None

        if review_decision is not None:
            decision_type, message = review_decision
            state = await agent.aget_state(agent_config)
            decision_count = self._pending_human_decision_count(state)

            resume_command = self._human_review_resume_command(
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
                        async for delta in message.reasoning:
                            if not delta or not delta.strip():
                                continue

                            await self._send_notice(
                                room.room_id,
                                f"[thinking] {delta.strip()}",
                                thread_root_event_id=thread_root_event_id,
                                reply_to_event_id=event.event_id,
                            )

                    async def consume_text() -> tuple[str, str | None]:
                        text = ""
                        message_event_id: str | None = None
                        last_edit_at = 0.0
                        min_edit_interval_seconds = 1.0

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
                        await self._send_notice(
                            room.room_id,
                            f"tokens — in: {usage.get('input_tokens')}, "
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
                        self._format_interrupt(interrupt),
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

                for task in done:
                    task.result()

                if interrupt_was_reported:
                    for task in consumer_tasks:
                        task.cancel()

                    await asyncio.gather(*consumer_tasks, return_exceptions=True)
                    break

        except Exception:
            reaction = "❌"
            raise
        finally:
            await self.client.room_typing(room.room_id, False)
            await self._react_to_event(room.room_id, event.event_id, reaction)

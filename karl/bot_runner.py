import asyncio
import hashlib
import os

import yaml
from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    HumanMessage,
    messages_to_dict,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt._tool_call_stream import ToolCallStream
from langgraph.stream import AsyncGraphRunStream, StreamChannel
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

        memory_file = os.path.join(os.getcwd(), f"bot-thread-{agent_thread_id}.yaml")
        working_memory_file = memory_file.replace(".yaml", ".working.yaml")

        if os.path.exists(memory_file):
            with open(memory_file) as f:
                messages: list[BaseMessage] = messages_from_dict(
                    yaml.load(f, Loader=yaml.FullLoader) or []
                )

        else:
            messages: list[BaseMessage] = []

        messages.append(HumanMessage(content=event.body))

        await self.client.room_typing(room.room_id, True)
        await self._send_notice(
            room.room_id,
            "Thinking…",
            thread_root_event_id=thread_root_event_id,
            reply_to_event_id=event.event_id,
        )

        reaction = "✅"
        try:
            agent_config = {
                "configurable": {
                    "thread_id": agent_thread_id,
                    "matrix_room_id": room.room_id,
                    "matrix_thread_root_event_id": thread_root_event_id,
                }
            }
            stream: AsyncGraphRunStream = await agent.astream_events(
                dict(messages=messages), version="v3", config=agent_config
            )

            async def consume_messages():
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
                    messages.append(full_message)

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

            async def consume_tool_calls():
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

            async def consume_values():
                async for value in stream.values:
                    all_messages: list[BaseMessage] = value["messages"]
                    with open(working_memory_file, "w") as f:
                        yaml.dump(messages_to_dict(all_messages), f)

            await asyncio.gather(
                consume_messages(), consume_tool_calls(), consume_values()
            )
        except Exception:
            reaction = "❌"
            raise
        finally:
            await self.client.room_typing(room.room_id, False)
            await self._react_to_event(room.room_id, event.event_id, reaction)

        with open(memory_file, "w") as f:
            yaml.dump(messages_to_dict(messages), f)

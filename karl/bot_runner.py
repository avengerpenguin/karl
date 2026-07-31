import asyncio
import datetime
import os
from datetime import timedelta

import markdown
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
from .agents.autodidact import create as create_autodidact_agent

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    raise ImportError("Please install karl[matrix] to use Matrix")

from .bot import PersonalBot


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

    async def _send_text_message(self, room_id: str, text: str) -> str | None:
        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "format": "org.matrix.custom.html",
                "body": text,
                "formatted_body": markdown.markdown(text).strip(),
            },
        )
        return getattr(response, "event_id", None)

    async def _edit_text_message(self, room_id: str, event_id: str, text: str) -> None:
        formatted_body = markdown.markdown(text).strip()

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

    async def generate_reply(self, room: MatrixRoom, event: RoomMessageText) -> None:
        agent = await create_autodidact_agent(MODEL)

        current_day = (datetime.datetime.now() - timedelta(hours=4)).date().isoformat()
        memory_file = os.path.join(os.getcwd(), f"bot-{current_day}.yaml")
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
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": "Thinking…",
            },
        )

        reaction = "✅"
        try:
            stream: AsyncGraphRunStream = await agent.astream_events(
                dict(messages=messages), version="v3"
            )

            async def consume_messages():
                messages_out: StreamChannel = stream.messages
                async for message in messages_out:
                    message: AsyncChatModelStream

                    async def consume_reasoning() -> None:
                        async for delta in message.reasoning:
                            if not delta or not delta.strip():
                                continue

                            await self.client.room_send(
                                room_id=room.room_id,
                                message_type="m.room.message",
                                content={
                                    "msgtype": "m.notice",
                                    "body": f"[thinking] {delta.strip()}",
                                },
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
                        await self.client.room_send(
                            room_id=room.room_id,
                            message_type="m.room.message",
                            content={
                                "msgtype": "m.notice",
                                "body": f"tokens — in: {usage.get('input_tokens')}, "
                                f"out: {usage.get('output_tokens')}, "
                                f"total: {usage.get('total_tokens')}",
                            },
                        )

                    if message_event_id is None:
                        await self._send_text_message(room.room_id, final_text)
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

                    await self.client.room_send(
                        room_id=room.room_id,
                        message_type="m.room.message",
                        content={
                            "msgtype": "m.notice",
                            "body": f"Calling tool `{tool_name}` with input `{tool_args_snip}`",
                        },
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

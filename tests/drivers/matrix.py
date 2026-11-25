import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest_asyncio
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from nio import (
    AsyncClient,
    JoinResponse,
    RegisterResponse,
    RoomCreateResponse,
    RoomMessageNotice,
    RoomMessageText,
    RoomPreset,
    RoomSendResponse,
)

import karl.bot as bot_module
from karl.bot_runner import KarlBot


_TEST_PASSWORD = "matrix-acceptance-test-password"


class MatrixWorld:
    def __init__(self, homeserver: str):
        self.homeserver = homeserver
        self._users: list[MatrixUser] = []
        self._rooms: dict[frozenset[str], str] = {}
        self._bots: list[KarlHarness] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

    async def create_user(self, name: str) -> "MatrixUser":
        username = f"{name}-{uuid.uuid4().hex}"
        client = AsyncClient(self.homeserver)
        response = await client.register(username, _TEST_PASSWORD)
        if not isinstance(response, RegisterResponse):
            await client.close()
            raise RuntimeError(f"Failed to register Matrix user {name}: {response}")

        user = MatrixUser(self, name, response.user_id, client)
        self._users.append(user)
        return user

    def start_karl(self, owner: "MatrixUser", tools=None) -> "KarlHarness":
        bot = KarlHarness(self, owner, tools=tools)
        self._bots.append(bot)
        return bot

    async def close(self) -> None:
        for bot in reversed(self._bots):
            await bot.close()
        for user in reversed(self._users):
            await user.client.close()

    def room_for(self, first: "MatrixUser", second: "MatrixUser") -> str:
        try:
            return self._rooms[frozenset((first.user_id, second.user_id))]
        except KeyError as error:
            raise RuntimeError(
                f"No direct room exists between {first.name} and {second.name}"
            ) from error


class MatrixUser:
    def __init__(
        self,
        world: MatrixWorld,
        name: str,
        user_id: str,
        client: AsyncClient,
    ):
        self.world = world
        self.name = name
        self.user_id = user_id
        self.client = client

    async def send_direct_message(
        self,
        recipient: "MatrixUser",
        body: str,
    ) -> "SentMessage":
        room_id = self.world.room_for(self, recipient)
        await self.client.sync(timeout=0)
        since = self.client.next_batch

        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": body},
        )
        if not isinstance(response, RoomSendResponse):
            raise RuntimeError(f"Failed to send Matrix message: {response}")

        return SentMessage(
            sender=self,
            recipient=recipient,
            room_id=room_id,
            event_id=response.event_id,
            since=since,
        )

    def __repr__(self) -> str:
        return f"MatrixUser(name={self.name!r}, user_id={self.user_id!r})"


class ToolCallingFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = (
            self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        )
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tool_call["name"],
                            "args": json.dumps(tool_call["args"]),
                            "id": tool_call["id"],
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                    chunk_position="last",
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
            return
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                chunk_position="last",
            )
        )
        if run_manager:
            run_manager.on_llm_new_token(message.content, chunk=chunk)
        yield chunk


class KarlHarness(MatrixUser):
    def __init__(self, world: MatrixWorld, owner: MatrixUser, tools=None):
        self.world = world
        self.owner = owner
        self.tools = tools or []
        self.name = "karl"
        self.user_id = ""
        self.client = None
        self._provisioning_client: AsyncClient | None = None
        self._bot: KarlBot | None = None
        self._task: asyncio.Task | None = None
        self._reply_finished = asyncio.Event()
        self._token_paths: tuple[Path, Path] | None = None

    async def __aenter__(self):
        username = f"karl-{uuid.uuid4().hex}"
        provisioning_client = AsyncClient(self.world.homeserver)
        response = await provisioning_client.register(username, _TEST_PASSWORD)
        if not isinstance(response, RegisterResponse):
            await provisioning_client.close()
            raise RuntimeError(f"Failed to register Karl's Matrix user: {response}")

        self.user_id = response.user_id
        self._provisioning_client = provisioning_client

        room_response = await provisioning_client.room_create(
            is_direct=True,
            invite=[self.owner.user_id],
            preset=RoomPreset.trusted_private_chat,
        )
        if not isinstance(room_response, RoomCreateResponse):
            raise RuntimeError(f"Failed to create direct room: {room_response}")

        join_response = await self.owner.client.join(room_response.room_id)
        if not isinstance(join_response, JoinResponse):
            raise RuntimeError(f"Alice failed to join direct room: {join_response}")

        self.world._rooms[frozenset((self.owner.user_id, self.user_id))] = (
            room_response.room_id
        )
        await provisioning_client.close()
        self._provisioning_client = None

        async def agent_factory():
            if not self.tools:
                model = GenericFakeChatModel(messages=iter(["Four"]))
            else:
                tool = self.tools[0]
                model = ToolCallingFakeChatModel(
                    messages=iter(
                        [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": tool.name,
                                        "args": {"location": "Paris"},
                                        "id": "call-1",
                                        "type": "tool_call",
                                    }
                                ],
                            ),
                            AIMessage(content="It is 20 degrees"),
                        ]
                    )
                )
            return create_agent(model=model, tools=self.tools)

        bot_module.BOT_PASSWORD = _TEST_PASSWORD
        localpart = self.user_id.split(":", 1)[0]
        token_directory = Path(".store")
        token_directory.mkdir(exist_ok=True)
        self._token_paths = (
            token_directory / f".{localpart}-access-token",
            token_directory / f".{localpart}-sync-token",
        )
        for token_path in self._token_paths:
            token_path.unlink(missing_ok=True)

        self._bot = KarlBot(
            agent_factory,
            self.world.homeserver,
            self.user_id,
            f"test-{uuid.uuid4().hex}",
            self.owner.user_id,
        )
        generate_reply = self._bot.generate_reply

        async def tracked_generate_reply(room, event):
            try:
                await generate_reply(room, event)
            finally:
                self._reply_finished.set()

        self._bot.generate_reply = tracked_generate_reply
        self.client = self._bot.client
        self._task = asyncio.create_task(self._bot.start("Karl"))
        await self._wait_until_ready()
        return self

    async def _wait_until_ready(self, timeout: float = 5.0) -> None:
        async def ready() -> None:
            while True:
                if self._task.done():
                    self._task.result()
                if self.client.access_token and any(
                    callback.filter is RoomMessageText
                    for callback in self.client.event_callbacks
                ):
                    return
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(ready(), timeout)
        except TimeoutError as error:
            raise RuntimeError(
                "Karl did not become ready to receive messages"
            ) from error

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

    async def close(self) -> None:
        try:
            if self._task is not None:
                if not self._reply_finished.is_set():
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._reply_finished.wait(), 5.0)
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        finally:
            self._task = None
            if self._provisioning_client is not None:
                await self._provisioning_client.close()
                self._provisioning_client = None
            if self.client is not None:
                await self.client.close()
            if self._token_paths is not None:
                for token_path in self._token_paths:
                    token_path.unlink(missing_ok=True)
                self._token_paths = None


class SentMessage:
    def __init__(
        self,
        sender: MatrixUser,
        recipient: MatrixUser,
        room_id: str,
        event_id: str,
        since: str,
    ):
        self.sender = sender
        self.recipient = recipient
        self.room_id = room_id
        self.event_id = event_id
        self.since = since

    async def wait_for_reply(self, timeout: float = 15.0) -> "ReceivedMessage":
        return (await self._wait_for_messages(RoomMessageText, 1, timeout))[0]

    async def wait_for_notices(
        self,
        count: int = 2,
        timeout: float = 15.0,
    ) -> list["ReceivedMessage"]:
        return await self._wait_for_messages(RoomMessageNotice, count, timeout)

    async def _wait_for_messages(
        self,
        event_type,
        count: int,
        timeout: float,
    ) -> list["ReceivedMessage"]:
        async def poll() -> list[ReceivedMessage]:
            since = self.since
            messages = []
            while len(messages) < count:
                response = await self.sender.client.sync(
                    since=since,
                    timeout=250,
                )
                since = response.next_batch
                room = response.rooms.join.get(self.room_id)
                if room is None:
                    continue

                for event in room.timeline.events:
                    content = getattr(event, "source", {}).get("content", {})
                    relation = content.get("m.relates_to", {})
                    if (
                        isinstance(event, event_type)
                        and event.sender == self.recipient.user_id
                        and relation.get("rel_type") == "m.thread"
                        and relation.get("event_id") == self.event_id
                    ):
                        messages.append(ReceivedMessage(self.recipient, event.body))
                        if len(messages) == count:
                            return messages
            return messages

        try:
            return await asyncio.wait_for(poll(), timeout)
        except TimeoutError as error:
            raise TimeoutError(
                f"No {event_type.__name__} events from {self.recipient.name} "
                f"to event {self.event_id} within {timeout} seconds"
            ) from error


@dataclass(frozen=True)
class ReceivedMessage:
    sender: MatrixUser
    body: str


@pytest_asyncio.fixture
async def matrix(synapse_url):
    async with MatrixWorld(synapse_url) as world:
        yield world


@pytest_asyncio.fixture
async def alice(matrix):
    return await matrix.create_user("alice")

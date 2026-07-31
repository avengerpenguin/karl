import abc
import asyncio
import hashlib
import os
from io import BytesIO
from typing import List

import sys

from nio import (
    AsyncClient,
    MatrixRoom,
    RoomMessageText,
    LoginResponse,
    LoginError,
    RoomPreset,
    RoomCreateResponse,
    RoomCreateError,
    UploadError,
    AsyncClientConfig,
    ReceiptType,
    SyncResponse,
    JoinedRoomsResponse,
    JoinedRoomsError,
    JoinedMembersResponse,
    JoinedMembersError,
    RoomMember,
)

BOT_PASSWORD = os.getenv("BOT_PASSWORD")


class PersonalBot(abc.ABC):
    def __init__(
        self, homeserver, bot_user_id, device_id, my_user_id, allow_room_creation=False
    ):
        print(f"Creating with with user {bot_user_id}")
        self.client = AsyncClient(
            homeserver,
            bot_user_id,
            store_path="./store/",
            device_id=device_id,
            config=AsyncClientConfig(
                max_limit_exceeded=10,
                max_timeouts=10,
                store_sync_tokens=True,
                max_timeout_retry_wait_time=60,
                backoff_factor=2,
            ),
        )
        self.my_user_id = my_user_id
        self.allow_room_creation = allow_room_creation

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText) -> None:
        print(f"{event.sender}: {event.body}", file=sys.stderr)

        if event.sender != self.my_user_id:
            return

        await self.client.update_receipt_marker(
            room.room_id, event.event_id, ReceiptType.read
        )
        await self.client.room_read_markers(
            room.room_id, fully_read_event=event.event_id
        )

        asyncio.create_task(self.generate_reply(room, event))

    @abc.abstractmethod
    async def generate_reply(self, room: MatrixRoom, event: RoomMessageText) -> None:
        raise NotImplementedError("generate_reply must be implemented in subclass")

    async def get_or_create_dm_room(self):
        # rooms = self.client.rooms
        rooms_response: (
            JoinedRoomsResponse | JoinedRoomsError
        ) = await self.client.joined_rooms()
        if isinstance(rooms_response, JoinedRoomsError):
            raise RuntimeError(f"Failed to list joined rooms: {rooms_response.message}")

        room_ids = set(rooms_response.rooms)
        # room_ids = list(self.client.rooms.keys())
        print("Found rooms: " + ", ".join(room_ids), file=sys.stderr)
        chosen_room = None
        for room_id in room_ids:
            members_response: (
                JoinedMembersResponse | JoinedMembersError
            ) = await self.client.joined_members(room_id)
            if isinstance(members_response, JoinedMembersError):
                raise RuntimeError(
                    f"Failed to list members in room {room_id}: {members_response.message}"
                )

            members: List[RoomMember] = members_response.members
            member_ids = [member.user_id for member in members]

            print(
                f"Checking room: {room_id} {member_ids}",
                file=sys.stderr,
            )
            if len(members) == 2 and self.my_user_id in member_ids:
                chosen_room = room_id
                break

        for rid in room_ids:
            if not rid == chosen_room:
                print(f"Leaving room: {rid}", file=sys.stderr)
                await self.client.room_leave(rid)

        if chosen_room:
            print(f"Found existing DM room: {chosen_room}, {members}", file=sys.stderr)
            return chosen_room

        if not self.allow_room_creation:
            raise RuntimeError("Failed to find or create DM room")

        # No DM found → create one
        response: RoomCreateResponse | RoomCreateError = await self.client.room_create(
            is_direct=True,
            invite=[self.my_user_id],
            preset=RoomPreset.trusted_private_chat,
        )
        if isinstance(response, RoomCreateError):
            raise RuntimeError(f"Failed to create DM: {response}")

        print(f"Created new DM room: {response.room_id}", file=sys.stderr)
        room_id = response.room_id

        # Give Matrix a moment to register the room
        await asyncio.sleep(1)

        return room_id

    async def start(self, bot_name: str, avatar_bytes: bytes | None = None):
        await self.log_in()
        sync_token_path = f".store/.{self.client.user.split(':')[0]}-sync-token"
        if os.path.exists(sync_token_path):
            os.makedirs(".store", exist_ok=True)
            with open(sync_token_path) as f:
                sync_token = f.read()
        else:
            sync_token = None

        await self.client.sync(
            since=sync_token,
            set_presence="online",
            timeout=300000,
        )

        asyncio.create_task(self.get_or_create_dm_room())
        asyncio.create_task(self.client.set_displayname(bot_name))
        if avatar_bytes is not None:
            print("Image bytes provided; queuing avatar update", file=sys.stderr)
            asyncio.create_task(self.update_avatar(avatar_bytes))

        self.client.add_event_callback(
            self.message_callback,
            RoomMessageText,
        )

        async def sync_callback(response):
            with open(sync_token_path, "w") as f:
                f.write(response.next_batch)

        self.client.add_response_callback(sync_callback, SyncResponse)

        try:
            await self.client.sync_forever(
                timeout=60000,
                full_state=False,
                set_presence="online",
                loop_sleep_time=10000,
                since=sync_token,
            )
        finally:
            await self.client.set_presence("offline")
            await self.client.close()

    async def log_in(self):
        access_token_path = f".store/.{self.client.user.split(':')[0]}-access-token"

        if os.path.exists(access_token_path):
            os.makedirs(".store", exist_ok=True)
            with open(access_token_path) as f:
                access_token = f.read()
                self.client.access_token = access_token

        else:
            login_success: LoginResponse | LoginError = await self.client.login(
                password=BOT_PASSWORD
            )

            if isinstance(login_success, LoginError):
                raise RuntimeError(
                    f"Login failed: {login_success.status_code} {login_success.message}",
                )

            access_token = login_success.access_token
            self.client.access_token = access_token
            with open(access_token_path, "w") as f:
                f.write(access_token)

    async def update_avatar(self, avatar_bytes: bytes):
        file_name = hashlib.sha1(avatar_bytes).hexdigest()
        upload_response, _ = await self.client.upload(
            BytesIO(avatar_bytes),
            content_type="image/webp",
            filename=f"{file_name}.webp",
            filesize=len(avatar_bytes),
        )
        if isinstance(upload_response, UploadError):
            print(f"Avatar upload failed: {upload_response.message}", file=sys.stderr)
        else:
            mxc_url = upload_response.content_uri
            print(
                f"Avatar uploaded successfully with MXC URL: {mxc_url}", file=sys.stderr
            )
            await self.client.set_avatar(mxc_url)

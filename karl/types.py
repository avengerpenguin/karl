from typing import Protocol


class MatrixBot(Protocol):
    def __init__(
        self,
        homeserver: str,
        bot_user_id,
        device_id,
        my_user_id,
    ) -> None: ...

    async def log_in(self) -> None: ...

    async def get_or_create_dm_room(self) -> str: ...

    async def update_avatar(self, avatar_bytes: bytes) -> None: ...

    async def start(self, bot_name: str, avatar_bytes: bytes | None = None) -> None: ...

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


def _env_flag(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None

    return value.strip().lower() in {"1", "true", "yes", "on"}


def in_container() -> bool:
    """Best-effort detection of Docker/Podman/Kubernetes runtimes."""
    override = _env_flag("KARL_IN_CONTAINER")
    if override is not None:
        return override

    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True

    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False

    return any(
        marker in cgroup
        for marker in ("docker", "containerd", "kubepods", "podman", "lxc")
    )


def _oauth_callback_bind_host() -> str:
    """Address the callback server listens on."""
    # Inside a container loopback is not reachable from the host browser,
    # so we have to listen on all interfaces and rely on published ports.
    return os.getenv("KARL_OAUTH_CALLBACK_BIND_HOST") or (
        "0.0.0.0" if in_container() else "127.0.0.1"
    )


def _oauth_callback_public_host() -> str:
    """Host the browser (and therefore the redirect URI) uses."""
    return os.getenv("KARL_OAUTH_CALLBACK_PUBLIC_HOST", "localhost")


def _oauth_callback_port_window() -> tuple[int, int]:
    base = int(os.getenv("KARL_OAUTH_CALLBACK_PORT_BASE", "18000"))
    # A wide span is fine on the host, but in a container the range has to be
    # small enough to publish in docker-compose.
    default_span = "16" if in_container() else "20000"
    span = int(os.getenv("KARL_OAUTH_CALLBACK_PORT_SPAN", default_span))

    return base, max(span, 1)


def _preferred_oauth_callback_port(service: str) -> int:
    base, span = _oauth_callback_port_window()
    digest = hashlib.sha256(service.encode("utf-8")).hexdigest()
    return base + (int(digest[:8], 16) % span)


def _allocate_oauth_callback_ports(services: list[str]) -> dict[str, int]:
    base, span = _oauth_callback_port_window()
    limit = base + span
    used_ports: set[int] = set()
    ports: dict[str, int] = {}

    if len(services) > span:
        raise ValueError(
            f"{len(services)} OAuth services do not fit in the callback port "
            f"range {base}-{limit - 1}; raise KARL_OAUTH_CALLBACK_PORT_SPAN."
        )

    for service in sorted(services):
        port = _preferred_oauth_callback_port(service)

        while port in used_ports:
            # Wrap inside the window so we never escape the published range.
            port = base + ((port + 1 - base) % span)

        used_ports.add(port)
        ports[service] = port

    return ports


def oauth_redirect_uri(port: int, path: str = "/callback") -> str:
    return f"http://{_oauth_callback_public_host()}:{port}{path}"


async def handle_redirect(auth_url: str) -> None:
    print(f"Visit: {auth_url}")

    if not in_container():
        os.system(f"open '{auth_url}'")


def make_handle_callback(port: int):
    async def handle_callback() -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        callback_received: asyncio.Future[tuple[str, str | None]] = loop.create_future()

        async def handle_request(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            request_line = await reader.readline()
            request = request_line.decode("utf-8", errors="replace").strip()

            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            try:
                _method, target, _version = request.split(" ", 2)
                params = parse_qs(urlparse(target).query)
                code = params["code"][0]
                state = params.get("state", [None])[0]

                if not callback_received.done():
                    callback_received.set_result((code, state))

                body = "OAuth callback received. You can close this tab."
                status = "200 OK"
            except Exception as exc:
                if not callback_received.done():
                    callback_received.set_exception(exc)

                body = "OAuth callback failed. Check the terminal for details."
                status = "400 Bad Request"

            response = (
                f"HTTP/1.1 {status}\r\n"
                "Content-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                "Connection: close\r\n"
                "\r\n"
                f"{body}"
            )
            writer.write(response.encode("utf-8"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(
            handle_request, _oauth_callback_bind_host(), port
        )

        async with server:
            return await callback_received

    return handle_callback


async def handle_callback() -> tuple[str, str | None]:
    loop = asyncio.get_running_loop()
    callback_received: asyncio.Future[tuple[str, str | None]] = loop.create_future()

    async def handle_request(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = await reader.readline()
        request = request_line.decode("utf-8", errors="replace").strip()

        while True:
            line = await reader.readline()
            if line in {b"\r\n", b"\n", b""}:
                break

        try:
            _method, target, _version = request.split(" ", 2)
            params = parse_qs(urlparse(target).query)
            code = params["code"][0]
            state = params.get("state", [None])[0]

            if not callback_received.done():
                callback_received.set_result((code, state))

            body = "OAuth callback received. You can close this tab."
            status = "200 OK"
        except Exception as exc:
            if not callback_received.done():
                callback_received.set_exception(exc)

            body = "OAuth callback failed. Check the terminal for details."
            status = "400 Bad Request"

        response = (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            "Connection: close\r\n"
            "\r\n"
            f"{body}"
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_request, "localhost", 8000)

    async with server:
        return await callback_received


class FileTokenStorage(TokenStorage):
    """Local JSON-file token storage for single-instance MCP OAuth clients."""

    def __init__(
        self,
        storage_key: str,
        token_file: Path | None = None,
    ):
        self.storage_key = storage_key
        self.token_file = token_file or Path(
            os.getenv("KARL_OAUTH_TOKEN_FILE", ".karl/oauth-tokens.json")
        )

    def _read_all(self) -> dict[str, Any]:
        if not self.token_file.exists():
            return {}

        with self.token_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_all(self, data: dict[str, Any]) -> None:
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

        tmp_file = self.token_file.with_suffix(f"{self.token_file.suffix}.tmp")
        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")

        os.replace(tmp_file, self.token_file)
        self.token_file.chmod(0o600)

    def _read_service_data(self) -> dict[str, Any]:
        return self._read_all().get(self.storage_key, {})

    def _write_service_data(self, service_data: dict[str, Any]) -> None:
        data = self._read_all()
        data[self.storage_key] = service_data
        self._write_all(data)

    async def get_tokens(self) -> OAuthToken | None:
        """Get stored tokens."""
        tokens = self._read_service_data().get("tokens")
        if tokens is None:
            return None

        return OAuthToken.model_validate(tokens)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        service_data = self._read_service_data()
        service_data["tokens"] = tokens.model_dump(mode="json")
        self._write_service_data(service_data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get stored client information."""
        client_info = self._read_service_data().get("client_info")
        if client_info is None:
            return None

        return OAuthClientInformationFull.model_validate(client_info)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        service_data = self._read_service_data()
        service_data["client_info"] = client_info.model_dump(mode="json")
        self._write_service_data(service_data)


class InMemoryTokenStorage(TokenStorage):
    """Demo In-memory token storage implementation."""

    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        """Get stored tokens."""
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get stored client information."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        self.client_info = client_info

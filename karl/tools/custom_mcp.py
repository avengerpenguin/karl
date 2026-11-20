"""Load MCP servers as LangChain tools, isolating each server so one failure can't break the rest."""

import asyncio
import json
import logging
import os
import sys
import traceback
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import (
    SSETransport,
    StdioTransport,
    StreamableHttpTransport,
)
from langchain.mcp import as_langchain_tool
from langchain_core.tools import BaseTool
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from karl.oauth import (
    _allocate_oauth_callback_ports,
    FileTokenStorage,
    handle_redirect,
    make_handle_callback,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0

Connection = dict[str, Any]


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten nested (Base)ExceptionGroups into the actual root errors."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaf_exceptions(sub)]
    return [exc]


def _transport_kind(connection: Connection) -> str:
    return connection.get("transport") or (
        "stdio" if "command" in connection else "http"
    )


def _describe(connection: Connection) -> str:
    kind = _transport_kind(connection)
    if kind == "stdio":
        return f"stdio: {connection.get('command')} {' '.join(connection.get('args', []))}".rstrip()
    return f"{kind}: {connection.get('url')}"


def _build_client(connection: Connection) -> Client:
    """Translate our (langchain-mcp-adapters style) connection dict into a FastMCP client."""
    kind = _transport_kind(connection)
    if kind == "stdio":
        transport = StdioTransport(
            command=connection["command"],
            args=connection.get("args", []),
            env=connection.get("env"),
            cwd=connection.get("cwd"),
        )
    elif kind in {"http", "streamable_http", "streamable-http"}:
        transport = StreamableHttpTransport(
            url=connection["url"],
            headers=connection.get("headers"),
            auth=connection.get("auth"),
        )
    elif kind == "sse":
        transport = SSETransport(
            url=connection["url"],
            headers=connection.get("headers"),
            auth=connection.get("auth"),
        )
    else:
        raise ValueError(f"Unsupported MCP transport {kind!r}")
    return Client(transport)


async def _load_one(
    name: str,
    connection: Connection,
    timeout: float | None,
    tool_name_prefix: bool,
) -> list[BaseTool]:
    # OAuth may block on a browser round-trip, so don't time those out
    if isinstance(connection.get("auth"), OAuthClientProvider):
        timeout = None
    try:
        client = _build_client(connection)

        async def discover():
            async with client:
                return await client.list_tools()

        mcp_tools = await asyncio.wait_for(discover(), timeout)
        tools = [await as_langchain_tool(tool, client) for tool in mcp_tools]
        if tool_name_prefix:
            for tool in tools:
                tool.name = f"{name}_{tool.name}"

        print(
            f"[MCP] '{name}' loaded {len(tools)} tools ({_describe(connection)})",
            file=sys.stderr,
        )
        return tools
    except Exception as exc:  # noqa: BLE001 - we deliberately isolate any failure
        summary = "; ".join(
            f"{type(e).__name__}: {e}" for e in _leaf_exceptions(exc)
        ) or repr(exc)
        print(
            f"WARNING: [MCP] server '{name}' failed to load and will be skipped "
            f"({_describe(connection)}) -> {summary}",
            file=sys.stderr,
        )
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exception(exc, file=sys.stderr)
        return []


async def load_mcp_tools_isolated(
    connections: dict[str, Connection],
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
    tool_name_prefix: bool = False,
) -> list[BaseTool]:
    """Load tools from each MCP server independently; failing servers are dropped, not fatal."""
    if not connections:
        return []
    results = await asyncio.gather(
        *(
            _load_one(name, conn, timeout, tool_name_prefix)
            for name, conn in connections.items()
        )
    )
    return [tool for tools in results for tool in tools]


def _attach_oauth(mcp_config: dict[str, Connection]) -> None:
    oauth_services = [s for s, c in mcp_config.items() if c.get("auth") == "oauth"]
    ports = _allocate_oauth_callback_ports(oauth_services)

    for service in oauth_services:
        callback_port = ports[service]
        mcp_config[service]["auth"] = OAuthClientProvider(
            server_url=mcp_config[service]["url"],
            client_metadata=OAuthClientMetadata(
                client_name="Karl MCP Client",
                redirect_uris=[AnyUrl(f"http://localhost:{callback_port}/callback")],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="user",
            ),
            storage=FileTokenStorage(service),
            redirect_handler=handle_redirect,
            callback_handler=make_handle_callback(callback_port),
        )


async def get_custom_mcp_tools() -> list[BaseTool]:
    # Expand ${VAR} references (e.g. ${JUPYTER_TOKEN}) so secrets needn't be inlined in the JSON
    mcp_config = json.loads(os.path.expandvars(os.getenv("CUSTOM_MCP_URLS", "{}")))
    _attach_oauth(mcp_config)
    return await load_mcp_tools_isolated(mcp_config, tool_name_prefix=True)

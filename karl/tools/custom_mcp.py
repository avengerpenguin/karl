import json
import os

from karl.oauth import (
    _allocate_oauth_callback_ports,
    FileTokenStorage,
    handle_redirect,
    make_handle_callback,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl


async def get_custom_mcp_tools():
    mcp_config = json.loads(os.getenv("CUSTOM_MCP_URLS", "{}"))
    oauth_services = [
        service
        for service, mcp in mcp_config.items()
        if mcp.get("auth", None) == "oauth"
    ]
    oauth_callback_ports = _allocate_oauth_callback_ports(oauth_services)

    for service, mcp in mcp_config.items():
        if mcp.get("auth", None) == "oauth":
            storage = FileTokenStorage(service)
            callback_port = oauth_callback_ports[service]
            redirect_uri = f"http://localhost:{callback_port}/callback"

            oauth_auth = OAuthClientProvider(
                server_url=mcp["url"],
                client_metadata=OAuthClientMetadata(
                    client_name="Karl MCP Client",
                    redirect_uris=[AnyUrl(redirect_uri)],
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    scope="user",
                ),
                storage=storage,
                redirect_handler=handle_redirect,
                callback_handler=make_handle_callback(callback_port),
            )
            mcp["auth"] = oauth_auth

    return await MultiServerMCPClient(mcp_config, tool_name_prefix=True).get_tools()

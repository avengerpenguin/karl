import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_tools():
    client = MultiServerMCPClient(
        {
            "messages": {
                "transport": "http",
                "url": "https://mcp.slack.com/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('SLACK_MCP_TOKEN')}",
                }
            },
        }
    )
    return [
        tool for tool in await client.get_tools()
        if tool.name in {
            "slack_search_public_and_private",
            "slack_search_channels",
            "slack_search_users",
            "slack_read_channel",
            "slack_read_thread",
            "slack_read_user_profile",
        }
    ]

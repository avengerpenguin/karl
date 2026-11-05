import os

from karl.tools.custom_mcp import load_mcp_tools_isolated

SLACK_TOOLS = {
    "slack_search_public_and_private",
    "slack_search_channels",
    "slack_search_users",
    "slack_read_channel",
    "slack_read_thread",
    "slack_read_user_profile",
}


async def get_tools():
    tools = await load_mcp_tools_isolated(
        {
            "slack": {
                "transport": "http",
                "url": "https://mcp.slack.com/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('SLACK_MCP_TOKEN')}",
                },
            },
        }
    )
    return [tool for tool in tools if tool.name in SLACK_TOOLS]

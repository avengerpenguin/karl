from pathlib import Path

import pytest
import pytest_asyncio
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from ..drivers.matrix import WeatherChatModel


@pytest.fixture(scope="session")
def synapse_url():
    data_dir = Path(__file__).parent.parent / "synapse-data"

    container = (
        DockerContainer("matrixdotorg/synapse:latest")
        .with_env("SYNAPSE_CONFIG_PATH", "/data/homeserver.yaml")
        .with_copy_into_container(
            data_dir / "homeserver.yaml",
            "/data/homeserver.yaml",
        )
        .with_copy_into_container(
            data_dir / "test.log.config",
            "/data/test.log.config",
        )
        .with_copy_into_container(
            data_dir / "test.signing.key",
            "/data/test.signing.key",
        )
        .with_exposed_ports(8008)
    )

    with container:
        wait_for_logs(
            container,
            "Synapse now listening on TCP port 8008",
            timeout=30,
        )
        host = container.get_container_host_ip()
        port = container.get_exposed_port(8008)
        yield f"http://{host}:{port}"


@pytest_asyncio.fixture
async def karl_stub(matrix, alice):
    async with matrix.start_karl(owner=alice) as bot:
        yield bot


@pytest_asyncio.fixture
async def karl_broken(matrix, alice):
    # Create version that has exceptions coming out of langchain for any reason
    async with matrix.start_karl(owner=alice, error=True) as bot:
        yield bot


@pytest_asyncio.fixture
async def karl_with_named_model(matrix, alice):
    async with matrix.start_karl(owner=alice, model_name="stub-model-1") as bot:
        yield bot


@pytest_asyncio.fixture
async def karl_with_tool(matrix, alice, get_weather):
    async with matrix.start_karl(owner=alice, tools=[get_weather]) as bot:
        yield bot


@pytest_asyncio.fixture
async def karl_with_hitl_tool(matrix, alice, get_weather):
    async with matrix.start_karl(
        owner=alice,
        tools=[get_weather],
        # Use HumanInTheLoopMiddleware to add approval to this tool
        require_approval=["get_weather"],
    ) as bot:
        yield bot


@pytest_asyncio.fixture
async def karl_with_two_hitl_tools(matrix, alice, get_weather, get_time):
    async with matrix.start_karl(
        owner=alice,
        tools=[get_weather, get_time],
        # Use HumanInTheLoopMiddleware to add approval to this tool
        require_approval=["get_weather", "get_time"],
    ) as bot:
        yield bot


@pytest_asyncio.fixture
async def karl_needing_authorization(matrix, alice, oauth_authority):
    async def agent_factory(session):
        async def get_weather(location: str) -> str:
            """Get today's calendar after the user signs in."""
            await oauth_authority.require_authorization(session.elicitor)
            return "Today's calendar is clear."

        return create_agent(
            model=WeatherChatModel(messages=iter(())),
            tools=[get_weather],
            checkpointer=InMemorySaver(),
        )

    async with matrix.start_karl(
        owner=alice,
        agent_factory=agent_factory,
    ) as bot:
        bot.authorization_url = oauth_authority.authorization_url
        yield bot

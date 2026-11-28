import asyncio
import json
import uuid
from pathlib import Path
import socket
from functools import cache
from json import JSONDecodeError
from textwrap import dedent
from urllib.parse import urlparse

import pytest
import yaml

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool, BaseTool
from langchain_openai import ChatOpenAI
from vcr import VCR

from karl.oauth import handle_redirect, redirect_presenter
from karl.types import Elicitor

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
DOCKERHUB_HOSTS = {
    "auth.docker.io",
    "index.docker.io",
    "registry-1.docker.io",
    "production.cloudflare.docker.com",
}
MLX_GATEWAY_PORT = 5276
LOCAL_PORTS = {MLX_GATEWAY_PORT}


pytest_plugins = ["tests.drivers.matrix"]


def running_in_docker() -> bool:
    """Return True when the tests are running inside a Docker container."""
    if Path("/.dockerenv").exists():
        return True

    try:
        return "docker" in Path("/proc/1/cgroup").read_text()
    except OSError:
        return False


MLX_GATEWAY_HOST = "host.docker.internal" if running_in_docker() else "127.0.0.1"


def ignore_local_request(request):
    """Don't record requests to local services or Docker Hub into cassettes."""
    parsed = urlparse(request.uri)
    return (
        None
        if (
            parsed.hostname in DOCKERHUB_HOSTS
            or parsed.hostname in LOCAL_HOSTS
            and parsed.port in LOCAL_PORTS
        )
        else request
    )


@cache
def mlx_gateway_available(
    host: str = MLX_GATEWAY_HOST,
    port: int = MLX_GATEWAY_PORT,
    timeout: float = 0.5,
) -> bool:
    """Return True if something is listening on the MLX gateway port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, RuntimeError):
        # RuntimeError covers pytest-recording's --block-network socket guard
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Abort xdist runs that would load MLXChatModel in every worker.

    Without the gateway, chat_model falls back to an in-process MLXChatModel. Under
    xdist, each worker would load its own copy of the model.
    """
    if hasattr(config, "workerinput"):
        return  # an xdist worker; the controller has already checked

    if config.getoption("numprocesses", default=None) and not mlx_gateway_available():
        raise pytest.UsageError(
            f"No MLX gateway at {MLX_GATEWAY_HOST}:{MLX_GATEWAY_PORT}. Running "
            "under xdist would load a separate MLXChatModel in every worker. "
            "Start the gateway, or run with -n 0."
        )


class PrettyJsonYamlDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def represent_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str",
            data,
            style="|",
        )
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        data,
    )


PrettyJsonYamlDumper.add_representer(str, represent_str)


class PrettyJsonYamlSerializer:
    def serialize(self, data):
        return yaml.dump(data, Dumper=PrettyJsonYamlDumper)

    def deserialize(self, data):
        loaded = yaml.load(data, Loader=yaml.SafeLoader)

        for interaction in loaded.get("interactions", []):
            response = interaction.get("response", {})
            body = response.get("body", {})

            if "string" in body:
                value = body["string"]

                if isinstance(value, str):
                    value = value.encode("utf-8")

                if not isinstance(value, bytes):
                    value = bytes(value)

                body["string"] = value

        return loaded


def pytest_recording_configure(config, vcr: VCR):
    vcr.register_serializer(
        "yaml",
        serializer=PrettyJsonYamlSerializer(),
    )


def pretty_print_json_body(response):
    body = response["body"].get("string")

    # Normalise to str for JSON parsing
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError:
            pass

    try:
        parsed = json.loads(body)
    except JSONDecodeError:
        return response  # not JSON, leave unchanged
    except UnicodeDecodeError:
        return response  # Not even a UTF-8 string; it might be binary data

    # Pretty-print JSON (still just a normal str!)
    response["body"]["string"] = json.dumps(parsed, indent=2).encode("utf-8")

    return response


def pytest_collection_modifyitems(items):
    for item in items:
        # Only touch VCR-marked tests that are parameterised
        if item.get_closest_marker("vcr") is None:
            continue

        # originalname is the bare function name (no [param] suffix);
        # falls back to name for non-parametrised tests.
        base_name = getattr(item, "originalname", None) or item.name
        if base_name == item.name:
            continue  # not parametrised, nothing to change

        item.add_marker(pytest.mark.default_cassette(f"{base_name}.yaml"))


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-stainless-retry-count"],
        "serializer": "yaml",
        "before_record_request": ignore_local_request,
        "before_record_response": pretty_print_json_body,
        "ignore_localhost": False,
        "allow_playback_repeats": True,
    }


@pytest.fixture(
    scope="session",
    params=dedent("""\
        mlx-community/Qwen3.8-27B-4bit
    """)
    .strip()
    .splitlines(),
)
def chat_model(request: pytest.FixtureRequest) -> BaseChatModel:
    if mlx_gateway_available():
        # The OpenAI-compatible MLX gateway handles locking, so only one
        # generation runs at a time and the model is loaded only once.
        llm = ChatOpenAI(
            base_url=f"http://{MLX_GATEWAY_HOST}:{MLX_GATEWAY_PORT}/v1",
            api_key="dummy",
            model=request.param,
        )
    else:
        # Self-contained fallback with no network needed. It loads the model
        # in this process, so only use it when the gateway isn't running.
        from karl.mlxproxy.chatmodel import MLXChatModel

        llm = MLXChatModel(
            model_id=request.param,
        )
    return llm


class OAuthAuthority:
    """Test double for an OAuth provider that waits for an explicit browser visit."""

    def __init__(self, matrix) -> None:
        self.authorization_url = f"https://authorization.test/sign-in/{uuid.uuid4()}"
        self._authorized = asyncio.Event()
        matrix.register_url_visitor(self.authorization_url, self.authorize)

    async def authorize(self) -> None:
        self._authorized.set()

    async def require_authorization(self, elicitor: Elicitor) -> None:
        elicitation_id: str | None = None

        async def present(auth_url: str) -> None:
            nonlocal elicitation_id
            elicitation_id = await elicitor.request_url(
                message="Sign in to continue",
                url=auth_url,
            )

        async with redirect_presenter(present):
            await handle_redirect(self.authorization_url)
        await self._authorized.wait()
        if elicitation_id is not None:
            await elicitor.complete(elicitation_id)


@pytest.fixture
def oauth_authority(matrix) -> OAuthAuthority:
    return OAuthAuthority(matrix)


@pytest.fixture
def get_weather() -> BaseTool:
    def f(location: str):
        """Get current weather and optional forecast."""
        return f"Current weather in {location}: 20 degrees"

    return StructuredTool.from_function(f, name="get_weather")


@pytest.fixture
def get_time() -> BaseTool:
    def f():
        """Get current time."""
        return "12:00"

    return StructuredTool.from_function(f, name="get_time")


@pytest.fixture
def get_calendar() -> BaseTool:
    def f():
        """Get today's calendar."""
        return "You have no meetings today."

    return StructuredTool.from_function(f, name="get_calendar")

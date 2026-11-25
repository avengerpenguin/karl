from pathlib import Path

from typing import Any, Generator

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from ..drivers import matrix


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


@pytest.fixture
def matrix_homeserver(synapse_url) -> Generator[matrix.MatrixHomeServer, Any, None]:
    """Agent implements create_world(synapse_url) in tests/drivers/matrix_driver.py."""
    yield matrix.MatrixHomeServer(synapse_url)

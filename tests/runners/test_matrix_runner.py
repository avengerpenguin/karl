from pathlib import Path
import time

import httpx
import pytest
from testcontainers.core.container import DockerContainer


@pytest.fixture(scope="session")
def matrix_homeserver():
    data_dir = Path(__file__).parent / "synapse-data"

    with (
        DockerContainer("matrixdotorg/synapse:<pinned-version>")
        .with_volume_mapping(str(data_dir.resolve()), "/data", mode="rw")
        .with_env("SYNAPSE_CONFIG_PATH", "/data/homeserver.yaml")
        .with_exposed_ports(8008)
    ) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(8008)
        base_url = f"http://{host}:{port}"

        deadline = time.monotonic() + 30

        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{base_url}/_matrix/client/versions",
                    timeout=1,
                )
                if response.is_success:
                    break
            except httpx.HTTPError:
                pass

            time.sleep(0.25)
        else:
            raise RuntimeError("Synapse did not become ready")

        yield base_url

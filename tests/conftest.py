import json
from json import JSONDecodeError
from urllib.parse import urlparse

import pytest
import yaml
from vcr import VCR


OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1"}
OLLAMA_PORT = 11434


def ignore_ollama_request(request):
    parsed = urlparse(request.uri)
    return (
        None
        if parsed.hostname in OLLAMA_HOSTS and parsed.port == OLLAMA_PORT
        else request
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
        "before_record_request": ignore_ollama_request,
        "before_record_response": pretty_print_json_body,
        "ignore_localhost": False,
        "allow_playback_repeats": True,
    }

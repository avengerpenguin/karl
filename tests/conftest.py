import json

import pytest
import yaml
from vcr import VCR


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
                print(f"Checking body: {type(value)}")

                if isinstance(value, str):
                    value = value.encode("utf-8")

                if not isinstance(value, bytes):
                    value = bytes(value)

                body["string"] = value

        return loaded

def pytest_recording_configure(config, vcr: VCR):
    vcr.register_serializer(
        "pretty_json_yaml",
        serializer=PrettyJsonYamlSerializer(),
    )


def pretty_print_json_body(response):
    body = response["body"].get("string")

    # Normalise to str for JSON parsing
    if isinstance(body, bytes):
        body = body.decode("utf-8")

    try:
        parsed = json.loads(body)
    except Exception:
        return response  # not JSON, leave unchanged

    # Pretty-print JSON (still just a normal str!)
    response["body"]["string"] = json.dumps(parsed, indent=2).encode("utf-8")

    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-stainless-retry-count"],
        "serializer": "pretty_json_yaml",
        "before_record_response": pretty_print_json_body,
        "ignore_localhost": False,
        "allow_playback_repeats": True,
    }

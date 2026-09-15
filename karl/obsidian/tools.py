import subprocess
import sys

from langchain_core.tools import tool


def run_obsidian(args: list[str], timeout_seconds: int = 30) -> str:
    print(f"Running Obsidian CLI: {args!r}", file=sys.stderr)
    try:
        result = subprocess.run(
            ["obsidian", *args],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"Timeout for Obsidian CLI: {args!r}", file=sys.stderr)
        raise RuntimeError(
            f"Obsidian CLI timed out after {timeout_seconds}s. "
            f"Command args: {args!r}"
        ) from exc

    if result.returncode != 0:
        print(f"Failed Obsidian CLI: {args!r}", file=sys.stderr)
        raise RuntimeError(
            "Obsidian CLI failed\n"
            f"Exit code: {result.returncode}\n"
            f"Args: {args!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    print(f"Completed Obsidian CLI: {args!r}", file=sys.stderr)
    return result.stdout


def obsidian_args(vault: str | None = None) -> list[str]:
    return [f"vault={vault}"] if vault else []


@tool
def list_obsidian_vaults() -> list[str]:
    """
    Lists all Obsidian vaults available.
    """
    return run_obsidian(["vaults"]).strip().splitlines()


@tool
def list_obsidian_notes_opened_recently(vault: str | None = None):
    """
    Lists all files the user has opened recently in a given vault in Obsidian.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "recents",
        ]
    ).strip().splitlines()


@tool
def search_obsidian_notes(query: str, vault: str | None = None) -> list[str]:
    """
    Searches Obsidian notes for the given query.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "search",
            f"query={query}",
        ]
    ).strip().splitlines()


@tool
def read_obsidian_note(file_name: str, vault: str | None = None) -> str:
    """
    Reads the content of an Obsidian note by its file name.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return f"# {file_name.split('/')[-1]}\n" + run_obsidian(
        [
            *obsidian_args(vault),
            "read",
            f"path={file_name}",
        ]
    )


@tool
def append_to_obsidian_note(file_name: str, content: str, vault: str | None = None) -> str:
    """
    Appends content to an Obsidian note by its file name.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "append",
            f"path={file_name}",
            f"content={content}",
        ],
        timeout_seconds=60,
    )


@tool
def read_daily_note(vault: str | None = None) -> str:
    """
    Read a user's daily note in a given vault.
    Behaves like read_obsidian_note but without having to know the path.
    Saves calculating the path manually from the data as the format could vary.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "daily:read",
        ]
    )


@tool
def get_daily_note_path(vault: str | None = None) -> str:
    """
    Get the path to a user's daily note. Saves calculating it manually from the data as the format could vary.
    Useful for linking to the note from other notes.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "daily:path",
        ]
    )


@tool
def append_to_daily_note(content: str, vault: str | None = None) -> str:
    """
    Appends content to a user's daily note in a given vault.
    Behaves like append_to_obsidian_note but without having to know the path.
    Saves calculating the path manually from the data as the format could vary.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "daily:append",
            f"content={content}",
        ],
        timeout_seconds=60,
    )


@tool
def view_obsidian_base(file_name: str, vault: str | None = None, format: str = 'md'):
    """
    Views the content of an Obsidian base by its file name.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    Pass optional format parameter to specify the format of the output. Default is md (markdown).
    Available formats: json, csv, tsv, paths
    """
    return run_obsidian(
        [
            *obsidian_args(vault),
            "base:query",
            f"file={file_name}",
            f"format={format}",
        ]
    )

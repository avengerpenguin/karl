import subprocess

from langchain_core.tools import tool


@tool
def list_obsidian_vaults() -> list[str]:
    """
    Lists all Obsidian vaults available.
    """
    result = subprocess.run(
        "obsidian vaults", shell=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


@tool
def list_obsdian_notes_opened_recently(vault: str | None = None):
    """
    Lists all files the user has opened recently in a given vault in Obsidian.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    command_base = f"obsidian vault={vault}" if vault else "obsidian"
    result = subprocess.run(
        f"{command_base} recents", shell=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


@tool
def search_obsidian_notes(query: str, vault: str | None = None) -> list[str]:
    """
    Searches Obsidian notes for the given query.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    command_base = f"obsidian vault={vault}" if vault else "obsidian"
    return subprocess.run(
        f"{command_base} search query={query} ", shell=True, capture_output=True,
        text=True).stdout.strip().splitlines()


@tool
def read_obsidian_note(file_name: str, vault: str | None = None) -> str:
    """
    Reads the content of an Obsidian note by its file name.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    command_base = f"obsidian vault={vault}" if vault else "obsidian"
    return f"# {file_name.split('/')[-1]}\n" + subprocess.run(
        f"{command_base} read path=\"{file_name}\"", shell=True, capture_output=True,
        text=True).stdout


@tool
def append_to_obsidian_note(file_name: str, content: str, vault: str | None = None) -> str:
    """
    Appends content to an Obsidian note by its file name.
    Pass optional vault parameter to limit to a particular vault. Use list_obsidian_vaults to get a list of vaults.
    """
    command_base = f"obsidian vault={vault}" if vault else "obsidian"
    return subprocess.run(
        f"{command_base} append path=\"{file_name}\" content=\"{content}\"", shell=True,
        capture_output=True,
        text=True).stdout

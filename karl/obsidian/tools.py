import subprocess

from langchain_core.tools import tool


@tool
def list_obsidian_vaults() -> list[str]:
    """
    Lists all Obsidian vaults available.
    """
    result = subprocess.run("obsidian vaults", shell=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


@tool
def list_obsdian_notes_opened_recently(vault: str = "private-notes"):
    """
    Lists all files the user has opened recently in a given vault in Obsidian.
    """
    result = subprocess.run(f"obsidian vault={vault} recents", shell=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()


@tool
def search_obsidian_notes(query: str, vault: str = "private-notes") -> list[str]:
    """
    Searches Obsidian notes for the given query.
    """
    return subprocess.run(f"obsidian vault={vault} search query={query} ", shell=True, capture_output=True,
                          text=True).stdout.strip().splitlines()


@tool
def read_obsidian_note(file_name: str, vault: str = "private-notes") -> str:
    """
    Reads the content of an Obsidian note by its file name.
    """
    return f"# {file_name.split('/')[-1]}\n" + subprocess.run(f"obsidian vault={vault} read path=\"{file_name}\"", shell=True, capture_output=True,
                          text=True).stdout


@tool
def append_to_obsidian_note(file_name: str, content: str, vault: str = "private-notes") -> str:
    """
    Appends content to an Obsidian note by its file name.
    """
    return subprocess.run(f"obsidian vault={vault} append path=\"{file_name}\" content=\"{content}\"", shell=True, capture_output=True,
                   text=True).stdout

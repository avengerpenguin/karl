import subprocess

from langchain_core.tools import tool


@tool
def search_obsidian_notes(query: str) -> str:
    """
    Searches Obsidian notes for the given query.
    """
    return subprocess.run(f"obsidian search query={query} vault=private-notes", shell=True, capture_output=True,
                          text=True).stdout


@tool
def read_obsidian_note(file_name: str) -> str:
    """
    Reads the content of an Obsidian note by its file name.
    """
    return f"# {file_name.split('/')[-1]}\n" + subprocess.run(f"obsidian read path=\"{file_name}\" vault=private-notes", shell=True, capture_output=True,
                          text=True).stdout


@tool
def append_to_obsidian_note(file_name: str, content: str) -> str:
    """
    Appends content to an Obsidian note by its file name.
    """
    return subprocess.run(f"obsidian append path=\"{file_name}\" vault=private-notes content=\"{content}\"", shell=True, capture_output=True,
                   text=True).stdout

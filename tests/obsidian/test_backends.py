from textwrap import dedent

from deepagents import create_deep_agent
from deepagents.backends.protocol import GrepResult, GrepMatch, WriteResult
from karl.mlxproxy.chatmodel import MLXChatModel
from karl.obsidian.backends import ObsidianBackend
from langchain.agents import create_agent
from deepagents.middleware import FilesystemMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


def test_ls():
    backend = ObsidianBackend("Test Vault")

    assert '/Welcome.md' in {f['path'] for f in backend.ls("").entries}
    assert '/Welcome.md' in {f['path'] for f in backend.ls("/").entries}
    assert '/Welcome.md' not in {f['path'] for f in backend.ls("/Test Folder/").entries}

def test_read():
    backend = ObsidianBackend("Test Vault")

    assert "Welcome to your new vault" in backend.read("/Welcome.md").file_data['content']
    assert "Welcome to your new vault" in backend.read("Welcome.md").file_data['content']
    assert "Line 2" in backend.read("Welcome.md").file_data['content']
    assert "Welcome to your new vault" in backend.read("Welcome.md").file_data['content'] == dedent("""\
    Welcome to your new vault
    Line 2
    new line here too
    """)
    assert "Welcome to your new vault" not in backend.read("Welcome.md", offset=1).file_data['content']
    assert 'hello' in backend.read("/Test Folder/File in Folder.md").file_data['content']

def test_glob():
    backend = ObsidianBackend("Test Vault")

    assert '/Welcome.md' in {f['path'] for f in backend.glob("*.md").matches}
    assert '/Welcome.md' in {f['path'] for f in backend.glob("*.md", "/").matches}
    assert '/Welcome.md' not in {f['path'] for f in backend.glob("*.md", "/Test Folder/").matches}

    assert '/Welcome.md' in {f['path'] for f in backend.glob("**").matches}
    assert '/Test Folder/File in Folder.md' in {f['path'] for f in backend.glob("**/*.md").matches}
    assert '/Test Folder/File in Folder.md' in {f['path'] for f in backend.glob("Test Folder/*").matches}
    assert '/Test Folder/File in Folder.md' in {f['path'] for f in backend.glob("Test Folder/**").matches}

def test_grep():
    backend = ObsidianBackend("Test Vault")

    assert backend.grep("vault") == GrepResult(matches=[
        GrepMatch(path="/Welcome.md", line=0, text="Welcome to your new vault"),
    ])

def test_grep2():
    backend = ObsidianBackend("Test Vault")

    assert backend.grep("new") == GrepResult(matches=[
        GrepMatch(path="/Welcome.md", line=0, text="Welcome to your new vault"),
        GrepMatch(path="/Welcome.md", line=2, text="new line here too"),
    ])

def test_write():
    backend = ObsidianBackend("Test Vault")
    try:
        assert backend.write("/New File.md", "New file content") == WriteResult(
            path="/New File.md",
        )
        assert backend.read("/New File.md").file_data['content'] == "New file content\n"
    finally:
        assert backend._cli(["delete", "file=New File.md"]) == [
            "Moved to trash: New File.md",
        ]

def test_write_existing_file():
    backend = ObsidianBackend("Test Vault")
    assert backend.write("/New File2.md", "New file content")
    assert backend.write("/New File2.md", "New file content") == WriteResult(
        error="File exists",
    )
    assert backend._cli(["delete", "file=New File2.md"]) == [
        "Moved to trash: New File2.md",
    ]

def test_write_to_new_folder():
    backend = ObsidianBackend("Test Vault")
    try:
        assert backend.write("/New Folder/New File.md", "New file content") == WriteResult(
            path="/New Folder/New File.md",
        )
        assert backend.read("/New Folder/New File.md").file_data['content'] == "New file content\n"
    finally:
        assert backend._cli(["delete", "path=New Folder/New File.md"]) == [
            "Moved to trash: New Folder/New File.md",
        ]


def test_write_file_with_quotes():
    backend = ObsidianBackend("Test Vault")
    assert backend.write("/New File25.md", "New file isn't free from quotes") == WriteResult(
        path="/New File25.md",
    )
    assert backend.read("/New File25.md").file_data['content'] == dedent("""\
        New file isn't free from quotes
    """)
    assert backend._cli(["delete", "file=New File25.md"]) == [
        "Moved to trash: New File25.md",
    ]


def test_edit():
    backend = ObsidianBackend("Test Vault")
    assert backend.write("/New File3.md", dedent("""\
        New file content
        I will edit this line
        And will edit this line and fail the test
    """)) == WriteResult(path="/New File3.md")
    assert backend.edit("/New File3.md", "will edit", "have edited")
    assert backend.read("/New File3.md").file_data['content'] == dedent("""\
        New file content
        I have edited this line
        And will edit this line and fail the test
    """)
    assert backend._cli(["delete","file=New File3.md"]) == [
        "Moved to trash: New File3.md",
    ]

def test_edit_replace_all():
    backend = ObsidianBackend("Test Vault")
    assert backend.write("/New File4.md", "New file content\nI will edit this line\nAnd this line too")
    assert backend.edit("/New File4.md", "this line", "this edited line", replace_all=True)
    assert backend.read("/New File4.md").file_data['content'] == dedent("""\
        New file content
        I will edit this edited line
        And this edited line too
    """)
    assert backend._cli(["delete", "file=New File4.md"]) == [
        "Moved to trash: New File4.md",
    ]

def test_edit_on_non_existent_file():
    backend = ObsidianBackend("Test Vault")
    assert backend.edit("/New File5.md", "", dedent("""\
        New file content but I called edit
    """)) == WriteResult(path="/New File5.md")
    assert backend.read("/New File5.md").file_data['content'] == dedent("""\
        New file content but I called edit
    """)
    assert backend._cli(["delete", "file=New File5.md"]) == [
        "Moved to trash: New File5.md",
    ]


def test_agent_memory(chat_model: str | BaseChatModel):
    memory_system_prompt = dedent("""\
        You are testing a filesystem-backed memory system.

        You have filesystem tools. You must use them.

        Memory rules:
        1. Long-term user facts are stored in /AGENTS.md.
        2. If the user asks you to remember a fact, you must write or edit /AGENTS.md.
        3. If the user asks about a remembered fact, you must read or search /AGENTS.md before answering.
        4. Do not answer "None needed" when the user asks you to remember or retrieve a fact.
        5. Do not rely only on chat history for remembered facts. Use the filesystem memory.
        6. After saving a memory, briefly confirm what was saved.
        7. When retrieving a memory, answer with the remembered value.
    """)

    backend = ObsidianBackend(vault="Test Vault")
    agent1 = create_deep_agent(
        system_prompt=memory_system_prompt,
        model=chat_model,
        memory=["/AGENTS.md"],
        backend=backend,
    )
    response1 = agent1.invoke(
        dict(messages=[HumanMessage("Remember a fact about me: my favourite colour is purple. Save this as a memory.")])
    )
    print(response1['messages'][-1])
    assert "purple" in backend.read("/AGENTS.md").file_data["content"].lower()

    agent2 = create_deep_agent(
        system_prompt=memory_system_prompt,
        model=chat_model,
        memory=["/AGENTS.md"],
        backend=backend,
    )
    response2 = agent2.invoke(dict(messages=[HumanMessage("What is my favourite colour?")]))
    ai_message: AIMessage = response2['messages'][-1]
    assert 'purple' in ai_message.text.lower()
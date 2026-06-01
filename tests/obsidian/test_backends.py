from textwrap import dedent
from deepagents.backends.protocol import GrepResult, GrepMatch, WriteResult
from karl.obsidian.backends import ObsidianBackend
from langchain.agents import create_agent
from deepagents.middleware import FilesystemMiddleware
from langchain_core.messages import HumanMessage, AIMessage


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

def test_gresp():
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
        assert backend._cli("delete 'file=New File.md'") == [
            "Moved to trash: New File.md",
        ]

def test_write_existing_file():
    backend = ObsidianBackend("Test Vault")
    assert backend.write("/New File2.md", "New file content")
    assert backend.write("/New File2.md", "New file content") == WriteResult(
        error="File exists",
    )
    assert backend._cli("delete 'file=New File2.md'") == [
        "Moved to trash: New File2.md",
    ]

def test_write_file_with_quotes():
    backend = ObsidianBackend("Test Vault")
    assert backend.write("/New File25.md", "New file isn't free from quotes") == WriteResult(
        path="/New File25.md",
    )
    assert backend.read("/New File25.md").file_data['content'] == dedent("""\
        New file isn't free from quotes
    """)
    assert backend._cli("delete 'file=New File25.md'") == [
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
    assert backend._cli("delete 'file=New File3.md'") == [
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
    assert backend._cli("delete 'file=New File4.md'") == [
        "Moved to trash: New File4.md",
    ]

def test_agent_memory():
    agent1 = create_agent(
        model="ollama:gemma4:31b",
        middleware=[
            FilesystemMiddleware(
                system_prompt="Use the filesystem to store memories and information about the user",
                backend=ObsidianBackend(vault="Test Vault"),
            ),
        ]
    )
    response: AIMessage = agent1.invoke(
        dict(messages=[HumanMessage("Remember a fact about me: my favourite colour is purple. Save this as a memory.")])
    )
    # tool_call: AIMessage = response['messages'][-3]
    # assert 'purple' in tool_call.tool_calls[0]['args']['content']
    agent2 = create_agent(
        model="ollama:gemma4:31b",
        middleware=[
            FilesystemMiddleware(
                system_prompt="Use the filesystem to retrieve and store memories and information about the user",
                backend=ObsidianBackend(vault="Test Vault"),
            ),
        ]
    )
    response = agent2.invoke(dict(messages=[HumanMessage("Check your filesystem memories. What is my favourite colour?")]))
    ai_message: AIMessage = response['messages'][-1]
    assert 'purple' in ai_message.text.lower()

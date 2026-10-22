# Karl: General AI Assistant Toolkit

Collection of tools on top of [`deepagents`](https://github.com/langchain-ai/deepagents) that turn LangChain/LangGraph agents into a personal assistant wired into the tools you already use every day.

## Key Features

The primary things being actively developed are:

- Matrix as a UI layer. The "general" agent runs as a bot that connects to a home server and opens a DM room with you.
- Matrix threads for context. The Matrix thread ID is used to allow for distinct context windows.
- Approve/deny tools via chat. Human-in-the-loop support is done via Matrix interactions.
- Tool wrappers for multiple use cases, migrating things out to a general MCP config loader if services offer that instead.
- Emerging support for wrapping MLX on Mac to allow for use of local models more (work in progress)

## Tools

- **Email** — list folders, search, fetch and archive messages over IMAP
- **Calendar** — read and manage Google Calendar events
- **Jira** — read, create, update, assign, comment on and transition issues
- **Confluence** — search spaces, read pages and comments, update content
- **GitLab** — merge requests (created / assigned / review requested), diffs, CI pipelines and job logs
- **Slack** — read and post messages
- **Obsidian** — vaults, recent notes, search, read/append notes and daily notes
- **LinkedIn** — find unanswered chats and past reply examples for drafting
- **Google** — Search Console, AdSense and Analytics data
- **SQL** — query configured databases
- **Web** — web search and URL fetching
- **SQL** — query configured databases
- **Web** — web search and URL fetching
- **Custom MCP** — bring your own MCP servers for anything not covered above

- **Web** — web search and URL fetching
- **Custom MCP** — bring your own MCP servers for anything not covered above

Knowledge is persisted on a filesystem backend (local shell or an Obsidian vault) as
markdown with wikilinks and YAML frontmatter, so agents can share context across sessions.

## Agents

Agents are just set configurations being experimented with. The "general" is the one currently in the most use, but
there's some attempt to make specialised ones too.

The general agent is

## Installation

For everything:

```shell
pip install -e '.[imap,beeper,todoist,gitlab,jira,confluence,tavily,matrix,mysql,google]'
```

Then run:

```shell
python -m karl --help
```

Most commonly run thing right now is:

```shell
python -m karl bot
```

Note this requires a lot of secrets in the env. Recommended path is to install direnv and create a `.envrc.local` file for all secrets.

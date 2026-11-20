import json
import re
import sys

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def messages_to_markdown(messages):
    """Serializes a list of LangChain BaseMessages into clean Markdown."""
    sections = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            role = "User"

        content = m.content.strip()
        if isinstance(m, AIMessage):
            content = strip_thinking_prefix(content)
        sections.append(f"### {role}\n\n{content}")

    return "\n\n---\n\n".join(sections)


def markdown_to_messages(md_text):
    """Parses a Markdown transcript back into a list of LangChain messages."""
    messages = []
    # Split blocks by the ### role header
    parts = re.split(r"\n*---\n*|\n+(?=###\s)", md_text.strip())

    for part in parts:
        part = part.strip()
        if not part.startswith("###"):
            continue

        lines = part.split("\n", 1)
        role = lines[0].replace("###", "").strip().lower()
        content = lines[1].strip() if len(lines) > 1 else ""

        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))

    return messages


def load_messages_from_stdin(system_prompt: str, raw_input: str) -> list:
    # Check if the input looks like our structured Markdown format
    if raw_input.strip().startswith("###"):
        messages = markdown_to_messages(raw_input)
        # Optional safeguard: Ensure a system prompt is present, or prepend one
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=system_prompt))
        return messages
    else:
        # Fallback: Treat as a raw initial user input
        return [SystemMessage(content=system_prompt), HumanMessage(content=raw_input)]


def _content_to_stdout_text(content) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")

                if block_type in {"text", "text_delta"}:
                    parts.append(block.get("text", ""))
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])

        return "".join(parts)

    return str(content)


def _message_stdout_text(message) -> str:
    content_blocks = getattr(message, "content_blocks", None)

    if content_blocks:
        return _content_to_stdout_text(content_blocks)

    return _content_to_stdout_text(getattr(message, "content", ""))


def _log_tool_activity(message) -> bool:
    logged_tool_activity = False
    content_blocks = getattr(message, "content_blocks", None) or []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        if block.get("type") not in {"tool_call", "tool_call_chunk"}:
            continue

        logged_tool_activity = True
        name = (
            block.get("name")
            or block.get("tool_name")
            or block.get("id")
            or "unknown_tool"
        )
        args = block.get("args") or block.get("args_chunk") or {}

        print(
            f"[tool call] {name} {json.dumps(args, ensure_ascii=False) if args else ''}",
            file=sys.stderr,
            flush=True,
        )

    for tool_call in getattr(message, "tool_calls", None) or []:
        logged_tool_activity = True
        name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args", {})

        print(
            f"[tool call] {name} {json.dumps(args, ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )

    for tool_call_chunk in getattr(message, "tool_call_chunks", None) or []:
        logged_tool_activity = True
        name = (
            tool_call_chunk.get("name") or tool_call_chunk.get("id") or "unknown_tool"
        )
        args = tool_call_chunk.get("args") or {}

        print(
            f"[tool call chunk] {name} {args}",
            file=sys.stderr,
            flush=True,
        )

    if getattr(message, "type", None) == "tool":
        logged_tool_activity = True
        name = (
            getattr(message, "name", None)
            or getattr(message, "tool_call_id", None)
            or "tool"
        )
        preview = _content_to_stdout_text(getattr(message, "content", ""))
        preview = preview.replace("\n", "\\n")[:500]

        print(
            f"[tool result] {name}: {preview}",
            file=sys.stderr,
            flush=True,
        )

    return logged_tool_activity


def strip_thinking_prefix(message: str) -> str:
    # 1. Handle Qwen style blocks
    if "</think>" in message:
        return message.split("</think>")[-1].lstrip()

    # 2. Handle Gemma 4 style blocks
    # If the raw tag <channel|> or its literal text counterpart is visible
    if "<channel|>" in message:
        return message.split("<channel|>")[-1].lstrip()

    if "</seed:think>" in message:
        return message.split("</seed:think>")[-1].lstrip()

    # Fallback for Gemma 4 if the closing tag is eaten by the tokenizer,
    # but the opening structural block remains at the start of the text.
    if message.startswith("<|channel>thought"):
        # If we can't find a clean tag but know it's a raw un-stripped stream,
        # we look for where the structural prefix breaks.
        # Gemma 4 typically transitions into standard text with a double newline or space.
        pass

    return message


def stream_model_to_stdout(agent, messages: LanguageModelInput) -> str:
    output_parts = []

    for event in agent.stream(messages, stream_mode="messages"):
        if isinstance(event, tuple) and len(event) == 2:
            message, _metadata = event
        else:
            message = event

        if getattr(message, "type", None) == "tool" or _log_tool_activity(message):
            continue

        # Check the custom metadata flag emitted by MLXChatModel
        metadata = getattr(message, "response_metadata", {}) or {}
        if metadata.get("is_reasoning") is True:
            continue  # Silently consume the thought block without printing it

        text = _message_stdout_text(message)
        if not text:
            continue

        # Catch and drop any lingering literal string tags from the stdout stream
        clean_text = (
            text.replace("<|channel>thought", "")
            .replace("<think>", "")
            .replace("</think>", "")
        )
        if not clean_text.strip() and not text.endswith("\n"):
            continue

        print(clean_text, end="", flush=True, file=sys.stderr)
        output_parts.append(clean_text)

    if output_parts:
        print(flush=True, file=sys.stderr)

    return "".join(output_parts)

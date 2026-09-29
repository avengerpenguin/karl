import json
import os
import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

try:
    from mlx_lm import generate, load, stream_generate
except ImportError:
    raise ImportError("Please install karl[mlx] to use MLX proxy server")


DEFAULT_MLX_CHAT_MODEL = os.getenv(
    "MLX_DEFAULT_MODEL",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
)


def _tool_system_prompt(
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | bool | None,
) -> str:
    tool_names = [
        tool.get("function", {}).get("name")
        for tool in tools
        if tool.get("function", {}).get("name")
    ]

    if tool_choice in (None, False, "auto"):
        choice_instruction = "Use a tool only when it is necessary to answer the user."
    elif tool_choice in (True, "required", "any"):
        choice_instruction = "You must call one of the available tools."
    elif isinstance(tool_choice, str):
        choice_instruction = f"You must call the `{tool_choice}` tool."
    elif isinstance(tool_choice, dict):
        choice_instruction = (
            f"Follow this tool choice constraint: {json.dumps(tool_choice)}"
        )
    else:
        choice_instruction = "Use a tool only when it is necessary to answer the user."

    return (
        "You have access to tools.\n"
        f"{choice_instruction}\n\n"
        "When calling tools, respond with JSON only, using exactly this shape:\n"
        '{"tool_calls":[{"name":"tool_name","args":{"arg_name":"value"}}]}\n\n'
        "Do not wrap the JSON in Markdown.\n"
        "Do not invent tool names.\n"
        f"Available tool names: {', '.join(tool_names)}\n\n"
        "Tool schemas:\n"
        f"{json.dumps(tools, ensure_ascii=False, indent=2)}"
    )


def _plain_transcript_prompt(messages: list[dict[str, Any]]) -> str:
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    return f"{transcript}\nassistant:"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])

        return "\n".join(parts)

    return str(content)


def _to_mlx_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.type
        content = _content_to_text(message.content)

        if isinstance(message, AIMessage) and message.tool_calls:
            content = content.rstrip()
            tool_json = json.dumps(
                {"tool_calls": message.tool_calls},
                ensure_ascii=False,
            )
            content = f"{content}\n{tool_json}" if content else tool_json

        if isinstance(message, ToolMessage):
            content = json.dumps(
                {
                    "tool_call_id": message.tool_call_id,
                    "content": content,
                },
                ensure_ascii=False,
            )

        converted.append(
            {
                "role": role,
                "content": content,
            }
        )

    return converted


def _extract_json(text: str) -> Any | None:
    stripped = text.strip()

    if stripped.startswith("```"):
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.DOTALL,
        )
        if match:
            stripped = match.group(1).strip()

    candidates = [stripped]

    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    array_match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
    if array_match:
        candidates.append(array_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _apply_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text

    earliest_stop_index: int | None = None

    for stop_sequence in stop:
        index = text.find(stop_sequence)
        if index == -1:
            continue

        if earliest_stop_index is None or index < earliest_stop_index:
            earliest_stop_index = index

    if earliest_stop_index is None:
        return text

    return text[:earliest_stop_index]


def _stream_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    return str(response)


class MLXChatModel(BaseChatModel):
    """LangChain chat model wrapper for local in-process mlx-lm models.

    This class implements the pieces LangChain agents expect from a `BaseChatModel`,
    including `bind_tools()`. Tool support is prompt-and-parse based because mlx-lm
    itself does not expose an OpenAI-style tool-calling API.
    """

    model_id: str = DEFAULT_MLX_CHAT_MODEL
    max_tokens: int = 512
    temperature: float = 0.0
    verbose: bool = False

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._model, self._tokenizer = load(self.model_id)

    @property
    def _llm_type(self) -> str:
        return "mlx-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load(self.model_id)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools in the shape expected by LangChain agents.

        `create_agent()` calls this method when tools are provided. Provider-backed
        chat models send these schemas to the provider. For mlx-lm, we instead pass
        the converted schemas through runnable kwargs and inject them into the prompt
        during `_generate()`.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        return self.bind(
            tools=formatted_tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_loaded()

        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        prompt = self._messages_to_prompt(
            messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        generation_kwargs = self._generation_kwargs(kwargs)

        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **generation_kwargs,
        )

        response_text = _apply_stop(response_text, stop)
        tool_calls = self._parse_tool_calls(response_text, tools)

        if tool_calls:
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            message = AIMessage(content=response_text)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={
                        "model": self.model_id,
                    },
                )
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        self._ensure_loaded()

        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        # Tool calls need to be parsed from a complete response, so streaming is only
        # token-streamed for ordinary chat responses. If tools are bound, fall back to
        # `_generate()` and emit the final message as one chunk.
        if tools:
            result = self._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            message = result.generations[0].message
            chunk = AIMessageChunk(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                response_metadata=message.response_metadata,
            )
            yield ChatGenerationChunk(message=chunk)
            return

        prompt = self._messages_to_prompt(messages)
        generation_kwargs = self._generation_kwargs(kwargs)

        emitted = ""

        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **generation_kwargs,
        ):
            token_text = _stream_response_text(response)

            if not token_text:
                continue

            emitted += token_text

            if stop:
                stopped = _apply_stop(emitted, stop)
                token_text = stopped[len(emitted) - len(token_text) :]
                emitted = stopped

                if not token_text:
                    break

            if run_manager is not None:
                run_manager.on_llm_new_token(token_text)

            yield ChatGenerationChunk(
                message=AIMessageChunk(content=token_text),
            )

            if stop and emitted != _apply_stop(emitted + "x", stop)[:-1]:
                break

    def _generation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        verbose = kwargs.pop("verbose", self.verbose)

        # Consume unsupported sampling aliases so they cannot leak into mlx-lm.
        kwargs.pop("temperature", None)
        kwargs.pop("temp", None)

        # Ignore common provider/LangChain kwargs that mlx-lm does not accept.
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("response_format", None)

        generation_kwargs = {
            "max_tokens": max_tokens,
            "verbose": verbose,
        }

        # Only forward known mlx-lm kwargs if you intentionally support them.
        for key in (
            "sampler",
            "logits_processors",
            "max_kv_size",
            "prompt_cache",
            "prefill_step_size",
        ):
            if key in kwargs:
                generation_kwargs[key] = kwargs.pop(key)

        return generation_kwargs

    def _messages_to_prompt(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | bool | None = None,
    ) -> str:
        mlx_messages = _to_mlx_messages(messages)

        if tools:
            mlx_messages = [
                {
                    "role": "system",
                    "content": _tool_system_prompt(tools, tool_choice),
                },
                *mlx_messages,
            ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    mlx_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Some tokenizer templates are strict about supported roles, especially
                # `tool`. Fall back to a plain transcript if the template rejects them.
                pass

        return _plain_transcript_prompt(mlx_messages)

    def _parse_tool_calls(
        self,
        text: str,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if not tools:
            return []

        allowed_tool_names = {
            tool.get("function", {}).get("name")
            for tool in tools
            if tool.get("function", {}).get("name")
        }

        parsed = _extract_json(text)
        if not parsed:
            return []

        raw_calls: Any

        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            raw_calls = parsed["tool_calls"]
        elif isinstance(parsed, dict) and parsed.get("name"):
            raw_calls = [parsed]
        elif isinstance(parsed, list):
            raw_calls = parsed
        else:
            return []

        tool_calls: list[dict[str, Any]] = []

        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue

            name = raw_call.get("name")
            args = (
                raw_call.get("args")
                if "args" in raw_call
                else raw_call.get("arguments", {})
            )

            if name not in allowed_tool_names:
                continue

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}

            if args is None:
                args = {}

            if not isinstance(args, dict):
                args = {"input": args}

            tool_calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": raw_call.get("id") or f"call_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            )

        return tool_calls

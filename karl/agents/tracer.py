import logging
from langchain_core.callbacks import AsyncCallbackHandler

logger = logging.getLogger("karl.trace")


class ConsoleTraceHandler(AsyncCallbackHandler):
    def _log(self, event: str, **fields: object) -> None:
        rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
        logger.info("[trace] %s %s", event, rendered)

    async def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        self._log(
            "chat_model_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            model=serialized.get("name") if isinstance(serialized, dict) else None,
            tags=tags,
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id,
        parent_run_id=None,
        chunk=None,
        **kwargs,
    ):
        # Avoid dumping whole responses if noisy.
        self._log(
            "llm_token",
            run_id=run_id,
            parent_run_id=parent_run_id,
            token_preview=token[:80],
        )

    async def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._log("llm_end", run_id=run_id, parent_run_id=parent_run_id)

    async def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._log(
            "llm_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=repr(error),
        )

    async def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        name=None,
        **kwargs,
    ):
        self._log(
            "tool_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            input_preview=str(input_str)[:300],
        )

    async def on_tool_end(
        self, output, *, run_id, parent_run_id=None, name=None, **kwargs
    ):
        self._log(
            "tool_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            output_preview=str(output)[:300],
        )

    async def on_tool_error(
        self, error, *, run_id, parent_run_id=None, name=None, **kwargs
    ):
        self._log(
            "tool_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            error=repr(error),
        )

    async def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        self._log(
            "chain_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized.get("name") if isinstance(serialized, dict) else None,
            tags=tags,
        )

    async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._log("chain_end", run_id=run_id, parent_run_id=parent_run_id)

    async def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._log(
            "chain_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=repr(error),
        )

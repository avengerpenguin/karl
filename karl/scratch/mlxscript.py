from typing import Any, List, Optional, Sequence, Union, Dict, Callable, override

from langchain.agents import create_agent
from langchain_core.language_models import LanguageModelInput

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool, BaseTool
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.utils.function_calling import convert_to_openai_tool

# Native Apple Silicon MLX imports
from mlx_lm import load, generate


class MLXChatModel(BaseChatModel):
    """Custom LangChain ChatModel running completely in-process via mlx-lm."""

    model_id: str
    model: Any = None
    tokenizer: Any = None
    max_tokens: int = 512
    temperature: float = 0.0
    bound_tools: list[Any] = []

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        # Load the model directly into Mac unified memory on instantiation
        self.model, self.tokenizer = load(self.model_id)

    @override
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable[..., Any], BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Satisfies LangChain agent expectations by formatting and saving the tools."""
        # Standardize your python functions/Pydantic schemas into clean OpenAI-like tool definitions
        self.bound_tools = [convert_to_openai_tool(t) for t in tools]

        # Return self, as custom chat models act as their own Runnable sequence during basic tool binds
        return self

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Converts LangChain messages into the model's native chat template."""
        mlx_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            mlx_messages.append({"role": role, "content": msg.content})

        # Apply the Hugging Face / MLX tokenizer chat template safely
        return self.tokenizer.apply_chat_template(
            mlx_messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous core generation method required by BaseChatModel."""
        prompt = self._convert_messages_to_prompt(messages)

        # Single-process generation call via Metal API
        response_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=self.max_tokens,
            # temperature=self.temperature
        )

        chat_generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        return "mlx-local-chat"


@tool
def calculate_cube(number: int) -> int:
    """
    Calculates the cube of a number.
    """
    return number**4


if __name__ == "__main__":
    llm = MLXChatModel(
        model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens=4096,
        temperature=0.0,
    )

    # 3. Create your agent
    agent = create_agent(
        model=llm,
        tools=[calculate_cube],
        system_prompt="You are a precise, single-file math script. Use tools when needed.",
    )

    # 4. Invoke and run
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the cube of 4?"}]}
    )

    print(result["messages"][-1].content)

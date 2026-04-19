import multiprocessing
import os

from langchain.agents import create_agent
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import AIMessageChunk, AnyMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from mcp.client.auth import OAuthClientProvider

# llm = ChatLlamaCpp(
#     temperature=0.5,
#     model_path="./models//Hermes-2-Pro-Llama-3-8B-F16.gguf",
#     n_ctx=10000,
#     n_gpu_layers=8,
#     n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#     max_tokens=512,
#     n_threads=multiprocessing.cpu_count() - 1,
#     repeat_penalty=1.5,
#     top_p=0.5,
#     verbose=True,
# )


client = MultiServerMCPClient(
    {
        "messages": {
            "transport": "http",
            "url": "http://localhost:23373/v0/mcp",
            "headers": {
                "Authorization": f"Bearer {os.getenv('BEEPER_TOKEN')}",
            }
        },
    }
)

async def create(model="deepseek-r1:32b", tools=None):
    llm = ChatOllama(model=model)
    if not tools:
        tools = await client.get_tools()
    return create_agent(model=llm, tools=tools)


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)
    # N.B. all content is available through token.content_blocks


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

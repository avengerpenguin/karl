import re
from html import unescape

import httpx
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk
from langchain_ollama import ChatOllama

from . import _render_message_chunk, _render_completed_message

CV_URL = "https://rossfenning.co.uk/cv.pdf"
CV_TTL_URL = (
    "https://raw.githubusercontent.com/avengerpenguin/rossfenning.co.uk/main/content/extra/cv.ttl"
)

def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<nav.*?>.*?</nav>", " ", html)
    html = re.sub(r"(?is)<footer.*?>.*?</footer>", " ", html)
    html = re.sub(r"(?is)<header.*?>.*?</header>", " ", html)
    html = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"[ \t\r\f\v]+", " ", html)
    html = re.sub(r"\n\s+\n", "\n\n", html)
    return html.strip()


async def _fetch_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; karl/0.1; +https://rossfenning.co.uk)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "html" in content_type or "<html" in response.text.lower():
            return _strip_html(response.text)
        return response.text.strip()


async def review_job_ad(url: str, model: str = "MFDoom/deepseek-r1-tool-calling:8b") -> None:
    job_ad = await _fetch_text(url)
    cv_ttl = await _fetch_text(CV_TTL_URL)
    llm = ChatOllama(model=model)
    agent = create_agent(model=llm)

    messages = [
        SystemMessage(
            content=(
                "You are a CV tailoring assistant.\n"
                "Given a job advert and a CV, produce practical advice to improve the CV "
                "for this specific role.\n\n"
                "Return:\n"
                "1. A concise summary of the role\n"
                "2. Specific suggested CV edits, grouped by section if possible\n"
                "3. Missing keywords/skills to consider adding only if truthful\n"
                "4. Suggested learning/upskilling areas\n"
                "5. Any red flags or gaps\n\n"
                "Be careful not to invent experience. Prefer edits that improve relevance, "
                "wording, and emphasis over fabrication."
                "At the end, suggest a unified diff for the Turtle RDF file for the CV.\n"
            )
        ),
        HumanMessage(
            content=(
                f"Job advert URL: {url}\n\n"
                f"Job advert content:\n{job_ad}\n\n"
                f"Candidate CV PDF: {CV_URL}\n\n"
                f"Current CV Turtle content:\n{cv_ttl}\n\n"
                "Return a unified diff patch for content/extra/cv.ttl only."
            )
        ),
    ]

    async for chunk in agent.astream(
            {"messages": messages},
            stream_mode=["updates", "messages"],
            version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):  # `source` captures node name
                    _render_completed_message(update["messages"][-1])


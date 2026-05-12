from io import BytesIO

import requests
from langchain_core.tools import tool
from pypdf import PdfReader


@tool
def fetch_url(url: str) -> str:
    """
    Fetches the content of a URL. Useful for quickly checking the content of a website or PDF given a URL.
    If the URL is a website, the raw HTML will be returned.
    If the URL is a PDF, the text content of it will be parsed and returned.
    """
    r = requests.get(url)
    r.raise_for_status()
    if url.endswith(".pdf"):
        reader = PdfReader(BytesIO(r.content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    else:
        return r.text

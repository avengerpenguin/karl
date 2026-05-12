from langchain_core.tools import tool
from .http import fetch_url


@tool
def fetch_cv() -> str:
    """
    Fetches the user's CV. Useful for getting a quick overview of their work experience when evaluating emails
    from recruiters.

    Returns the text content of the CV parsed from the PDF.
    """
    return fetch_url.invoke("https://rossfenning.co.uk/cv.pdf")

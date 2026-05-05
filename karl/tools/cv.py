from io import BytesIO

import requests
from langchain_core.tools import tool
from pypdf import PdfReader


@tool
def fetch_cv() -> bytes | str:
    """
    Fetches the user's CV. Useful for getting a quick overview of their work experience when evaluating emails
    from recruiters.

    Returns the bytes of a CV in PDF format or an error message if the request fails.
    """
    r = requests.get("https://rossfenning.co.uk/cv.pdf")
    if r.ok:
        reader = PdfReader(BytesIO(r.content))
        return "\n\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

    return f"Error fetching CV: {r.status_code}\n{r.text}"

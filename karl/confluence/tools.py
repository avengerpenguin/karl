import os


try:
    from atlassian.confluence import Confluence
except ImportError:
    raise ImportError("Please install karl[confluence] to use Confluence tools")

import requests

session = requests.Session()
confluence: Confluence = Confluence(
    url=os.getenv("ATLASSIAN_BASE_URL"),
    username=os.getenv("ATLASSIAN_USER"),
    password=os.getenv("ATLASSIAN_API_TOKEN"),
    session=session,
)

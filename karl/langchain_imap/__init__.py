from importlib import metadata

from langchain_imap.retrievers import ImapRetriever
from langchain_imap.utils import ImapConfig

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ImapRetriever",
    "ImapConfig",
    "__version__",
]

"""Langchain tool wrappers around an OpenSearch cluster.

Configuration (environment):
    OPENSEARCH_URL       required, e.g. https://user:pass@opensearch.internal:9200
    OPENSEARCH_USER      optional, overrides any user in the URL
    OPENSEARCH_PASSWORD  optional, overrides any password in the URL
    OPENSEARCH_VERIFY_CERTS  optional, "false" to disable TLS verification (dev only)
    OPENSEARCH_TIMEOUT_SECONDS  optional, default 30
"""

import json
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit, unquote

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from opensearchpy import OpenSearch
except ImportError:
    raise ImportError("Please `pip install opensearch-py` to use OpenSearch tools")


URL_ENV_VAR = "OPENSEARCH_URL"
DEFAULT_SIZE = 10
HARD_MAX_SIZE = 100
MAX_FIELD_CHARS = 500
TIMEOUT_SECONDS = int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "30"))

# Query clauses that read data only - anything else is rejected before it is sent.
_FORBIDDEN_KEYS = {"script", "script_score", "script_fields", "scripted_metric"}


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _client() -> OpenSearch:
    url = os.getenv(URL_ENV_VAR, "").strip()
    if not url:
        raise ValueError(
            f"{URL_ENV_VAR} is not set. Export it, e.g. "
            f"{URL_ENV_VAR}=https://opensearch.internal:9200"
        )

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise ValueError(
            f"{URL_ENV_VAR} must be an http(s) URL, got scheme {parts.scheme!r}."
        )

    user = os.getenv("OPENSEARCH_USER") or unquote(parts.username or "")
    password = os.getenv("OPENSEARCH_PASSWORD") or unquote(parts.password or "")

    return OpenSearch(
        hosts=[
            {
                "host": parts.hostname or "localhost",
                "port": parts.port or (443 if parts.scheme == "https" else 9200),
            }
        ],
        http_auth=(user, password) if user else None,
        use_ssl=parts.scheme == "https",
        verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() != "false",
        ssl_show_warn=False,
        timeout=TIMEOUT_SECONDS,
        max_retries=2,
        retry_on_timeout=True,
    )


# -----------------------------------------------------------------------------
# Guards and coercion
# -----------------------------------------------------------------------------


def _assert_read_only(body: Any, path: str = "query") -> None:
    """Reject scripted clauses, which can execute arbitrary code on the cluster."""
    if isinstance(body, dict):
        for key, value in body.items():
            if key in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"{path}.{key} is not permitted - scripted queries are disabled."
                )
            _assert_read_only(value, f"{path}.{key}")
    elif isinstance(body, list):
        for index, value in enumerate(body):
            _assert_read_only(value, f"{path}[{index}]")


def _parse_query(query: str | dict | None) -> dict:
    if not query:
        return {"match_all": {}}
    if isinstance(query, str):
        try:
            query = json.loads(query)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"`query` must be a JSON object describing an OpenSearch query clause, "
                f'e.g. {{"match": {{"message": "timeout"}}}}. Parse error: {exc}'
            ) from exc
    if not isinstance(query, dict):
        raise ValueError("`query` must be a JSON object, not a list or scalar.")

    # Tolerate the model wrapping the clause in a full search body.
    if set(query) == {"query"} and isinstance(query["query"], dict):
        query = query["query"]

    _assert_read_only(query)
    return query


def _truncate(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS] + f"...[+{len(value) - MAX_FIELD_CHARS} chars]"
    return value


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class SearchHit(BaseModel):
    index: str
    id: str
    score: float | None = None
    source: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    index: str
    total_hits: int
    total_is_lower_bound: bool = False
    returned: int = 0
    took_ms: int = 0
    timed_out: bool = False
    hits: list[SearchHit] = Field(default_factory=list)
    aggregations: dict[str, Any] | None = None
    note: str | None = None


class IndexSummary(BaseModel):
    name: str
    health: str | None = None
    status: str | None = None
    docs_count: int | None = None
    store_size: str | None = None


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


@tool
def list_opensearch_indices(name_pattern: str = "*") -> list[IndexSummary]:
    """
    Lists the indices available in OpenSearch, with document counts and size.
    Call this first to discover the valid `index` argument for the other OpenSearch tools.
    Set name_pattern to a wildcard pattern such as "logs-*" to narrow the list (default: all).
    System indices (those starting with ".") are omitted.
    """
    rows = _client().cat.indices(
        index=name_pattern, format="json", h="health,status,index,docs.count,store.size"
    )
    return [
        IndexSummary(
            name=row["index"],
            health=row.get("health"),
            status=row.get("status"),
            docs_count=int(row["docs.count"]) if row.get("docs.count") else None,
            store_size=row.get("store.size"),
        )
        for row in sorted(rows, key=lambda r: r["index"])
        if not row["index"].startswith(".")
    ]


@tool
def describe_opensearch_index(index: str) -> dict[str, Any]:
    """
    Returns the field mapping for an index, so you know which fields exist and what
    their types are before writing a query.
    Set index to a name from list_opensearch_indices (wildcards and aliases are allowed).

    Field types matter: `keyword` fields need exact-value `term` queries, `text` fields
    need analysed `match` queries, and only `date`/numeric fields support `range` queries.
    """
    mappings = _client().indices.get_mapping(index=index)

    fields: dict[str, dict[str, str]] = {}

    def walk(properties: dict[str, Any], prefix: str = "") -> None:
        for name, definition in properties.items():
            path = f"{prefix}{name}"
            if "properties" in definition:
                walk(definition["properties"], f"{path}.")
            else:
                fields.setdefault(path, {"type": definition.get("type", "object")})

    for mapping in mappings.values():
        walk(mapping.get("mappings", {}).get("properties", {}))

    return {
        "index": index,
        "resolved_indices": sorted(mappings),
        "field_count": len(fields),
        "fields": dict(sorted(fields.items())),
    }


@tool
def search_opensearch(
    index: str,
    query: str | dict | None = None,
    size: int = DEFAULT_SIZE,
    fields: list[str] | None = None,
    sort: str | None = None,
) -> SearchResult:
    """
    Runs a read-only OpenSearch query and returns the matching documents.

    Set index to an index name, alias or wildcard pattern from list_opensearch_indices.
    Set query to an OpenSearch Query DSL clause as JSON - the part that goes inside
    "query" - for example:
        {"match": {"message": "connection refused"}}
        {"term": {"status.keyword": "ERROR"}}
        {"bool": {"must": [{"match": {"service": "checkout"}}],
                  "filter": [{"range": {"@timestamp": {"gte": "now-24h"}}}]}}
    Omit query to match everything (useful for eyeballing a few sample documents).
    Set size to cap how many documents come back (default 10, maximum 100).
    Set fields to a list of source fields to return, e.g. ["@timestamp", "message"] -
    do this whenever you can, documents can be very large.
    Set sort to "field:asc" or "field:desc", e.g. "@timestamp:desc" for newest first.

    Notes:
    - Scripted queries are rejected; this tool only reads.
    - `total_hits` is how many documents matched, which may be far more than `returned`.
      Refine the query or aggregate rather than paging through everything.
    - Long string values are truncated, marked with "...[+N chars]".
    """
    size = max(0, min(int(size), HARD_MAX_SIZE))

    body: dict[str, Any] = {"query": _parse_query(query), "track_total_hits": True}
    if fields:
        body["_source"] = fields

    response = _client().search(
        index=index,
        body=body,
        size=size,
        sort=sort or None,
        timeout=f"{TIMEOUT_SECONDS}s",
    )

    total = response["hits"]["total"]
    hits = [
        SearchHit(
            index=hit["_index"],
            id=hit["_id"],
            score=hit.get("_score"),
            source=_truncate(hit.get("_source", {})),
        )
        for hit in response["hits"]["hits"]
    ]

    note = None
    if total["value"] > len(hits):
        note = (
            f"{total['value']} documents matched but only {len(hits)} are shown. "
            f"Tighten the query, or use aggregate_opensearch to summarise instead."
        )

    return SearchResult(
        index=index,
        total_hits=total["value"],
        total_is_lower_bound=total.get("relation") == "gte",
        returned=len(hits),
        took_ms=response.get("took", 0),
        timed_out=response.get("timed_out", False),
        hits=hits,
        note=note,
    )


async def get_opensearch_tools():
    return [
        list_opensearch_indices,
        describe_opensearch_index,
        search_opensearch,
    ]

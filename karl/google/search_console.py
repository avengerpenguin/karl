"""Read-only Google Search Console tools backed by Application Default Credentials."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import google.auth
from googleapiclient.discovery import build
from langchain_core.tools import BaseTool, tool


SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
MAX_ROWS = 25_000

Dimension = Literal[
    "query",
    "page",
    "country",
    "device",
    "searchAppearance",
    "date",
]
SearchType = Literal["web", "image", "video", "news", "discover", "googleNews"]


def _service():
    """Build a Search Console client from ADC for one tool invocation."""
    credentials, _project = google.auth.default(scopes=SCOPES)
    return build(
        "searchconsole",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )


def _iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format; got {value!r}.") from exc


def _filter_expression(dimension: str, value: str) -> dict[str, str]:
    if value.startswith("regex:"):
        return {
            "dimension": dimension,
            "operator": "includingRegex",
            "expression": value.removeprefix("regex:"),
        }

    return {
        "dimension": dimension,
        "operator": "contains",
        "expression": value,
    }


@tool
def gsc_list_sites() -> dict[str, Any]:
    """List all Google Search Console properties available to the current credentials."""
    return _service().sites().list().execute()


@tool
def gsc_search_analytics(
    site_url: str,
    start_date: str,
    end_date: str,
    dimensions: list[Dimension] | None = None,
    row_limit: int = 100,
    search_type: SearchType = "web",
    query_filter: str | None = None,
    page_filter: str | None = None,
    country_filter: str | None = None,
    device_filter: Literal["DESKTOP", "MOBILE", "TABLET"] | None = None,
) -> dict[str, Any]:
    """Query Search Console clicks, impressions, CTR, and average position.

    Use the exact property URL returned by gsc_list_sites. Dates must be YYYY-MM-DD.
    Dimensions may include query, page, country, device, searchAppearance, and date.
    query_filter and page_filter use substring matching unless prefixed with ``regex:``.
    Results are capped at 25,000 rows and recent Search Console data may be incomplete.
    """
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    if not 1 <= row_limit <= MAX_ROWS:
        raise ValueError(f"row_limit must be between 1 and {MAX_ROWS}.")

    expressions: list[dict[str, str]] = []
    if query_filter:
        expressions.append(_filter_expression("query", query_filter))
    if page_filter:
        expressions.append(_filter_expression("page", page_filter))
    if country_filter:
        expressions.append(
            {
                "dimension": "country",
                "operator": "equals",
                "expression": country_filter.upper(),
            }
        )
    if device_filter:
        expressions.append(
            {
                "dimension": "device",
                "operator": "equals",
                "expression": device_filter,
            }
        )

    request_body: dict[str, Any] = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "type": search_type,
        "rowLimit": row_limit,
        "dimensions": dimensions or [],
    }
    if expressions:
        request_body["dimensionFilterGroups"] = [
            {"groupType": "and", "filters": expressions}
        ]

    response = (
        _service()
        .searchanalytics()
        .query(siteUrl=site_url, body=request_body)
        .execute()
    )
    rows = response.get("rows", [])
    response["resultMetadata"] = {
        "returnedRows": len(rows),
        "rowLimit": row_limit,
        "mayBeTruncated": len(rows) == row_limit,
    }
    return response


@tool
def gsc_inspect_url(site_url: str, inspection_url: str) -> dict[str, Any]:
    """Inspect one URL's Google indexing, crawl, canonical, and mobile status.

    site_url must exactly match a property returned by gsc_list_sites, and
    inspection_url must belong to that property. This is read-only and does not
    request indexing.
    """
    return (
        _service()
        .urlInspection()
        .index()
        .inspect(body={"siteUrl": site_url, "inspectionUrl": inspection_url})
        .execute()
    )


@tool
def gsc_list_sitemaps(site_url: str) -> dict[str, Any]:
    """List submitted sitemaps and their status for a Search Console property."""
    return _service().sitemaps().list(siteUrl=site_url).execute()


def get_google_search_console_tools() -> list[BaseTool]:
    """Return Karl's native, read-only Search Console tool set."""
    return [
        gsc_list_sites,
        gsc_search_analytics,
        gsc_inspect_url,
        gsc_list_sitemaps,
    ]

"""Read-only Google AdSense Management API tools."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
import re
from typing import Any

import google.auth
from googleapiclient.discovery import build
from langchain_core.tools import tool

SCOPES = ["https://www.googleapis.com/auth/adsense.readonly"]

MAX_PAGE_SIZE = 100
MAX_REPORT_ROWS = 10_000
MAX_REPORT_DAYS = 366
MAX_DIMENSIONS = 5
MAX_METRICS = 10
MAX_FILTERS = 10
MAX_ORDER_BY = 10
MAX_REPORT_COLUMNS = 20


def _validate_positive_limit(value: int, *, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _validate_account(account: str) -> None:
    if not isinstance(account, str) or not re.fullmatch(r"accounts/[^/\s]+", account):
        raise ValueError(
            "account must be a resource name in the form accounts/{account}"
        )


def _parse_iso_date(value: str, *, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format")
    return parsed


def _validate_string_list(
    values: list[str] | None,
    *,
    name: str,
    maximum: int,
    required: bool = False,
) -> list[str]:
    if values is None:
        values = []
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if required and not values:
        raise ValueError(f"{name} must contain at least one value")
    if len(values) > maximum:
        raise ValueError(f"{name} may contain at most {maximum} values")
    return values


@lru_cache(maxsize=1)
def _service() -> Any:
    credentials, _ = google.auth.default(scopes=SCOPES)
    return build("adsense", "v2", credentials=credentials, cache_discovery=False)


@tool
def adsense_list_accounts(
    page_size: int = 100,
    page_token: str | None = None,
) -> dict[str, Any]:
    """List Google AdSense accounts accessible to the current credentials."""
    _validate_positive_limit(page_size, name="page_size", maximum=MAX_PAGE_SIZE)
    request: dict[str, Any] = {"pageSize": page_size}
    if page_token:
        request["pageToken"] = page_token
    return _service().accounts().list(**request).execute()


@tool
def adsense_list_sites(
    account: str,
    page_size: int = 100,
    page_token: str | None = None,
) -> dict[str, Any]:
    """List sites belonging to an AdSense account resource such as accounts/pub-123."""
    _validate_account(account)
    _validate_positive_limit(page_size, name="page_size", maximum=MAX_PAGE_SIZE)
    request: dict[str, Any] = {"parent": account, "pageSize": page_size}
    if page_token:
        request["pageToken"] = page_token
    return _service().accounts().sites().list(**request).execute()


@tool
def adsense_generate_report(
    account: str,
    start_date: str,
    end_date: str,
    metrics: list[str],
    dimensions: list[str] | None = None,
    filters: list[str] | None = None,
    order_by: list[str] | None = None,
    limit: int = 1_000,
    currency_code: str | None = None,
    language_code: str | None = None,
    reporting_time_zone: str = "ACCOUNT_TIME_ZONE",
) -> dict[str, Any]:
    """Generate a bounded AdSense report for an explicit ISO date range.

    Metrics and dimensions use AdSense API names, for example
    ESTIMATED_EARNINGS, PAGE_VIEWS, CLICKS, DATE, and OWNED_SITE_DOMAIN_NAME.
    """
    _validate_account(account)
    start = _parse_iso_date(start_date, name="start_date")
    end = _parse_iso_date(end_date, name="end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date")
    if (end - start).days + 1 > MAX_REPORT_DAYS:
        raise ValueError(f"report date range may not exceed {MAX_REPORT_DAYS} days")

    metrics = _validate_string_list(
        metrics, name="metrics", maximum=MAX_METRICS, required=True
    )
    dimensions = _validate_string_list(
        dimensions, name="dimensions", maximum=MAX_DIMENSIONS
    )
    filters = _validate_string_list(filters, name="filters", maximum=MAX_FILTERS)
    order_by = _validate_string_list(order_by, name="order_by", maximum=MAX_ORDER_BY)
    if len(metrics) + len(dimensions) > MAX_REPORT_COLUMNS:
        raise ValueError(
            f"report may request at most {MAX_REPORT_COLUMNS} dimensions and metrics"
        )
    _validate_positive_limit(limit, name="limit", maximum=MAX_REPORT_ROWS)

    allowed_time_zones = {"ACCOUNT_TIME_ZONE", "GOOGLE_TIME_ZONE"}
    if reporting_time_zone not in allowed_time_zones:
        raise ValueError(
            "reporting_time_zone must be ACCOUNT_TIME_ZONE or GOOGLE_TIME_ZONE"
        )

    request: dict[str, Any] = {
        "account": account,
        "dateRange": "CUSTOM",
        "startDate_year": start.year,
        "startDate_month": start.month,
        "startDate_day": start.day,
        "endDate_year": end.year,
        "endDate_month": end.month,
        "endDate_day": end.day,
        "metrics": metrics,
        "limit": limit,
        "reportingTimeZone": reporting_time_zone,
    }
    if dimensions:
        request["dimensions"] = dimensions
    if filters:
        request["filters"] = filters
    if order_by:
        request["orderBy"] = order_by
    if currency_code:
        request["currencyCode"] = currency_code
    if language_code:
        request["languageCode"] = language_code

    return _service().accounts().reports().generate(**request).execute()


def get_google_adsense_tools() -> list[Any]:
    """Return all Google AdSense tools available to agents."""
    return [adsense_list_accounts, adsense_list_sites, adsense_generate_report]

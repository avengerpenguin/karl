from unittest.mock import MagicMock

import pytest

from karl.google import search_console


@pytest.fixture
def service(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(search_console, "_service", lambda: fake)
    return fake


def test_list_sites(service):
    expected = {"siteEntry": [{"siteUrl": "sc-domain:traditionalmead.uk"}]}
    service.sites.return_value.list.return_value.execute.return_value = expected

    assert search_console.gsc_list_sites.invoke({}) == expected


def test_search_analytics_builds_read_only_query_and_filters(service):
    service.searchanalytics.return_value.query.return_value.execute.return_value = {
        "rows": [{"keys": ["mead"], "clicks": 3, "impressions": 20}]
    }

    result = search_console.gsc_search_analytics.invoke(
        {
            "site_url": "sc-domain:traditionalmead.uk",
            "start_date": "2026-09-01",
            "end_date": "2026-09-30",
            "dimensions": ["query", "page"],
            "row_limit": 200,
            "query_filter": "regex:^mead",
            "page_filter": "/reviews/",
            "country_filter": "gbr",
            "device_filter": "MOBILE",
        }
    )

    service.searchanalytics.return_value.query.assert_called_once_with(
        siteUrl="sc-domain:traditionalmead.uk",
        body={
            "startDate": "2026-09-01",
            "endDate": "2026-09-30",
            "type": "web",
            "rowLimit": 200,
            "dimensions": ["query", "page"],
            "dimensionFilterGroups": [
                {
                    "groupType": "and",
                    "filters": [
                        {
                            "dimension": "query",
                            "operator": "includingRegex",
                            "expression": "^mead",
                        },
                        {
                            "dimension": "page",
                            "operator": "contains",
                            "expression": "/reviews/",
                        },
                        {
                            "dimension": "country",
                            "operator": "equals",
                            "expression": "GBR",
                        },
                        {
                            "dimension": "device",
                            "operator": "equals",
                            "expression": "MOBILE",
                        },
                    ],
                }
            ],
        },
    )
    assert result["resultMetadata"] == {
        "returnedRows": 1,
        "rowLimit": 200,
        "mayBeTruncated": False,
    }


@pytest.mark.parametrize(
    ("start_date", "end_date", "row_limit", "message"),
    [
        ("not-a-date", "2026-09-30", 100, "YYYY-MM-DD"),
        ("2026-10-01", "2026-09-30", 100, "on or after"),
        ("2026-09-01", "2026-09-30", 25_001, "between 1 and 25000"),
    ],
)
def test_search_analytics_validates_bounded_requests(
    service, start_date, end_date, row_limit, message
):
    with pytest.raises(ValueError, match=message):
        search_console.gsc_search_analytics.invoke(
            {
                "site_url": "sc-domain:traditionalmead.uk",
                "start_date": start_date,
                "end_date": end_date,
                "row_limit": row_limit,
            }
        )

    service.searchanalytics.assert_not_called()


def test_inspect_url_is_read_only(service):
    expected = {"inspectionResult": {"indexStatusResult": {"verdict": "PASS"}}}
    inspect = service.urlInspection.return_value.index.return_value.inspect
    inspect.return_value.execute.return_value = expected

    result = search_console.gsc_inspect_url.invoke(
        {
            "site_url": "sc-domain:traditionalmead.uk",
            "inspection_url": "https://traditionalmead.uk/mead/",
        }
    )

    inspect.assert_called_once_with(
        body={
            "siteUrl": "sc-domain:traditionalmead.uk",
            "inspectionUrl": "https://traditionalmead.uk/mead/",
        }
    )
    assert result == expected


def test_list_sitemaps(service):
    expected = {"sitemap": [{"path": "https://traditionalmead.uk/sitemap.xml"}]}
    list_sitemaps = service.sitemaps.return_value.list
    list_sitemaps.return_value.execute.return_value = expected

    result = search_console.gsc_list_sitemaps.invoke(
        {"site_url": "sc-domain:traditionalmead.uk"}
    )

    list_sitemaps.assert_called_once_with(siteUrl="sc-domain:traditionalmead.uk")
    assert result == expected

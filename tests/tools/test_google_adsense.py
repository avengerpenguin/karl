from unittest.mock import MagicMock

import pytest

from karl.google import adsense


@pytest.fixture
def service(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(adsense, "_service", lambda: fake)
    return fake


def test_list_accounts(service):
    expected = {"accounts": [{"name": "accounts/pub-123"}]}
    list_accounts = service.accounts.return_value.list
    list_accounts.return_value.execute.return_value = expected

    result = adsense.adsense_list_accounts.invoke(
        {"page_size": 50, "page_token": "next-page"}
    )

    list_accounts.assert_called_once_with(pageSize=50, pageToken="next-page")
    assert result == expected


def test_list_sites(service):
    expected = {"sites": [{"name": "accounts/pub-123/sites/example.com"}]}
    list_sites = service.accounts.return_value.sites.return_value.list
    list_sites.return_value.execute.return_value = expected

    result = adsense.adsense_list_sites.invoke(
        {
            "account": "accounts/pub-123",
            "page_size": 25,
            "page_token": "next-sites",
        }
    )

    list_sites.assert_called_once_with(
        parent="accounts/pub-123", pageSize=25, pageToken="next-sites"
    )
    assert result == expected


def test_generate_report_builds_bounded_read_only_request(service):
    expected = {
        "headers": [{"name": "DATE"}, {"name": "ESTIMATED_EARNINGS"}],
        "rows": [{"cells": [{"value": "2026-09-01"}, {"value": "1.23"}]}],
    }
    generate = service.accounts.return_value.reports.return_value.generate
    generate.return_value.execute.return_value = expected

    result = adsense.adsense_generate_report.invoke(
        {
            "account": "accounts/pub-123",
            "start_date": "2026-09-01",
            "end_date": "2026-09-30",
            "dimensions": ["DATE", "OWNED_SITE_DOMAIN_NAME"],
            "metrics": ["PAGE_VIEWS", "CLICKS", "ESTIMATED_EARNINGS"],
            "filters": ["OWNED_SITE_DOMAIN_NAME==traditionalmead.uk"],
            "order_by": ["+DATE"],
            "limit": 500,
            "currency_code": "GBP",
            "language_code": "en-GB",
            "reporting_time_zone": "ACCOUNT_TIME_ZONE",
        }
    )

    generate.assert_called_once_with(
        account="accounts/pub-123",
        dateRange="CUSTOM",
        startDate_year=2026,
        startDate_month=9,
        startDate_day=1,
        endDate_year=2026,
        endDate_month=9,
        endDate_day=30,
        dimensions=["DATE", "OWNED_SITE_DOMAIN_NAME"],
        metrics=["PAGE_VIEWS", "CLICKS", "ESTIMATED_EARNINGS"],
        filters=["OWNED_SITE_DOMAIN_NAME==traditionalmead.uk"],
        orderBy=["+DATE"],
        limit=500,
        currencyCode="GBP",
        languageCode="en-GB",
        reportingTimeZone="ACCOUNT_TIME_ZONE",
    )
    assert result == expected


@pytest.mark.parametrize(
    ("account", "page_size", "message"),
    [
        ("pub-123", 10, "accounts"),
        ("accounts/", 10, "accounts"),
        ("accounts/pub 123", 10, "accounts"),
        ("accounts/pub-123/sites/example", 10, "accounts"),
        ("accounts/pub-123", 0, "between 1 and 100"),
        ("accounts/pub-123", 101, "between 1 and 100"),
    ],
)
def test_list_sites_validates_inputs(service, account, page_size, message):
    with pytest.raises(ValueError, match=message):
        adsense.adsense_list_sites.invoke({"account": account, "page_size": page_size})

    service.accounts.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"account": "pub-123"}, "accounts"),
        ({"start_date": "09/01/2026"}, "YYYY-MM-DD"),
        ({"end_date": "2026-08-31"}, "must not be after"),
        ({"end_date": "2027-09-02"}, "366 days"),
        ({"metrics": []}, "at least one"),
        ({"metrics": ["CLICKS"] * 11}, "at most 10"),
        ({"dimensions": ["DATE"] * 6}, "at most 5"),
        ({"filters": ["A==B"] * 11}, "at most 10"),
        ({"order_by": ["+DATE"] * 11}, "at most 10"),
        ({"limit": 10_001}, "between 1 and 10000"),
        ({"reporting_time_zone": "UTC"}, "ACCOUNT_TIME_ZONE"),
    ],
)
def test_generate_report_validates_bounded_requests(service, overrides, message):
    request = {
        "account": "accounts/pub-123",
        "start_date": "2026-09-01",
        "end_date": "2026-09-30",
        "metrics": ["ESTIMATED_EARNINGS"],
    }
    request.update(overrides)

    with pytest.raises(ValueError, match=message):
        adsense.adsense_generate_report.invoke(request)

    service.accounts.assert_not_called()

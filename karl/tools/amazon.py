import os
import time
from typing import Any

import requests
from langchain_core.tools import tool

CLIENT_ID = os.getenv("AMAZON_CREDENTIAL_ID")
CLIENT_SECRET = os.getenv("AMAZON_CREDENTIAL_SECRET")
PARTNER_TAG = os.getenv("AMAZON_PARTNER_TAG", "traditionalmead-21")
MARKETPLACE = os.getenv("AMAZON_MARKETPLACE", "www.amazon.co.uk")
TOKEN_ENDPOINT = os.getenv(
    "AMAZON_CREATORS_TOKEN_ENDPOINT",
    "https://api.amazon.co.uk/auth/o2/token",
)
SEARCH_ENDPOINT = os.getenv(
    "AMAZON_CREATORS_SEARCH_ENDPOINT",
    "https://creatorsapi.amazon/catalog/v1/searchItems",
)

_ACCESS_TOKEN: str | None = None
_ACCESS_TOKEN_EXPIRES_AT = 0.0


def _response_body_for_error(response: requests.Response) -> str:
    """Return a concise, useful response body for diagnostics."""
    content_type = response.headers.get("content-type", "")
    body = response.text.strip()
    if not body:
        return "<empty response body>"
    if "application/json" not in content_type.lower() and len(body) > 500:
        return f"{body[:500]}..."
    return body


def _get_amazon_access_token() -> str:
    """Get and cache an OAuth token for the Amazon Creators API."""
    global _ACCESS_TOKEN, _ACCESS_TOKEN_EXPIRES_AT

    if _ACCESS_TOKEN and time.time() < _ACCESS_TOKEN_EXPIRES_AT:
        return _ACCESS_TOKEN

    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "AMAZON_CREDENTIAL_ID and AMAZON_CREDENTIAL_SECRET must be set"
        )

    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "creatorsapi::default",
    }
    response = requests.post(
        TOKEN_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=20,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"token endpoint returned {response.status_code}: "
            f"{_response_body_for_error(response)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"token endpoint did not return JSON: {_response_body_for_error(response)}"
        ) from exc

    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError(f"token response missing access_token: {data}")

    expires_in = int(data.get("expires_in", 3600))
    # Refresh a little early rather than risking expiry during an API request.
    _ACCESS_TOKEN = access_token
    _ACCESS_TOKEN_EXPIRES_AT = time.time() + max(0, expires_in - 60)
    return access_token


def _items_from_search_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract product items from known Creators API response shapes."""
    # SearchItems is expected to return searchResult.items. Support itemsResult.items
    # too because GetItems examples use that shape, and retaining a products fallback
    # makes diagnostics gentler if Amazon changes or wraps the response.
    for key in ("searchResult", "itemsResult"):
        result = data.get(key)
        if isinstance(result, dict) and isinstance(result.get("items"), list):
            return result["items"]

    products = data.get("products")
    if isinstance(products, list):
        return products

    return []


def _item_title(item: dict[str, Any]) -> str:
    title = item.get("itemInfo", {}).get("title", {}).get("displayValue")
    return title or item.get("title") or "Untitled product"


@tool
def search_amazon_products_with_affiliate_links(query: str) -> str:
    """
    Search the Amazon product catalog for matching items.
    Returns a Markdown list of relevant products containing pre-formatted affiliate tracking links.
    Use this whenever you need to find products to recommend to users.
    """
    try:
        token = _get_amazon_access_token()
    except Exception as e:
        return f"Authentication failed with Amazon Creators API: {str(e)}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-marketplace": MARKETPLACE,
    }
    payload = {
        "keywords": query,
        "partnerTag": PARTNER_TAG,
        "marketplace": MARKETPLACE,
        "resources": [
            "images.primary.small",
            "itemInfo.title",
        ],
    }

    try:
        response = requests.post(
            SEARCH_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
        if response.status_code != 200:
            return (
                f"Amazon API returned an error ({response.status_code}): "
                f"{_response_body_for_error(response)}"
            )

        try:
            data = response.json()
        except ValueError:
            return (
                "Amazon API returned a non-JSON response: "
                f"{_response_body_for_error(response)}"
            )

        items = _items_from_search_response(data)
        if not items:
            return "No products found matching your search query."

        results = []
        for item in items:
            title = _item_title(item)
            asin = item.get("asin") or item.get("ASIN")
            # Amazon requires that associates use the links returned by the API,
            # rather than editing URL parameters manually.
            affiliate_url = item.get("detailPageURL") or item.get("detailPageUrl")

            if not affiliate_url:
                continue

            asin_text = f" — ASIN: `{asin}`" if asin else ""
            results.append(f"- [{title}]({affiliate_url}){asin_text}")

        return "\n".join(results) if results else "No linkable products found."

    except Exception as e:
        return f"Failed to execute product search: {str(e)}"

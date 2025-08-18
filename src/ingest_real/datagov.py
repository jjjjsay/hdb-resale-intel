import time
from typing import Iterator, Optional
import requests

# Public Data.gov.sg endpoints
BASE_V2 = "https://api-production.data.gov.sg/v2/public/api"
BASE_V1 = "https://api-open.data.gov.sg/v1/public/api"
BASE_LEGACY = "https://data.gov.sg/api/action"  # CKAN/Datastore API

def list_rows(dataset_id: str, limit: int = 10000, api_key: Optional[str] = None) -> Iterator[list]:
    """
    Yield lists of rows for a dataset using the v2 list-rows API with pagination.
    """
    url = f"{BASE_V2}/datasets/{dataset_id}/list-rows"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    params = {"limit": limit}
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != 1:
            raise RuntimeError(f"list_rows error: {js}")
        data = js["data"]
        rows = data.get("rows", [])
        if not rows:
            break
        yield rows
        next_link = data.get("links", {}).get("next")
        if not next_link:
            break
        url = next_link    # absolute URL returned by API
        params = None      # next-link already includes query args

def initiate_download(dataset_id: str, api_key: Optional[str] = None, body: Optional[dict] = None) -> dict:
    url = f"{BASE_V1}/datasets/{dataset_id}/initiate-download"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    r = requests.get(url, headers=headers, json=body or {}, timeout=60)
    r.raise_for_status()
    return r.json()

def poll_download(dataset_id: str, api_key: Optional[str] = None, body: Optional[dict] = None,
                  retries: int = 60, sleep_s: float = 2.0) -> str:
    """
    Poll Data.gov.sg for a downloadable URL for the dataset. More generous retries.
    Returns a direct URL (CSV/GeoJSON).
    """
    url = f"{BASE_V1}/datasets/{dataset_id}/poll-download"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    for _ in range(retries):
        r = requests.get(url, headers=headers, json=body or {}, timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("code") == 1 and "data" in js:
            data = js["data"]
            if data.get("url"):
                return data["url"]
        time.sleep(sleep_s)
    raise TimeoutError(f"poll_download timed out for {dataset_id}")

def datastore_fetch(resource_id: str, limit: int = 5000, api_key: Optional[str] = None,
                    max_rows: Optional[int] = None) -> Iterator[list]:
    """
    Fallback: CKAN Datastore pagination for a known resource_id.
    Yields lists of records.
    """
    offset = 0
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    while True:
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        r = requests.get(f"{BASE_LEGACY}/datastore_search", headers=headers, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        if not js.get("success"):
            raise RuntimeError(f"datastore_search error: {js}")
        result = js["result"]
        records = result.get("records", [])
        if not records:
            break
        yield records
        offset += len(records)
        if max_rows is not None and offset >= max_rows:
            break

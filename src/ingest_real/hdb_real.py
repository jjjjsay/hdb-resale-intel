import time
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

from ..utils.paths import DATA_RAW

# HDB Resale prices (2017 onwards) CKAN resource id
RESOURCE_ID_2017_ONWARDS = "f1765b54-a209-4718-8d38-a39237f502b3"
CKAN_URL = "https://data.gov.sg/api/action/datastore_search"


def _ckan_fetch_all(
    resource_id: str,
    api_key: Optional[str] = None,
    limit_start: int = 32000,
    min_limit: int = 1000,
    timeout: int = 60,
    sleep_secs: float = 0.2,
) -> pd.DataFrame:
    """
    Robust CKAN pager with automatic backoff on HTTP 422.
    Returns a DataFrame with all records.
    """
    headers = {"Authorization": api_key} if api_key else {}
    frames: List[pd.DataFrame] = []

    limit = limit_start
    offset = 0

    while True:
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        r = requests.get(CKAN_URL, params=params, headers=headers, timeout=timeout)

        # Back off on 422 (Unprocessable Entity) – CKAN sometimes needs smaller page sizes
        if r.status_code == 422 and limit > min_limit:
            limit = max(min_limit, limit // 2)
            continue

        r.raise_for_status()
        js = r.json()
        result = js.get("result", {})
        recs = result.get("records", [])
        if not recs:
            break

        frames.append(pd.DataFrame.from_records(recs))
        got = len(recs)
        offset += got

        # stop when the last page is shorter than the limit
        if got < limit:
            break

        if sleep_secs:
            time.sleep(sleep_secs)

    if not frames:
        raise RuntimeError("CKAN returned no records.")

    return pd.concat(frames, ignore_index=True)


def fetch_hdb_resale_real(
    api_key: Optional[str] = None,
    out_path: Path = DATA_RAW / "hdb_resale_2017_onwards.csv",
    local_fallback: Optional[Path] = None,
) -> Path:
    """
    Fetch HDB resale data (2017 onwards). Try CKAN; if that fails, use a local CSV fallback.
    Writes CSV to `out_path` and returns the path.
    """
    if local_fallback is None:
        local_fallback = out_path  # default to same file if you already have it locally

    try:
        df = _ckan_fetch_all(RESOURCE_ID_2017_ONWARDS, api_key=api_key)
        # Standardize headings like the official export
        rename = {
            "month": "month",
            "town": "town",
            "flat_type": "flat_type",
            "block": "block",
            "street_name": "street_name",
            "storey_range": "storey_range",
            "floor_area_sqm": "floor_area_sqm",
            "flat_model": "flat_model",
            "lease_commence_date": "lease_commence_year",  # CKAN field name may vary
            "lease_commence_year": "lease_commence_year",
            "remaining_lease": "remaining_lease",
            "resale_price": "resale_price",
        }
        df = df.rename(columns=rename)

        # Keep the expected columns only if present
        cols = [
            "month", "town", "flat_type", "block", "street_name",
            "storey_range", "floor_area_sqm", "flat_model",
            "lease_commence_year", "remaining_lease", "resale_price",
        ]
        present = [c for c in cols if c in df.columns]
        if present:
            df = df[present]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[HDB] CKAN fetch OK → wrote {len(df):,} rows to {out_path}")
        return out_path

    except Exception as e:
        print(f"[HDB] CKAN failed: {e}. Falling back to local file: {local_fallback}")
        if not Path(local_fallback).exists():
            raise FileNotFoundError(
                f"Local fallback not found at {local_fallback}. "
                f"Please place the raw HDB CSV there."
            )
        # Copy/normalize local fallback into out_path if needed
        df = pd.read_csv(local_fallback)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[HDB] Using local fallback → wrote {len(df):,} rows to {out_path}")
        return out_path

from __future__ import annotations
import time
from typing import Optional, Tuple, Dict
import requests
from pathlib import Path

ONEMAP_SEARCH = "https://developers.onemap.sg/commonapi/search"

def geocode_onemap(search_val: str, sleep_s: float = 0.15) -> Optional[Tuple[float, float]]:
    """
    Returns (lat, lon) or None.
    """
    try:
        params = {
            "searchVal": search_val,
            "returnGeom": "Y",
            "getAddrDetails": "Y",
            "pageNum": 1,
        }
        r = requests.get(ONEMAP_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        results = js.get("results") or []
        time.sleep(sleep_s)  # courtesy pause for rate limits
        if not results:
            return None
        lat = float(results[0]["LATITUDE"])
        lon = float(results[0]["LONGITUDE"])
        return (lat, lon)
    except Exception:
        return None  # swallow DNS/timeouts/HTTP errors

def cached_geocode_many(values, cache_file: Path) -> Dict[str, Tuple[float, float]]:

    cache: Dict[str, Tuple[float, float]] = {}
    if cache_file.exists():
        import pandas as pd
        df = pd.read_csv(cache_file)
        for _, row in df.iterrows():
            cache[str(row["key"])] = (float(row["lat"]), float(row["lon"]))  # type: ignore

    new_rows = []
    for v in values:
        key = str(v)
        if key in cache or not key.strip():
            continue
        coords = geocode_onemap(key)
        if coords:
            cache[key] = coords
            new_rows.append({"key": key, "lat": coords[0], "lon": coords[1]})

    if new_rows:
        import pandas as pd
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            pd.DataFrame(new_rows).to_csv(cache_file, mode="a", header=False, index=False)
        else:
            pd.DataFrame(new_rows).to_csv(cache_file, index=False)

    return cache

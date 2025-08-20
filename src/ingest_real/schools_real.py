from pathlib import Path
import pandas as pd
from typing import Optional
import requests
from .datagov import poll_download, list_rows
from ..utils.paths import DATA_RAW
from ..utils.geo import cached_geocode_many
from pathlib import Path as P

# MOE: General Information of Schools datasetId (Data.gov.sg)
SCHOOLS_DATASET_ID = "d_688b934f82c1059ed0a6993d2a829089"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    # common MOE columns to harmonize if present
    ren = {
        "schoolname": "school_name",
        "school_name": "school_name",
        "postalcode": "postal_code",
        "postal_code": "postal_code",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df

def _geocode_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    # If lat/lon not present, build address string and geocode via OneMap (cached)
    lat_col = next((c for c in df.columns if c in ("lat","latitude")), None)
    lon_col = next((c for c in df.columns if c in ("lon","longitude","lng")), None)

    if lat_col and lon_col:
        df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
        return df

    # Build a geocode key like: "SCHOOL NAME ADDRESS POSTAL Singapore"
    addr_cols = [c for c in df.columns if c in ("address","addr1","addr","street_name")]
    postal_col = next((c for c in df.columns if "postal" in c), None)

    def make_key(row):
        parts = []
        if "school_name" in df.columns:
            parts.append(str(row.get("school_name", "")))
        for c in addr_cols:
            parts.append(str(row.get(c, "")))
        if postal_col:
            parts.append(str(row.get(postal_col, "")))
        parts.append("Singapore")
        return " ".join(p for p in parts if p and p.lower() != "nan")

    keys = df.apply(make_key, axis=1).tolist()
    cache_file = P(__file__).resolve().parents[2] / "data" / "cache" / "onemap_school_geocode.csv"
    cache = cached_geocode_many(keys, cache_file)

    lats, lons = [], []
    for k in keys:
        coords = cache.get(str(k))
        if coords:
            lats.append(coords[0]); lons.append(coords[1])
        else:
            lats.append(None); lons.append(None)
    df["lat"] = lats; df["lon"] = lons
    return df

def _try_local_csv(out: Path) -> Optional[Path]:
    """
    Manual fallback: user-provided CSV at data/raw/schools/schools.csv
    Should contain at least either (school_name + address/postal) OR (lat, lon).
    """
    local = DATA_RAW / "schools" / "schools.csv"
    if not local.exists():
        return None
    df = pd.read_csv(local)
    df = _normalize_columns(df)
    df = _geocode_if_needed(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

def _osm_fallback(out: Path) -> Path:
    """
    Last resort: use OpenStreetMap Overpass to fetch school nodes in Singapore.
    Provides columns: school_name (if tagged), lat, lon.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """
        [out:json][timeout:60];
        area["name"="Singapore"]->.sg;
        (
          node["amenity"="school"](area.sg);
        );
        out body;
    """
    r = requests.post(overpass_url, data={"data": query}, timeout=90)
    r.raise_for_status()
    data = r.json()
    rows = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            lat = el.get("lat"); lon = el.get("lon")
            name = (el.get("tags") or {}).get("name")
            rows.append({"school_name": name, "lat": lat, "lon": lon})
    if not rows:
        raise RuntimeError("OSM fallback returned no schools.")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

def fetch_schools_real(api_key: str | None = None) -> Path:
    """
    Robust schools ingestion with 4 attempts:
      1) Data.gov.sg poll-download (CSV)
      2) v2 list-rows pagination
      3) Local CSV (data/raw/schools/schools.csv)
      4) OSM Overpass (amenity=school)
    Output: data/raw/schools/schools.csv
    """
    out = DATA_RAW / "schools" / "schools.csv"

    # Attempt 1: poll-download → CSV
    try:
        url = poll_download(SCHOOLS_DATASET_ID, api_key=api_key)
        df = pd.read_csv(url)
        df = _normalize_columns(df)
        df = _geocode_if_needed(df)
        df.to_csv(out, index=False)
        return out
    except Exception as e:
        print(f"[SCHOOLS] poll_download failed: {e}. Trying list_rows fallback...")

    # Attempt 2: v2 list-rows → stream into DataFrame
    try:
        pages = []
        for rows in list_rows(SCHOOLS_DATASET_ID, limit=5000, api_key=api_key):
            # rows should already be dict-like; if nested, normalize
            if rows and isinstance(rows[0], dict):
                pages.append(pd.DataFrame(rows))
            else:
                # attempt to wrap
                pages.append(pd.DataFrame({"row": rows}))
        if pages:
            df = pd.concat(pages, ignore_index=True)
            # If CKAN-like nesting exists (e.g., 'row' key), try normalize
            if "row" in df.columns and isinstance(df.loc[0, "row"], dict):
                df = pd.json_normalize(df["row"])
            df = _normalize_columns(df)
            df = _geocode_if_needed(df)
            df.to_csv(out, index=False)
            return out
        else:
            print("[SCHOOLS] list_rows returned no pages.")
    except Exception as e:
        print(f"[SCHOOLS] list_rows failed: {e}. Trying local CSV fallback...")

    # Attempt 3: Local CSV
    local = _try_local_csv(out)
    if local is not None:
        return local

    # Attempt 4: OSM Overpass
    print("[SCHOOLS] Using OpenStreetMap Overpass fallback...")
    return _osm_fallback(out)

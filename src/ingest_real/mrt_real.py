from pathlib import Path
import pandas as pd
import re, json, requests, time
from .datagov import poll_download
from ..utils.paths import DATA_RAW

# LTA MRT Station Exit (GeoJSON) datasetId
MRT_DATASET_ID = "d_b39d3a0871985372d7e1637193335da5"

# Approximate centroids per HDB town — used to synthesize proxy MRT points if everything else fails
TOWN_CENTROIDS = {
    "ANG MO KIO": (1.3691, 103.8454),
    "BEDOK": (1.3236, 103.9273),
    "BISHAN": (1.3508, 103.8485),
    "BUKIT BATOK": (1.3502, 103.7513),
    "BUKIT MERAH": (1.2773, 103.8195),
    "BUKIT PANJANG": (1.3786, 103.7623),
    "BUKIT TIMAH": (1.3294, 103.8021),
    "CENTRAL AREA": (1.2893, 103.8519),
    "CHOA CHU KANG": (1.3854, 103.7445),
    "CLEMENTI": (1.3151, 103.7643),
    "GEYLANG": (1.3146, 103.8935),
    "HOUGANG": (1.3612, 103.8863),
    "JURONG EAST": (1.3331, 103.7436),
    "JURONG WEST": (1.3405, 103.7090),
    "KALLANG/WHAMPOA": (1.3138, 103.8574),
    "MARINE PARADE": (1.3013, 103.9050),
    "PASIR RIS": (1.3721, 103.9490),
    "PUNGGOL": (1.4050, 103.9020),
    "QUEENSTOWN": (1.2941, 103.7865),
    "SEMBAWANG": (1.4491, 103.8190),
    "SENGKANG": (1.3933, 103.8957),
    "SERANGOON": (1.3522, 103.8670),
    "TAMPINES": (1.3526, 103.9447),
    "TOA PAYOH": (1.3333, 103.8500),
    "WOODLANDS": (1.4360, 103.7865),
    "YISHUN": (1.4304, 103.8353),
}

def _parse_description(desc: str):
    """Extract station name and exit code from the HTML-ish 'Description' in GeoJSON."""
    station = None
    exit_code = None
    if not isinstance(desc, str):
        return station, exit_code
    st = re.search(r"<th>STATION_NA</th>\\s*<td>(.*?)</td>", desc)
    ex = re.search(r"<th>EXIT_CODE</th>\\s*<td>(.*?)</td>", desc)
    if st: station = st.group(1).strip()
    if ex: exit_code = ex.group(1).strip()
    return station, exit_code

def _write_csv(rows, out: Path) -> Path:
    df = pd.DataFrame(rows)
    if {"lat", "lon"}.issubset(df.columns):
        df = df.dropna(subset=["lat","lon"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

def _try_parse_local_geojson(out_csv: Path) -> Path | None:
    """Manual fallback: if user places data/raw/mrt/mrt_exits.geojson locally, parse it."""
    local_gj = DATA_RAW / "mrt" / "mrt_exits.geojson"
    if not local_gj.exists():
        local_gj = DATA_RAW / "mrt" / "mrt_exits.json"
        if not local_gj.exists():
            return None
    with open(local_gj, "r", encoding="utf-8") as f:
        gj = json.load(f)
    rows = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [None, None]
        lon, lat = coords[0], coords[1]
        station, exit_code = _parse_description(props.get("Description", ""))
        rows.append({
            "station_name": station or props.get("STATION_NA") or props.get("Name"),
            "exit_code": exit_code,
            "lat": lat,
            "lon": lon,
        })
    return _write_csv(rows, out_csv)

def _overpass_try(endpoints: list[str], query: str, timeout_s: int = 40) -> dict | None:
    """Try multiple Overpass mirrors; return JSON or None."""
    for idx, url in enumerate(endpoints, start=1):
        try:
            r = requests.post(url, data={"data": query}, timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[MRT] Overpass attempt {idx}/{len(endpoints)} failed: {e}")
            time.sleep(2)  # brief backoff
    return None

def _overpass_fallback(out_csv: Path) -> Path | None:
    """
    Fallback B: Use OpenStreetMap Overpass API to fetch subway stations in Singapore.
    Returns CSV path or None if all mirrors fail.
    """
    # several mirrors (rotate if one is slow/down)
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ]
    # compact query bounded to Singapore for speed
    query = """
        [out:json][timeout:25];
        area["name"="Singapore"]->.sg;
        (
          node["railway"="station"]["station"="subway"](area.sg);
          node["railway"="station"]["subway"="yes"](area.sg);
        );
        out body;
    """
    data = _overpass_try(endpoints, query, timeout_s=35)
    if not data:
        return None

    rows = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            lat = el.get("lat"); lon = el.get("lon")
            name = (el.get("tags") or {}).get("name")
            rows.append({"station_name": name, "exit_code": None, "lat": lat, "lon": lon})
    if not rows:
        return None
    return _write_csv(rows, out_csv)

def _proxy_from_town_centroids(out_csv: Path) -> Path:
    """
    Last-resort fallback: synthesize proxy 'MRT' points from HDB town centroids.
    This keeps distance features meaningful across the island when all networks are down.
    """
    rows = []
    for town, (lat, lon) in TOWN_CENTROIDS.items():
        rows.append({
            "station_name": f"MRT Proxy - {town.title()}",
            "exit_code": None,
            "lat": lat,
            "lon": lon,
        })
    print("[MRT] Using proxy MRT points from town centroids as last resort.")
    return _write_csv(rows, out_csv)

def fetch_mrt_exits_real(api_key: str | None = None) -> Path:
    """
    Robust MRT ingestion with 4 attempts:
    1) Data.gov.sg poll-download (GeoJSON) → parse exits
    2) Local file fallback: data/raw/mrt/mrt_exits.(geojson|json)
    3) Overpass (multiple mirrors)
    4) Proxy: synthesized 'MRT' points from town centroids
    Always returns a CSV path; never raises on network failure.
    """
    out = DATA_RAW / "mrt" / "mrt_exits.csv"

    # Attempt 1: Data.gov.sg GeoJSON via poll-download
    try:
        url = poll_download(MRT_DATASET_ID, api_key=api_key)
        gj = requests.get(url, timeout=45).json()
        rows = []
        for feat in gj.get("features", []):
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates") or [None, None]
            lon, lat = coords[0], coords[1]
            station, exit_code = _parse_description(props.get("Description", ""))
            rows.append({
                "station_name": station or props.get("STATION_NA") or props.get("Name"),
                "exit_code": exit_code,
                "lat": lat,
                "lon": lon,
            })
        if rows:
            return _write_csv(rows, out)
        else:
            print("[MRT] poll_download returned no features; trying local file fallback...")
    except Exception as e:
        print(f"[MRT] poll_download failed: {e}. Trying local file fallback...")

    # Attempt 2: Local file
    local = _try_parse_local_geojson(out)
    if local is not None:
        return local

    # Attempt 3: Overpass mirrors
    overpass = _overpass_fallback(out)
    if overpass is not None:
        return overpass

    # Attempt 4: Proxy from town centroids (never fails)
    return _proxy_from_town_centroids(out)


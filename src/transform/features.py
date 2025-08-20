# src/transform/features.py
from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Paths & helpers ----------
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"

def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _find_hdb_csv() -> Path:
    """
    Find the HDB resale CSV even if it's placed either in data/raw/ or data/raw/hdb/.
    Prefers the canonical name if found, otherwise any CSV that has block+street_name.
    """
    candidates: list[Path] = []
    # Canonical first
    candidates.append(ROOT / "data" / "raw" / "hdb" / "hdb_resale_2017_onwards.csv")
    candidates.append(ROOT / "data" / "raw" / "hdb_resale_2017_onwards.csv")
    # Folders
    hdb_dir = ROOT / "data" / "raw" / "hdb"
    raw_dir = ROOT / "data" / "raw"
    if hdb_dir.exists():
        candidates.extend(sorted(hdb_dir.glob("*.csv")))
    if raw_dir.exists():
        candidates.extend(sorted(raw_dir.glob("*.csv")))

    REQUIRED = {"block", "street_name"}
    for p in candidates:
        try:
            head = pd.read_csv(p, nrows=200)
            cols = {c.strip().lower() for c in head.columns}
            if REQUIRED.issubset(cols):
                return p
        except Exception:
            continue
    raise FileNotFoundError(
        "Could not find an HDB CSV. Place it as "
        "`data/raw/hdb/hdb_resale_2017_onwards.csv` or "
        "`data/raw/hdb_resale_2017_onwards.csv` (must contain block, street_name)."
    )

# Optional precise geocodes per (block, street_name)
def _find_block_street_geocodes() -> Path | None:
    return _first_existing(
        ROOT / "data" / "raw" / "geocodes" / "block_street_geocodes.csv",
        ROOT / "data" / "raw" / "block_street_geocodes.csv",
    )

# Optional MRT & schools (must contain columns: name, lat, lon)
def _find_mrt_csv() -> Path | None:
    return _first_existing(
        ROOT / "data" / "raw" / "mrt" / "mrt_exits.csv",
        ROOT / "data" / "raw" / "mrt_exits.csv",
    )

def _find_schools_csv() -> Path | None:
    return _first_existing(
        ROOT / "data" / "raw" / "schools" / "schools.csv",
        ROOT / "data" / "raw" / "schools.csv",
    )

# ---------- Constants ----------
TOWN_CENTROIDS = {
    "ANG MO KIO": (1.3691, 103.8454),
    "BEDOK": (1.3236, 103.9273),
    "BISHAN": (1.3509, 103.8485),
    "BUKIT BATOK": (1.3496, 103.7496),
    "BUKIT MERAH": (1.2826, 103.8179),
    "BUKIT PANJANG": (1.3786, 103.7639),
    "BUKIT TIMAH": (1.3294, 103.8021),
    "CENTRAL AREA": (1.2920, 103.8545),
    "CHOA CHU KANG": (1.3854, 103.7443),
    "CLEMENTI": (1.3151, 103.7643),
    "GEYLANG": (1.3181, 103.8839),
    "HOUGANG": (1.3612, 103.8930),
    "JURONG EAST": (1.3333, 103.7430),
    "JURONG WEST": (1.3496, 103.7080),
    "KALLANG/WHAMPOA": (1.3133, 103.8641),
    "MARINE PARADE": (1.3030, 103.9010),
    "PASIR RIS": (1.3730, 103.9490),
    "PUNGGOL": (1.4043, 103.9020),
    "QUEENSTOWN": (1.2941, 103.7851),
    "SEMBAWANG": (1.4491, 103.8201),
    "SENGKANG": (1.3911, 103.8950),
    "SERANGOON": (1.3524, 103.8677),
    "TAMPINES": (1.3536, 103.9455),
    "TOA PAYOH": (1.3347, 103.8530),
}

# ---------- Feature helpers ----------
def _parse_remaining_lease(text: str) -> float:
    if not isinstance(text, str):
        return np.nan
    parts = text.lower().split()
    years = 0.0
    try:
        if "years" in parts:
            years += float(parts[parts.index("years") - 1])
        elif "year" in parts:
            years += float(parts[parts.index("year") - 1])
        if "months" in parts:
            years += float(parts[parts.index("months") - 1]) / 12.0
        elif "month" in parts:
            years += float(parts[parts.index("month") - 1]) / 12.0
    except Exception:
        return np.nan
    return years

def _storey_mid(s: str):
    try:
        a, _, b = s.split()
        return (int(a) + int(b)) / 2.0
    except Exception:
        return np.nan

def _haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine (meters). All inputs in decimal degrees."""
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

def _nearest_distance_m(points_df: pd.DataFrame, ref_df: pd.DataFrame, batch=10000, lat_col="lat", lon_col="lon") -> np.ndarray:
    """Nearest distance from each row in points_df to any ref_df point."""
    if ref_df is None or ref_df.empty:
        return np.full(len(points_df), np.nan, dtype="float64")

    pts_lat = points_df[lat_col].to_numpy(dtype="float64")
    pts_lon = points_df[lon_col].to_numpy(dtype="float64")
    ref_lat = ref_df["lat"].to_numpy(dtype="float64")
    ref_lon = ref_df["lon"].to_numpy(dtype="float64")

    out = np.full(len(points_df), np.nan, dtype="float64")
    n = len(points_df)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        la = pts_lat[start:stop][:, None]
        lo = pts_lon[start:stop][:, None]
        dists = _haversine_m(la, lo, ref_lat[None, :], ref_lon[None, :])  # (B, M)
        out[start:stop] = np.nanmin(dists, axis=1)
    return out

def _load_optional_csv(path: Path, required_cols=("lat", "lon")) -> pd.DataFrame | None:
    """
    Safely load optional CSV (returns None if missing, empty, or malformed).
    If 'required_cols' includes keys like ("block","street_name","lat","lon"), ensure they exist.
    """
    if not path or not path.exists():
        return None
    try:
        # Skip empty files outright
        if path.stat().st_size == 0:
            return None
    except Exception:
        return None

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    except Exception:
        return None

    # Require the necessary columns
    for c in required_cols:
        if c not in df.columns:
            return None

    # Coerce lat/lon if present
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # If the caller required block/street_name, normalize those
    for c in ("block", "street_name"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df = df.dropna(subset=[col for col in ["lat","lon"] if col in df.columns]).copy()
    return df if not df.empty else None

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["town","flat_type","street_name","block","flat_model","storey_range","remaining_lease"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

# ---------- Main ----------
def main():
    RAW_HDB = _find_hdb_csv()
    GEO_PATH = _find_block_street_geocodes()
    RAW_MRT = _find_mrt_csv()
    RAW_SCHOOLS = _find_schools_csv()

    print({"hdb_csv": str(RAW_HDB),
           "geocodes_csv": str(GEO_PATH) if GEO_PATH else None,
           "mrt_csv": str(RAW_MRT) if RAW_MRT else None,
           "schools_csv": str(RAW_SCHOOLS) if RAW_SCHOOLS else None})

    # 1) Load HDB
    hdb = pd.read_csv(RAW_HDB)
    hdb = _norm(hdb)

    # 2) Derive features
    if "storey_range" in hdb.columns and "storey_mid" not in hdb.columns:
        hdb["storey_mid"] = hdb["storey_range"].apply(_storey_mid)
    if "remaining_lease" in hdb.columns and "remaining_lease_years" not in hdb.columns:
        hdb["remaining_lease_years"] = hdb["remaining_lease"].apply(_parse_remaining_lease)
    if {"resale_price","floor_area_sqm"}.issubset(hdb.columns) and "price_per_sqm" not in hdb.columns:
        hdb["price_per_sqm"] = pd.to_numeric(hdb["resale_price"], errors="coerce") / pd.to_numeric(hdb["floor_area_sqm"], errors="coerce")
    if "lease_commence_date" not in hdb.columns and "lease_commence_year" in hdb.columns:
        hdb["lease_commence_date"] = hdb["lease_commence_year"]

    # 3) lat/lon via block+street geocodes, else town centroids
    if "lat" not in hdb.columns: hdb["lat"] = np.nan
    if "lon" not in hdb.columns: hdb["lon"] = np.nan

    precise = 0
    geo_df = _load_optional_csv(GEO_PATH, required_cols=("block", "street_name", "lat", "lon")) if GEO_PATH else None
    if geo_df is not None and not geo_df.empty:
        for c in ["block","street_name"]:
            if c in hdb.columns:
                hdb[c] = hdb[c].astype(str).str.strip()
        before_na = hdb["lat"].isna().sum()
        hdb = hdb.merge(
            geo_df[["block","street_name","lat","lon"]],
            on=["block","street_name"],
            how="left",
            suffixes=("", "")
        )
        if "lat_y" in hdb.columns and "lon_y" in hdb.columns:
            hdb["lat"] = hdb["lat_y"].combine_first(hdb.get("lat_x"))
            hdb["lon"] = hdb["lon_y"].combine_first(hdb.get("lon_x"))
            hdb.drop(columns=[c for c in ["lat_x","lat_y","lon_x","lon_y"] if c in hdb.columns], inplace=True)
        hdb["lat"] = pd.to_numeric(hdb["lat"], errors="coerce")
        hdb["lon"] = pd.to_numeric(hdb["lon"], errors="coerce")
        precise = before_na - hdb["lat"].isna().sum()

    # town-centroid fallback
    mask = hdb["lat"].isna() | hdb["lon"].isna()
    filled_centroid = 0
    if "town" in hdb.columns and mask.any():
        tt = hdb.loc[mask, "town"].astype(str).str.upper().map(TOWN_CENTROIDS)
        hdb.loc[mask, "lat"] = tt.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        hdb.loc[mask, "lon"] = tt.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        filled_centroid = hdb.loc[mask, ["lat","lon"]].notna().all(axis=1).sum()

    # ensure numeric
    hdb["lat"] = pd.to_numeric(hdb["lat"], errors="coerce")
    hdb["lon"] = pd.to_numeric(hdb["lon"], errors="coerce")

    # 4) Distances to MRT / schools (if their CSVs with lat/lon exist)
    mrt_df = _load_optional_csv(RAW_MRT) if RAW_MRT else None
    schools_df = _load_optional_csv(RAW_SCHOOLS) if RAW_SCHOOLS else None

    if "dist_to_mrt_m" not in hdb.columns:
        hdb["dist_to_mrt_m"] = np.nan
    if "dist_to_school_m" not in hdb.columns:
        hdb["dist_to_school_m"] = np.nan

    ok_mask = hdb[["lat","lon"]].notna().all(axis=1)
    if mrt_df is not None and ok_mask.any():
        hdb.loc[ok_mask, "dist_to_mrt_m"] = _nearest_distance_m(hdb.loc[ok_mask, ["lat","lon"]], mrt_df)
    if schools_df is not None and ok_mask.any():
        hdb.loc[ok_mask, "dist_to_school_m"] = _nearest_distance_m(hdb.loc[ok_mask, ["lat","lon"]], schools_df)

    # 5) Select & write
    keep_cols = [
        "month","town","flat_type","block","street_name",
        "storey_range","storey_mid","floor_area_sqm","flat_model",
        "lease_commence_date","remaining_lease","remaining_lease_years",
        "resale_price","price_per_sqm",
        "lat","lon","dist_to_mrt_m","dist_to_school_m",
    ]
    present = [c for c in keep_cols if c in hdb.columns]
    out_df = hdb[present].copy()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "features_real.csv"
    out_df.to_csv(out_path, index=False)

    # Summary
    with_coords = out_df[["lat","lon"]].notna().all(axis=1).sum()
    print({
        "wrote": str(out_path),
        "rows": int(len(out_df)),
        "with_coords": int(with_coords),
        "precise_block_street": int(max(0, precise)),
        "filled_by_town_centroid": int(max(0, filled_centroid)),
        "mrt_distances_computed": int(out_df["dist_to_mrt_m"].notna().sum()) if "dist_to_mrt_m" in out_df else 0,
        "school_distances_computed": int(out_df["dist_to_school_m"].notna().sum()) if "dist_to_school_m" in out_df else 0,
    })

if __name__ == "__main__":
    main()

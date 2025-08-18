import pandas as pd
import numpy as np
from pathlib import Path
from ..utils.paths import DATA_PROCESSED

def _parse_remaining_lease(text: str) -> float:
    """Convert strings like '61 years 03 months' -> 61.25 (years)."""
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

def _mid(s: str):
    try:
        a, _, b = s.split()
        return (int(a) + int(b)) / 2
    except Exception:
        return np.nan

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize types/whitespace
    for c in ["town","flat_type","street_name","block","flat_model","storey_range","remaining_lease"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # month standardization (string YYYY-MM for joins)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.strftime("%Y-%m")
    return df

def build_features(hdb_csv: Path, mrt_csv: Path, schools_csv: Path) -> Path:
    """
    Build processed features and write data/processed/features_real.csv.
    Guarantees keeping block/street_name by merging from the raw HDB CSV.
    """
    # ---------- Load RAW HDB ----------
    raw = pd.read_csv(hdb_csv)
    raw = _norm_cols(raw)

    # Start from raw (you can insert earlier filters/transforms here if needed)
    df = raw.copy()

    # ---------- Derivations ----------
    if "storey_range" in df.columns and "storey_mid" not in df.columns:
        df["storey_mid"] = df["storey_range"].apply(_mid)

    if "remaining_lease" in df.columns and "remaining_lease_years" not in df.columns:
        df["remaining_lease_years"] = df["remaining_lease"].apply(_parse_remaining_lease)

    if {"resale_price","floor_area_sqm"}.issubset(df.columns) and "price_per_sqm" not in df.columns:
        df["price_per_sqm"] = pd.to_numeric(df["resale_price"], errors="coerce") / pd.to_numeric(df["floor_area_sqm"], errors="coerce")

    # lease column harmonization
    if "lease_commence_date" not in df.columns and "lease_commence_year" in df.columns:
        df["lease_commence_date"] = df["lease_commence_year"]

    # ---------- Ensure block/street_name ----------
    need_cols = []
    if "block" not in df.columns:
        need_cols.append("block")
    if "street_name" not in df.columns:
        need_cols.append("street_name")

    if need_cols:
        # Build a join key present in both raw and df
        join_keys = [k for k in ["month","town","flat_type","floor_area_sqm","storey_range","resale_price"] if k in df.columns and k in raw.columns]
        # If storey_range missing in one side, drop it from keys
        if not join_keys:
            # fallback minimal keys
            join_keys = [k for k in ["month","town","flat_type","floor_area_sqm","resale_price"] if k in df.columns and k in raw.columns]

        if join_keys:
            add_cols = list(set(need_cols) & set(raw.columns))
            df = df.merge(raw[join_keys + add_cols].drop_duplicates(), on=join_keys, how="left")

    # ---------- Select columns to keep ----------
    keep_cols = [
        "month", "town", "flat_type", "block", "street_name",
        "storey_range", "floor_area_sqm", "flat_model",
        "lease_commence_date", "remaining_lease", "remaining_lease_years",
        "resale_price", "price_per_sqm",
        "lat", "lon", "dist_to_mrt_m", "dist_to_school_m",
        "storey_mid",
    ]
    present = [c for c in keep_cols if c in df.columns]
    out_df = df[present].copy()

    # ---------- Write ----------
    out = DATA_PROCESSED / "features_real.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out


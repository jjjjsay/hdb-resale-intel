# src/transform/make_block_geocodes.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from ..utils.geo import cached_geocode_many

ROOT = Path(__file__).resolve().parents[2]

# Search in both locations:
#  - repo default: data/raw/hdb
#  - your current layout: data/raw
HDB_DIRS = [
    ROOT / "data" / "raw" / "hdb",
    ROOT / "data" / "raw",
]

OUT = ROOT / "data" / "raw" / "geocodes" / "block_street_geocodes.csv"
CACHE = ROOT / "data" / "cache" / "onemap_block_geocode.csv"

REQUIRED = {"block", "street_name"}

def _find_hdb_csv() -> Path:
    """
    Find a suitable HDB resale CSV:
      1) Prefer 'hdb_resale_2017_onwards.csv' if present in either directory.
      2) Otherwise, pick the first CSV that contains the required columns.
    """
    # 1) Try canonical filename first
    for d in HDB_DIRS:
        if d.exists():
            p = d / "hdb_resale_2017_onwards.csv"
            if p.exists():
                return p

    # 2) Fallback: scan for any CSV that has the required columns
    for d in HDB_DIRS:
        if not d.exists():
            continue
        for candidate in d.glob("*.csv"):
            try:
                head = pd.read_csv(candidate, nrows=200)  # small sniff for columns
                cols = {c.strip().lower() for c in head.columns}
                if REQUIRED.issubset(cols):
                    return candidate
            except Exception:
                continue

    # Nothing found
    raise FileNotFoundError(
        f"No suitable CSV found in {', '.join(str(x) for x in HDB_DIRS)}. "
        f"Expected a file with columns {sorted(REQUIRED)} or a file named "
        f"'hdb_resale_2017_onwards.csv'."
    )

def main():
    raw_hdb = _find_hdb_csv()
    print({"using": str(raw_hdb)})

    # Read raw HDB (as strings to avoid dtype surprises) and normalize join keys
    df = pd.read_csv(raw_hdb, dtype=str)
    for c in ["block", "street_name", "town"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Unique (block, street_name) pairs to geocode
    pairs = (
        df.dropna(subset=["block", "street_name"])[["block", "street_name"]]
          .drop_duplicates()
          .reset_index(drop=True)
    )

    # Build OneMap search keys
    keys = (pairs["block"] + " " + pairs["street_name"] + " Singapore").tolist()

    # Use cached geocoder helper (writes/reads CACHE automatically)
    cache = cached_geocode_many(keys, CACHE)

    # Convert cache back to rows for (block, street_name, lat, lon)
    rows = []
    for _, r in pairs.iterrows():
        key = f"{r['block']} {r['street_name']} Singapore"
        coords = cache.get(key)
        if coords:
            rows.append({
                "block": r["block"],
                "street_name": r["street_name"],
                "lat": coords[0],
                "lon": coords[1],
            })

    # Write output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print({"wrote": str(OUT), "rows": len(rows)})

if __name__ == "__main__":
    main()

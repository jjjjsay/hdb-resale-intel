
from pathlib import Path
import pandas as pd
from ..utils.paths import DATA_RAW

def fetch_hdb_resale(sample: bool = True) -> Path:
    """Fetch HDB resale dataset and write to CSV. Returns the file path.
    This is a minimal offline-friendly stub. Replace with real data.gov.sg call.
    """
    out = DATA_RAW / "hdb" / ("hdb_resale_sample.csv" if sample else "hdb_resale_full.csv")
    if sample:
        df = pd.DataFrame(
            [
                {"month": "2024-01", "town": "ANG MO KIO", "flat_type": "3 ROOM", "floor_area_sqm": 67, "lease_commence_year": 1981, "remaining_lease": "56 years 10 months", "storey_range": "10 TO 12", "resale_price": 420000},
                {"month": "2024-02", "town": "BISHAN", "flat_type": "4 ROOM", "floor_area_sqm": 92, "lease_commence_year": 1991, "remaining_lease": "66 years 9 months", "storey_range": "13 TO 15", "resale_price": 680000},
            ]
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    else:
        raise NotImplementedError("Implement real HDB data fetch here.")
    return out

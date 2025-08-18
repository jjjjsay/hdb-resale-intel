
from pathlib import Path
import pandas as pd
from ..utils.paths import DATA_RAW

def fetch_mrt_exits(sample: bool = True) -> Path:
    out = DATA_RAW / "mrt" / ("mrt_exits_sample.csv" if sample else "mrt_exits.csv")
    if sample:
        df = pd.DataFrame(
            [
                {"station": "Bishan", "exit": "A", "lat": 1.351315, "lon": 103.848763},
                {"station": "Ang Mo Kio", "exit": "A", "lat": 1.369116, "lon": 103.849708},
            ]
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    else:
        raise NotImplementedError("Implement real MRT exits ingestion here.")
    return out

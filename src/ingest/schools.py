
from pathlib import Path
import pandas as pd
from ..utils.paths import DATA_RAW

def fetch_schools(sample: bool = True) -> Path:
    out = DATA_RAW / "schools" / ("schools_sample.csv" if sample else "schools.csv")
    if sample:
        df = pd.DataFrame(
            [
                {"school_name": "Catholic High School", "lat": 1.3501, "lon": 103.8369, "level": "Secondary"},
                {"school_name": "Anderson Primary", "lat": 1.3639, "lon": 103.8441, "level": "Primary"},
            ]
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    else:
        raise NotImplementedError("Implement real schools ingestion here.")
    return out

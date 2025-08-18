
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

(HDB_RAW := DATA_RAW / "hdb").mkdir(parents=True, exist_ok=True)
(MRT_RAW := DATA_RAW / "mrt").mkdir(parents=True, exist_ok=True)
(SCHOOLS_RAW := DATA_RAW / "schools").mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

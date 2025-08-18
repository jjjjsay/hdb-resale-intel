
from pathlib import Path

def ensure_parents(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

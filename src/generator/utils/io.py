# io.py
from pathlib import Path
import json
import pandas as pd

def ensure_dirs(*paths: Path) -> None:
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def read_csv(path: Path) -> pd.DataFrame: return pd.read_csv(path)
def write_csv(df, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); df.to_csv(path, index=False); return path

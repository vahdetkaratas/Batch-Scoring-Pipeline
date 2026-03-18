"""
Save scored CSV and optional run manifest JSON.
"""
import json
from pathlib import Path

import pandas as pd


SCORED_OUTPUTS_DIR = Path("data/scored_outputs")


def save_scored_output(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """
    Save scored output DataFrame to CSV. Default: data/scored_outputs/scored_batch_001.csv
    Creates directory if needed. Returns path written.
    """
    if path is None:
        SCORED_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SCORED_OUTPUTS_DIR / "scored_batch_001.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_run_manifest(scored_csv_path: Path, **meta) -> Path:
    """
    Write scored_batch_001.meta.json next to scored_batch_001.csv (audit trail).
    """
    scored_csv_path = Path(scored_csv_path)
    dest = scored_csv_path.parent / f"{scored_csv_path.stem}.meta.json"
    # JSON-serializable only
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()}
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return dest

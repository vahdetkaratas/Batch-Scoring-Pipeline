"""
Load input batch CSV.
"""
from pathlib import Path

import pandas as pd


def load_batch(path: str | Path) -> pd.DataFrame:
    """
    Load batch CSV from path. Expects Churn input schema (customerID + feature columns).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {path}")
    return pd.read_csv(path)

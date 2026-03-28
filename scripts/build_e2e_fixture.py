#!/usr/bin/env python3
"""
Regenerate tests/fixtures/e2e_pipeline.joblib from tests/fixtures/e2e_input.csv.

Run from repository root:
    python scripts/build_e2e_fixture.py

Uses validate_batch + preprocess_input, then fits a minimal sklearn Pipeline
(ColumnTransformer + LogisticRegression). Fixed synthetic labels for a stable fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_batch import load_batch  # noqa: E402
from src.validation.input_checks import validate_batch  # noqa: E402
from src.scoring.preprocess_input import preprocess_input  # noqa: E402

FIXTURE_CSV = ROOT / "tests" / "fixtures" / "e2e_input.csv"
OUT_MODEL = ROOT / "tests" / "fixtures" / "e2e_pipeline.joblib"

# Fixed labels so refitting stays deterministic (not a real churn signal).
FIXED_Y = np.array([0, 1, 0, 1, 0], dtype=np.int64)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> None:
    if not FIXTURE_CSV.exists():
        raise SystemExit(f"Missing fixture CSV: {FIXTURE_CSV}")

    df = load_batch(FIXTURE_CSV)
    validate_batch(df)
    X = preprocess_input(df)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", _one_hot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = LogisticRegression(random_state=0, max_iter=2000)
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X, FIXED_Y)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_MODEL)
    print(f"Wrote {OUT_MODEL} (rows={len(X)}, features shape after prep in fit)")


if __name__ == "__main__":
    main()

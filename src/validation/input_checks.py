"""
Input validation: required columns, non-empty batch, duplicate customer_id, numeric types.
TotalCharges and tenure are strictly validated to match preprocessing (no silent NaN / ambiguous bins).
"""
import numpy as np
import pandas as pd

# Churn input schema: customerID + feature columns (same as Telco raw minus Churn)
REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Must parse as finite numbers with no NaN after parse (same story as preprocess: no coerce-to-NaN).
NUMERIC_STRICT_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# TENURE_MAX matches preprocess_input.TENURE_BINS upper edge (72); validation rejects tenure above it so tenure_group is always defined.
TENURE_MIN = 0
TENURE_MAX = 72


def _strict_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    try:
        s = pd.to_numeric(df[col], errors="raise")
    except ValueError as e:
        raise ValueError(f"Column '{col}' must be numeric: {e}") from e
    if s.isna().any():
        raise ValueError(
            f"Column '{col}' must be numeric with no missing or unparseable values."
        )
    return s


def validate_batch(df: pd.DataFrame) -> None:
    """
    Raise ValueError if batch is invalid.
    - Required columns present
    - Batch not empty
    - No duplicate customerID
    - Strict numeric columns: parseable, finite, no NaN
    - tenure: whole numbers in [TENURE_MIN, TENURE_MAX] (matches tenure_group bins)
    """
    if df is None or df.empty:
        raise ValueError("Batch is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID found")

    strict_numeric = {
        col: _strict_numeric_series(df, col) for col in NUMERIC_STRICT_COLUMNS
    }

    t = strict_numeric["tenure"]
    if (t < TENURE_MIN).any() or (t > TENURE_MAX).any():
        raise ValueError(
            f"Column 'tenure' must be between {TENURE_MIN} and {TENURE_MAX} inclusive "
            "(required for defined tenure_group bins)."
        )
    # Integer months only — avoids ambiguous binning for fractional tenure.
    rem = np.mod(t.to_numpy(dtype=float), 1.0)
    if not np.allclose(rem, 0.0, rtol=0.0, atol=0.0):
        raise ValueError("Column 'tenure' must be whole numbers (integer months).")

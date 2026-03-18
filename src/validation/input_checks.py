"""
Input validation: required columns, non-empty batch, duplicate customer_id, numeric types.
Required columns, empty batch, duplicate IDs, numeric types.
"""
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

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges"]


def validate_batch(df: pd.DataFrame) -> None:
    """
    Raise ValueError if batch is invalid.
    - Required columns present
    - Batch not empty
    - No duplicate customerID
    - Numeric columns are numeric (or coercible)
    """
    if df is None or df.empty:
        raise ValueError("Batch is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID found")

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Column '{col}' must be numeric: {e}") from e

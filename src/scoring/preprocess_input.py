"""
Prepare batch for model: apply same feature engineering as Churn (TotalCharges, num_active_services, tenure_group).
Pipeline expects the same columns as churn training.
"""
import pandas as pd

# Mirrors churn training build_features so the joblib pipeline sees the same columns
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same logic as Churn build_features: TotalCharges numeric, num_active_services, tenure_group."""
    out = df.copy()

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    for col in SERVICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    if all(c in out.columns for c in SERVICE_COLUMNS):
        out["num_active_services"] = (
            out[SERVICE_COLUMNS]
            .apply(lambda x: (x.str.strip().str.lower() == "yes").astype(int))
            .sum(axis=1)
        )

    if "tenure" in out.columns:
        out["tenure_group"] = (
            pd.cut(
                out["tenure"].astype(int),
                bins=TENURE_BINS,
                labels=TENURE_LABELS,
                include_lowest=True,
            )
            .astype(object)
        )

    return out


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataframe with feature columns expected by Churn pipeline (after build_features).
    Drops customerID and Churn if present; keeps only feature columns.
    """
    fe = _build_features(df)
    # Drop identifier and optional Churn; pipeline expects feature matrix only
    drop = ["customerID"]
    if "Churn" in fe.columns:
        drop.append("Churn")
    return fe.drop(columns=[c for c in drop if c in fe.columns], errors="ignore")

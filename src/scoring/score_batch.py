"""
Run model predict_proba on preprocessed batch.
"""
from pathlib import Path

import pandas as pd


DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")


def load_pipeline(model_path: str | Path | None = None):
    """Load fitted sklearn pipeline (preprocessor + classifier) from joblib."""
    import joblib

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Copy from Churn project: "
            "customer-churn-prediction-system/artifacts/models/churn_model.joblib -> models/"
        )
    return joblib.load(path)


def score_batch(df_features: pd.DataFrame, pipeline=None):
    """
    Return array of churn probabilities (class 1) for each row.
    df_features: output of preprocess_input (no customerID, same columns as training).
    """
    if pipeline is None:
        pipeline = load_pipeline()
    proba = pipeline.predict_proba(df_features)[:, 1]
    return proba

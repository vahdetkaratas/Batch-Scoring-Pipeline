"""
Build output table: customer_id, churn_score, predicted_label, model_version, scoring_timestamp.
Scored output table columns.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DEFAULT_THRESHOLD = 0.35  # align with Churn threshold if available


def postprocess_predictions(
    customer_ids: pd.Series,
    churn_scores: pd.Series | np.ndarray | list,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model_version: str = "churn_v1",
) -> pd.DataFrame:
    """
    Build scored output DataFrame.
    customer_ids: from raw batch (e.g. df["customerID"]).
    churn_scores: probability from score_batch (array or series).
    """
    scores = np.asarray(churn_scores)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pred_label = (scores >= threshold).astype(int)
    return pd.DataFrame({
        "customer_id": customer_ids.values,
        "churn_score": scores,
        "predicted_label": pred_label,
        "model_version": model_version,
        "scoring_timestamp": ts,
    })

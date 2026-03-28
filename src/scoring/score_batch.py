"""
Run model predict_proba on preprocessed batch.
Positive churn probability is resolved from classes_ (never blind [:, 1]).
"""
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")


def _names_list(seq) -> list[str]:
    """Plain str list for order-sensitive column comparison."""
    return [str(x) for x in seq]


def _assert_feature_names_aligned(pipeline, df_features: pd.DataFrame) -> None:
    """
    If the fitted estimator exposes ``feature_names_in_``, require it to match
    ``df_features.columns`` exactly (names and order). Otherwise sklearn errors
    are often opaque.
    """
    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is None:
        return
    actual = _names_list(df_features.columns)
    want = _names_list(np.asarray(expected, dtype=object).ravel())
    if actual != want:
        raise ValueError(
            "Preprocessed feature columns do not match the model's expected input "
            "(names and order must be identical). "
            f"Model expects: {want!r}. "
            f"Preprocess produced: {actual!r}."
        )


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


def _estimator_classes(pipeline) -> np.ndarray:
    """Binary scoring requires classes_ on the fitted estimator (Pipeline forwards this)."""
    if not hasattr(pipeline, "classes_"):
        raise ValueError(
            "Scoring requires a fitted classifier with `classes_` (binary churn model). "
            f"Got estimator type: {type(pipeline).__name__!r}."
        )
    return np.asarray(pipeline.classes_)


def _positive_churn_column_index(classes: np.ndarray) -> int:
    """
    Map churn-positive class label to its column index in predict_proba output.

    Contract for this project (Telco-style churn):
    - Prefer numeric positive label ``1`` (standard sklearn 0/1 encoding).
    - Else accept string label ``'Yes'`` (case-insensitive) for No/Yes targets.

    predict_proba columns follow the same order as ``classes_``.
    """
    if classes.size != 2:
        raise ValueError(
            "Binary churn scoring requires exactly 2 classes; "
            f"got {classes.size}: {classes.tolist()!r}."
        )

    for idx, label in enumerate(classes):
        try:
            if int(label) == 1:
                return idx
        except (TypeError, ValueError):
            pass

    for idx, label in enumerate(classes):
        if isinstance(label, str) and label.strip().lower() == "yes":
            return idx

    raise ValueError(
        "Cannot resolve positive churn class: expected one of the two classes to be "
        "integer 1 or string 'Yes' (case-insensitive). "
        f"Got classes_={classes.tolist()!r}."
    )


def score_batch(df_features: pd.DataFrame, pipeline=None):
    """
    Return per-row probability of the positive churn class (see _positive_churn_column_index).

    df_features: output of preprocess_input (no customerID, same columns as training).
    """
    if pipeline is None:
        pipeline = load_pipeline()

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "Scoring requires predict_proba; "
            f"estimator {type(pipeline).__name__!r} has no predict_proba."
        )

    _assert_feature_names_aligned(pipeline, df_features)

    classes = _estimator_classes(pipeline)
    proba = pipeline.predict_proba(df_features)

    if proba.ndim != 2:
        raise ValueError(
            f"predict_proba must return a 2D array; got shape {proba.shape!r}."
        )
    if proba.shape[1] != classes.size:
        raise ValueError(
            f"predict_proba column count ({proba.shape[1]}) must match len(classes_) ({classes.size})."
        )

    col_idx = _positive_churn_column_index(classes)
    return proba[:, col_idx]

"""Tests for feature_names_in_ alignment before predict_proba."""
import numpy as np
import pandas as pd
import pytest

from src.scoring.score_batch import score_batch


class _FakePipelineAligned:
    def __init__(self, feature_names, proba_row):
        self.feature_names_in_ = np.asarray(feature_names)
        self.classes_ = np.array([0, 1])
        self._proba = np.asarray(proba_row, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        if self._proba.shape[0] == 1:
            return np.repeat(self._proba, n, axis=0)
        assert self._proba.shape[0] == n
        return self._proba


def test_score_batch_allows_matching_columns_order():
    cols = ["a", "b"]
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    fake = _FakePipelineAligned(cols, [[0.2, 0.8]])
    out = score_batch(df, pipeline=fake)
    np.testing.assert_allclose(out, [0.8])


def test_score_batch_rejects_column_order_mismatch():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    fake = _FakePipelineAligned(["b", "a"], [[0.2, 0.8]])
    with pytest.raises(ValueError, match="Preprocessed feature columns"):
        score_batch(df, pipeline=fake)


def test_score_batch_rejects_column_name_mismatch():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    fake = _FakePipelineAligned(["a", "c"], [[0.2, 0.8]])
    with pytest.raises(ValueError, match="Model expects"):
        score_batch(df, pipeline=fake)


def test_score_batch_skips_alignment_when_no_feature_names_in():
    class NoNames:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            n = len(X)
            return np.tile([0.3, 0.7], (n, 1))

    df = pd.DataFrame({"x": [1], "y": [2]})
    out = score_batch(df, pipeline=NoNames())
    assert out.shape == (1,)

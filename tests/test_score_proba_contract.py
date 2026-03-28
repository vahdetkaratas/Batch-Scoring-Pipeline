"""Tests for binary churn probability column selection (score_batch)."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.scoring.score_batch import (
    _positive_churn_column_index,
    score_batch,
)


class _FakeBinaryEstimator:
    def __init__(self, classes, proba_out):
        self.classes_ = np.asarray(classes)
        self._proba = np.asarray(proba_out, dtype=float)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        if self._proba.shape[0] == 1:
            return np.repeat(self._proba, n, axis=0)
        assert self._proba.shape[0] == n
        return self._proba


def test_positive_index_label_1_second_column():
    assert _positive_churn_column_index(np.array([0, 1])) == 1


def test_positive_index_label_1_first_column():
    assert _positive_churn_column_index(np.array([1, 0])) == 0


def test_positive_index_yes_no_strings():
    assert _positive_churn_column_index(np.array(["No", "Yes"])) == 1


def test_positive_index_yes_no_reversed():
    assert _positive_churn_column_index(np.array(["Yes", "No"])) == 0


def test_positive_index_rejects_three_classes():
    with pytest.raises(ValueError, match="exactly 2"):
        _positive_churn_column_index(np.array([0, 1, 2]))


def test_positive_index_rejects_unsupported_binary_labels():
    with pytest.raises(ValueError, match="integer 1 or string 'Yes'"):
        _positive_churn_column_index(np.array(["a", "b"]))


def test_score_batch_selects_column_matching_classes_order():
    # classes [1, 0] -> positive churn is label 1 -> first proba column
    proba_row = np.array([[0.25, 0.75]])
    fake = _FakeBinaryEstimator([1, 0], proba_row)
    df = pd.DataFrame({"x": [1]})
    out = score_batch(df, pipeline=fake)
    assert out.shape == (1,)
    np.testing.assert_allclose(out, [0.25])


def test_score_batch_matches_second_column_when_classes_0_1():
    proba_row = np.array([[0.9, 0.1]])
    fake = _FakeBinaryEstimator([0, 1], proba_row)
    df = pd.DataFrame({"x": [1]})
    out = score_batch(df, pipeline=fake)
    np.testing.assert_allclose(out, [0.1])


def test_score_batch_rejects_wrong_proba_width():
    fake = _FakeBinaryEstimator([0, 1], np.array([[0.5, 0.3, 0.2]]))
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="column count"):
        score_batch(df, pipeline=fake)


def test_score_batch_rejects_no_classes():
    bad = MagicMock(spec=["predict_proba"])
    bad.predict_proba = lambda X: np.array([[0.5, 0.5]])
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="classes_"):
        score_batch(df, pipeline=bad)


def test_score_batch_rejects_no_predict_proba():
    bad = MagicMock(spec=["classes_"])
    bad.classes_ = np.array([0, 1])
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="predict_proba"):
        score_batch(df, pipeline=bad)


def test_score_batch_rejects_non_2d_proba():
    fake = _FakeBinaryEstimator([0, 1], np.array([0.5, 0.5]))

    def bad_proba(X):
        return np.array([0.5, 0.5])

    fake.predict_proba = bad_proba
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="2D"):
        score_batch(df, pipeline=fake)

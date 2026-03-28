"""
Unit and component tests. Optional checks use data/input_batches/batch_001.csv when present.

Committed end-to-end scoring proof (no skip): see tests/test_e2e_scoring.py + tests/fixtures/.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_batch import load_batch
from src.validation.input_checks import validate_batch
from src.scoring.preprocess_input import preprocess_input
from src.scoring.postprocess_predictions import postprocess_predictions


BATCH_001 = Path("data/input_batches/batch_001.csv")


def test_load_batch_returns_dataframe():
    """load_batch returns DataFrame with expected columns when file exists."""
    if not BATCH_001.exists():
        pytest.skip("batch_001.csv not found")
    df = load_batch(BATCH_001)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "customerID" in df.columns
    assert "tenure" in df.columns


def test_validate_batch_accepts_valid_batch():
    """validate_batch does not raise for valid batch."""
    if not BATCH_001.exists():
        pytest.skip("batch_001.csv not found")
    df = load_batch(BATCH_001)
    validate_batch(df)


def test_validate_batch_rejects_empty():
    """validate_batch raises for empty DataFrame."""
    with pytest.raises(ValueError, match="empty"):
        validate_batch(pd.DataFrame())


def test_validate_batch_rejects_missing_columns():
    """validate_batch raises when required columns missing."""
    df = pd.DataFrame({"customerID": ["a"], "tenure": [1]})
    with pytest.raises(ValueError, match="Missing required"):
        validate_batch(df)


def test_validate_batch_rejects_duplicate_customer_id():
    """validate_batch raises for duplicate customerID."""
    if not BATCH_001.exists():
        pytest.skip("batch_001.csv not found")
    df = load_batch(BATCH_001)
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_batch(df)


def test_preprocess_input_has_feature_columns():
    """preprocess_input returns DataFrame with num_active_services and tenure_group."""
    if not BATCH_001.exists():
        pytest.skip("batch_001.csv not found")
    df = load_batch(BATCH_001)
    out = preprocess_input(df)
    assert "num_active_services" in out.columns
    assert "tenure_group" in out.columns
    assert "customerID" not in out.columns


def test_postprocess_predictions_output_schema():
    """postprocess_predictions output has customer_id, churn_score, predicted_label, model_version, scoring_timestamp."""
    ids = pd.Series(["a", "b"])
    scores = [0.2, 0.8]
    out = postprocess_predictions(ids, scores, threshold=0.5)
    assert list(out.columns) == [
        "customer_id",
        "churn_score",
        "predicted_label",
        "model_version",
        "scoring_timestamp",
    ]
    assert out["predicted_label"].tolist() == [0, 1]
    assert len(out) == 2


def test_postprocess_predicted_label_respects_threshold():
    """predicted_label is 1 when churn_score >= threshold."""
    ids = pd.Series(["x"])
    out = postprocess_predictions(ids, [0.35], threshold=0.35)
    assert out["predicted_label"].iloc[0] == 1
    out2 = postprocess_predictions(ids, [0.34], threshold=0.35)
    assert out2["predicted_label"].iloc[0] == 0


def test_main_returns_1_on_missing_input(tmp_path):
    from src.pipeline.run_batch_scoring import main

    code = main(["--input", str(tmp_path / "nope.csv"), "-q"])
    assert code == 1


def test_cli_help_exits_zero():
    r = subprocess.run(
        [sys.executable, "-m", "src.pipeline.run_batch_scoring", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert r.returncode == 0
    assert "--input" in r.stdout or "--input" in (r.stderr or "")

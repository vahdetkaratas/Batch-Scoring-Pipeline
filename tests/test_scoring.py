"""
Tests for batch scoring: load_batch, validate_batch, preprocess, score, postprocess.
Pipeline integration tests.
"""
import json
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


def test_run_batch_scoring_output_row_count_and_schema():
    """Full pipeline: output row count matches input; output has churn_score and required columns."""
    from src.pipeline.run_batch_scoring import run_batch_scoring

    if not BATCH_001.exists():
        pytest.skip("batch_001.csv not found")
    model_path = Path("models/churn_model.joblib")
    if not model_path.exists():
        pytest.skip("churn_model.joblib not found; copy from Churn project")
    out = run_batch_scoring(input_path=BATCH_001)
    df_in = load_batch(BATCH_001)
    assert len(out) == len(df_in)
    assert "churn_score" in out.columns
    assert "customer_id" in out.columns
    assert "predicted_label" in out.columns
    assert "model_version" in out.columns
    assert "scoring_timestamp" in out.columns


def test_run_batch_scoring_write_manifest(tmp_path):
    """Optional .meta.json next to output."""
    from src.pipeline.run_batch_scoring import run_batch_scoring

    if not BATCH_001.exists():
        pytest.skip("batch_001.csv not found")
    model_path = Path("models/churn_model.joblib")
    if not model_path.exists():
        pytest.skip("churn_model.joblib not found")
    out_csv = tmp_path / "scored.csv"
    run_batch_scoring(
        input_path=BATCH_001,
        output_path=out_csv,
        write_manifest=True,
    )
    meta = tmp_path / "scored.meta.json"
    assert meta.exists()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["row_count"] == len(load_batch(BATCH_001))
    assert "input_path" in data and "threshold" in data and "model_version" in data


def test_main_returns_1_on_missing_input(tmp_path):
    from src.pipeline.run_batch_scoring import main

    code = main(["--input", str(tmp_path / "nope.csv"), "-q"])
    assert code == 1


def test_main_returns_0_when_pipeline_ok():
    from src.pipeline.run_batch_scoring import main

    if not BATCH_001.exists() or not Path("models/churn_model.joblib").exists():
        pytest.skip("need batch and model")
    out = Path("data/scored_outputs/scored_batch_001.csv")
    code = main(["-i", str(BATCH_001), "-o", str(out), "-q"])
    assert code == 0


def test_cli_help_exits_zero():
    r = subprocess.run(
        [sys.executable, "-m", "src.pipeline.run_batch_scoring", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert r.returncode == 0
    assert "--input" in r.stdout or "--input" in (r.stderr or "")

"""
Committed fixture E2E: full batch scoring path with no pytest.skip.
"""
import json
from pathlib import Path

import pandas as pd

from src.data.load_batch import load_batch
from src.pipeline.run_batch_scoring import main, run_batch_scoring

FIXTURES = Path(__file__).resolve().parent / "fixtures"
E2E_INPUT = FIXTURES / "e2e_input.csv"
E2E_MODEL = FIXTURES / "e2e_pipeline.joblib"

EXPECTED_OUT_COLS = [
    "customer_id",
    "churn_score",
    "predicted_label",
    "model_version",
    "scoring_timestamp",
]

THRESHOLD = 0.5
MODEL_VERSION = "e2e_fixture_v1"


def test_fixture_files_committed():
    assert E2E_INPUT.is_file(), f"Missing {E2E_INPUT}"
    assert E2E_MODEL.is_file(), f"Missing {E2E_MODEL} — run: python scripts/build_e2e_fixture.py"


def test_e2e_run_batch_scoring_csv(tmp_path):
    out_csv = tmp_path / "scored.csv"
    run_batch_scoring(
        input_path=E2E_INPUT,
        output_path=out_csv,
        model_path=E2E_MODEL,
        threshold=THRESHOLD,
        model_version=MODEL_VERSION,
        write_manifest=False,
    )

    assert out_csv.is_file() and out_csv.stat().st_size > 0

    df_in = load_batch(E2E_INPUT)
    df_out = pd.read_csv(out_csv)

    assert list(df_out.columns) == EXPECTED_OUT_COLS
    assert len(df_out) == len(df_in)
    assert df_out["customer_id"].tolist() == df_in["customerID"].tolist()

    scores = df_out["churn_score"].astype(float)
    assert scores.ge(-1e-9).all() and scores.le(1.0 + 1e-9).all()

    labels = df_out["predicted_label"].astype(int)
    assert labels.isin([0, 1]).all()
    expected_labels = (scores >= THRESHOLD).astype(int)
    assert labels.tolist() == expected_labels.tolist()

    assert (df_out["model_version"] == MODEL_VERSION).all()

    # ISO-8601 UTC shape (non-empty, stable format)
    ts = df_out["scoring_timestamp"].astype(str)
    assert ts.str.len().ge(10).all()
    assert ts.str.endswith("+00:00").all()


def test_e2e_run_batch_scoring_manifest(tmp_path):
    out_csv = tmp_path / "scored_manifest.csv"
    run_batch_scoring(
        input_path=E2E_INPUT,
        output_path=out_csv,
        model_path=E2E_MODEL,
        threshold=THRESHOLD,
        model_version=MODEL_VERSION,
        write_manifest=True,
    )

    meta_path = tmp_path / "scored_manifest.meta.json"
    assert meta_path.is_file()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["row_count"] == len(load_batch(E2E_INPUT))
    assert data["threshold"] == THRESHOLD
    assert data["model_version"] == MODEL_VERSION
    for key in ("input_path", "output_path", "model_path", "scoring_timestamp"):
        assert key in data


def test_e2e_cli_main_exit_zero(tmp_path):
    out_csv = tmp_path / "cli_scored.csv"
    code = main(
        [
            "-i",
            str(E2E_INPUT),
            "-o",
            str(out_csv),
            "-m",
            str(E2E_MODEL),
            "-t",
            str(THRESHOLD),
            "--model-version",
            MODEL_VERSION,
            "-q",
        ]
    )
    assert code == 0
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    assert len(df) == len(load_batch(E2E_INPUT))

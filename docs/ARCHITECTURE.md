# Architecture

## Repository layout

```
batch-scoring-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── Makefile
├── .gitignore
├── config/config.yaml
├── data/
│   ├── input_batches/batch_001.csv   # demo input (tracked)
│   └── scored_outputs/               # pipeline output (gitignored except .gitkeep)
├── models/                           # churn_model.joblib (gitignored; .gitkeep keeps folder)
├── notebooks/01_scoring_output_review.ipynb
├── scripts/
│   ├── generate_sample_batch.py      # demo CSV for data/input_batches/
│   └── build_e2e_fixture.py          # regenerates tests/fixtures/e2e_pipeline.joblib
├── src/
│   ├── data/load_batch.py
│   ├── validation/input_checks.py
│   ├── scoring/{preprocess_input,score_batch,postprocess_predictions}.py
│   ├── pipeline/run_batch_scoring.py
│   └── utils/io_helpers.py
├── tests/
│   ├── fixtures/                     # e2e_input.csv, e2e_pipeline.joblib (tracked)
│   ├── test_scoring.py               # components; optional skips if batch_001 missing
│   ├── test_e2e_scoring.py         # full path, no pytest.skip
│   ├── test_input_contract.py      # TotalCharges / tenure vs preprocess
│   ├── test_score_proba_contract.py # positive class from classes_
│   └── test_score_feature_alignment.py  # feature_names_in_ vs preprocess columns
├── docs/
└── vercel_static/                    # optional portfolio landing page
```

## Pipeline

```
CSV → validate → preprocess (training-aligned features) → predict_proba → postprocess → CSV
```

## Modules

| Path | Responsibility |
|------|----------------|
| `src/data/load_batch.py` | Load input CSV |
| `src/validation/input_checks.py` | Required columns, non-empty batch, unique `customerID`; strict numerics (incl. `TotalCharges`); `tenure` in [0, 72] as whole months |
| `src/scoring/preprocess_input.py` | Feature engineering consistent with churn training |
| `src/scoring/score_batch.py` | Load joblib pipeline; optional `feature_names_in_` alignment; binary positive-class `predict_proba`; score rows |
| `src/scoring/postprocess_predictions.py` | Output columns, thresholded label, timestamp |
| `src/pipeline/run_batch_scoring.py` | Orchestration, CLI (`argparse`), optional `.meta.json` |
| `src/utils/io_helpers.py` | Write scored CSV and optional run manifest |

## Configuration

`config/config.yaml` — optional defaults: `paths.input_batch`, `paths.output_csv`, `paths.model`, `scoring.threshold`, `scoring.model_version`. CLI and Python arguments override these.

## Dependencies

| Package | Role |
|---------|------|
| `pandas`, `scikit-learn`, `joblib` | Core scoring |
| `PyYAML` | Config file |
| `pytest` | Tests under `tests/` (E2E fixture, input/score contracts, components) |
| `jupyter`, `matplotlib` | Review notebook |

**Python:** 3.10+ recommended (3.13 tested).

## Model artifact

Place a trained churn **pipeline** (joblib) at `models/churn_model.joblib`. Training preprocessing must match `preprocess_input.py`.

## Related assets

- **Notebook:** validates scored CSV, row alignment with input, score histogram.
- **`scripts/generate_sample_batch.py`:** regenerates demo `batch_001.csv`.
- **`scripts/build_e2e_fixture.py`:** regenerates committed E2E model `tests/fixtures/e2e_pipeline.joblib` after fixture CSV or preprocess changes.

Release checklist: **[PUBLISHING.md](PUBLISHING.md)**.

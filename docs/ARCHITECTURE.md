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
├── scripts/generate_sample_batch.py
├── src/
│   ├── data/load_batch.py
│   ├── validation/input_checks.py
│   ├── scoring/{preprocess_input,score_batch,postprocess_predictions}.py
│   ├── pipeline/run_batch_scoring.py
│   └── utils/io_helpers.py
├── tests/test_scoring.py
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
| `src/validation/input_checks.py` | Required columns, non-empty batch, unique `customerID`, numeric fields |
| `src/scoring/preprocess_input.py` | Feature engineering consistent with churn training |
| `src/scoring/score_batch.py` | Load joblib pipeline, score rows |
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
| `pytest` | Tests (`tests/test_scoring.py`) |
| `jupyter`, `matplotlib` | Review notebook |

**Python:** 3.10+ recommended (3.13 tested).

## Model artifact

Place a trained churn **pipeline** (joblib) at `models/churn_model.joblib`. Training preprocessing must match `preprocess_input.py`.

## Related assets

- **Notebook:** validates scored CSV, row alignment with input, score histogram.
- **`scripts/generate_sample_batch.py`:** regenerates demo `batch_001.csv`.

Release checklist: **[PUBLISHING.md](PUBLISHING.md)**.

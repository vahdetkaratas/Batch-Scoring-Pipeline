# Batch Scoring Pipeline

Production-style batch inference for tabular churn data: validate input CSV, align features with training-time preprocessing, score with a serialized sklearn pipeline, and write a versioned scored output CSV.

This project is intentionally focused on **offline / scheduled scoring** (ETL-style workloads), not low-latency request/response serving.

**License:** [LICENSE](LICENSE) (MIT)  
**Docs:** [docs/README.md](docs/README.md) · [Architecture](docs/ARCHITECTURE.md)

---

## What This Project Delivers

- A single orchestration entrypoint: `run_batch_scoring(...)`
- Strict input validation before scoring
- Training-aligned preprocessing in the scoring path
- Probability + label output with model version and timestamp
- CLI-friendly execution with clear exit codes
- Test coverage including a committed end-to-end fixture run

---

## Requirements

- **Python 3.10+** (3.13 tested)
- A trained sklearn pipeline at **`models/churn_model.joblib`** (not committed here)

This repository is scoring-only. Keep `src/scoring/preprocess_input.py` aligned with how your model was trained.

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: regenerate sample batch input:

```bash
python scripts/generate_sample_batch.py --rows 120
```

Run scoring from repo root:

```bash
python -m src.pipeline.run_batch_scoring
```

or:

```bash
make score
```

Defaults are read from `config/config.yaml` when present.

---

## CLI Usage

```bash
python -m src.pipeline.run_batch_scoring --help
```

Common flags:
- `-i`, `--input`
- `-o`, `--output`
- `-t`, `--threshold`
- `-m`, `--model-path`
- `--model-version`
- `--config`
- `--write-manifest`
- `-q`, `-v`

Exit codes:
- `0` success
- `1` failure

---

## Input / Output Contract

**Input (Telco-style):**  
`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

**Output columns:**

| Column | Description |
|--------|-------------|
| `customer_id` | Identifier |
| `churn_score` | Probability in `[0, 1]` |
| `predicted_label` | `1` if score >= threshold |
| `model_version` | Version tag written per run |
| `scoring_timestamp` | UTC ISO timestamp |

---

## Python API

```python
from src.pipeline.run_batch_scoring import run_batch_scoring

run_batch_scoring(
    input_path="data/input_batches/batch_001.csv",
    output_path="data/scored_outputs/scored_batch_001.csv",
    write_manifest=True,
)
```

---

## Validation & Review

- Output review notebook: `notebooks/01_scoring_output_review.ipynb`
- Expected default output path: `data/scored_outputs/scored_batch_001.csv`

---

## Tests

```bash
make test
python -m pytest tests/ -v
```

`tests/test_e2e_scoring.py` executes the full path (`load_batch` -> validate -> preprocess -> pipeline -> `predict_proba` -> postprocess -> CSV) using committed fixtures:
- `tests/fixtures/e2e_input.csv`
- `tests/fixtures/e2e_pipeline.joblib`

After changing fixture rows or preprocessing schema, regenerate fixture model:

```bash
python scripts/build_e2e_fixture.py
```

---

## Further Reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

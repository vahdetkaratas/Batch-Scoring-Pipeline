# Batch Scoring Pipeline

Batch inference for tabular churn: validate a CSV, align features with the training pipeline, score with a joblib model, write a versioned scored CSV. Intended for scheduled / ETL-style workloads, not low-latency APIs.

**Docs:** [docs/README.md](docs/README.md) · [Architecture](docs/ARCHITECTURE.md) · [Publishing](docs/PUBLISHING.md)  
**License:** [LICENSE](LICENSE) (MIT)

---

## Requirements

- **Python 3.10+** (3.13 tested)
- Trained pipeline at **`models/churn_model.joblib`** (copy from your churn training project)

---

## Setup

```bash
pip install -r requirements.txt
```

Regenerate demo input (optional):

```bash
python scripts/generate_sample_batch.py --rows 120
```

---

## Run

From the repository root:

```bash
python -m src.pipeline.run_batch_scoring
make score
```

Defaults come from `config/config.yaml` when present.

**CLI:** `python -m src.pipeline.run_batch_scoring --help`  
Flags include `-i` / `-o`, `-t` (threshold), `-m` (model path), `--model-version`, `--config`, `--write-manifest`, `-q` / `-v`. Exit **0** on success, **1** on failure.

**Notebook:** run the pipeline once, then open `notebooks/01_scoring_output_review.ipynb` (expects `data/scored_outputs/scored_batch_001.csv` from the default config paths).

---

## Schemas

**Input:** Telco-style churn columns — `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.

**Output:**

| Column | Description |
|--------|-------------|
| `customer_id` | Identifier |
| `churn_score` | Probability in [0, 1] |
| `predicted_label` | 1 if score ≥ threshold |
| `model_version` | Version string |
| `scoring_timestamp` | UTC ISO timestamp |

---

## Tests

```bash
make test
python -m pytest tests/ -v
```

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

## Further reading

- [docs/PRE_PUSH_CHECKLIST.md](docs/PRE_PUSH_CHECKLIST.md) — before first push
- [docs/PUBLISHING.md](docs/PUBLISHING.md) — GitHub & Vercel
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map & layout

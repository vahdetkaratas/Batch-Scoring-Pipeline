# Publishing & distribution

**Before the first push:** [PRE_PUSH_CHECKLIST.md](PRE_PUSH_CHECKLIST.md)

## Repository contents (recommended)

| Track in Git | Omit / gitignored |
|--------------|-------------------|
| `src/`, `tests/`, `config/`, `Makefile`, `.gitignore`, `LICENSE` | `models/*.joblib` |
| `requirements.txt`, `README.md`, `docs/` | `data/scored_outputs/*.csv` (regenerate with pipeline) |
| `data/input_batches/batch_001.csv` | `0_meta/`, `freelance.md` (local planning; see root `.gitignore`) |
| `notebooks/`, `scripts/`, `vercel_static/` | — |

After clone: install deps, add `models/churn_model.joblib`, run `python -m src.pipeline.run_batch_scoring`, then open the notebook if needed.

## Initial GitHub push

```bash
git init
git add .
git status
git commit -m "Initial commit: Batch Scoring Pipeline"
git remote add origin https://github.com/vahdetkaratas/Batch-Scoring-Pipeline.git
git branch -M main
git push -u origin main
```

**Repository settings (suggestions)**

- **Description:** *Batch inference: score CSV batches with a trained churn model — validation, preprocessing, scored CSV output.*
- **Topics:** `machine-learning`, `batch-inference`, `scikit-learn`, `python`

## Vercel (static site)

1. Connect the repository in Vercel.
2. **Root Directory:** `vercel_static`.
3. `vercel_static/index.html` links point to this repo; update if you fork.
4. Adjust `blob/main/` in notebook URLs if the default branch is not `main`.

See `vercel_static/README.md`.

## Model binary

Prefer not committing `churn_model.joblib` to keep the repo small and avoid binary drift. Document the copy step in README; cloners place the file under `models/`.

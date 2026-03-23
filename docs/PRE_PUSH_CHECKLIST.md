# Pre-push checklist

Quick checks before publishing the repository or redeploying the static site.

- [ ] `python -m pytest tests/ -v` passes (or `make test`).
- [ ] `python -m src.pipeline.run_batch_scoring --help` runs without error.
- [ ] `models/churn_model.joblib` is **not** committed (see `.gitignore`); README still explains how cloners obtain it.
- [ ] `vercel_static/index.html`: canonical / `og:url` match your live domain (`batch-scoring.vahdetkaratas.com` or fork URL).
- [ ] GitHub links (`View on GitHub`, notebook `blob/main/...`) match your default branch and username/org.

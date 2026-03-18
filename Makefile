# Batch Scoring Pipeline — optional shortcuts
.PHONY: score test

score:
	python -m src.pipeline.run_batch_scoring

test:
	python -m pytest tests/ -v

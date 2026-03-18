"""
Orchestrate batch scoring: load -> validate -> preprocess -> score -> postprocess -> save.
Optional: config/config.yaml; CLI flags when run as __main__.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data.load_batch import load_batch
from src.validation.input_checks import validate_batch
from src.scoring.preprocess_input import preprocess_input
from src.scoring.score_batch import load_pipeline, score_batch
from src.scoring.postprocess_predictions import postprocess_predictions
from src.utils.io_helpers import save_scored_output, write_run_manifest

logger = logging.getLogger(__name__)

INPUT_BATCHES_DIR = Path("data/input_batches")
CONFIG_PATH = Path("config/config.yaml")


def _load_config(path: Path | None = None) -> dict:
    """Load optional config YAML; return empty dict if missing."""
    p = path or CONFIG_PATH
    if not p.exists():
        return {}
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_batch_scoring(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    threshold: float | None = None,
    model_version: str | None = None,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    write_manifest: bool = False,
) -> pd.DataFrame:
    """
    Run full pipeline: load batch CSV -> validate -> preprocess -> score -> postprocess -> save.
    Config file defaults apply when arguments are omitted.
    Set write_manifest=True to emit a .meta.json next to the output CSV.
    """
    cfg = _load_config(Path(config_path) if config_path else None)
    paths_cfg = cfg.get("paths") or {}
    scoring_cfg = cfg.get("scoring") or {}

    input_path = Path(input_path or paths_cfg.get("input_batch") or INPUT_BATCHES_DIR / "batch_001.csv")
    output_path = output_path or paths_cfg.get("output_csv")
    threshold = threshold if threshold is not None else scoring_cfg.get("threshold", 0.35)
    model_version = model_version or scoring_cfg.get("model_version", "churn_v1")
    model_path = model_path or paths_cfg.get("model")

    logger.info("Loading batch from %s", input_path)
    df_raw = load_batch(input_path)
    validate_batch(df_raw)

    df_features = preprocess_input(df_raw)
    pipeline = load_pipeline(model_path=model_path)
    resolved_model = (Path(model_path) if model_path else Path("models/churn_model.joblib")).resolve()

    proba = score_batch(df_features, pipeline=pipeline)

    out = postprocess_predictions(
        df_raw["customerID"],
        proba,
        threshold=threshold,
        model_version=model_version,
    )

    written = save_scored_output(out, path=output_path)
    logger.info("Saved %s rows to %s", len(out), written)

    if write_manifest:
        ts = str(out["scoring_timestamp"].iloc[0])
        meta_path = write_run_manifest(
            written,
            input_path=str(input_path.resolve()),
            output_path=str(written.resolve()),
            row_count=len(out),
            threshold=threshold,
            model_version=model_version,
            model_path=str(resolved_model),
            scoring_timestamp=ts,
        )
        logger.info("Wrote run manifest %s", meta_path)

    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch churn scoring: CSV in, scored CSV out.",
    )
    p.add_argument("-i", "--input", dest="input_path", help="Input batch CSV path")
    p.add_argument("-o", "--output", dest="output_path", help="Output scored CSV path")
    p.add_argument("-t", "--threshold", type=float, help="Classification threshold (default: config or 0.35)")
    p.add_argument("-m", "--model", dest="model_path", help="Path to churn_model.joblib")
    p.add_argument("--model-version", dest="model_version", help="Label stored in output (default: config or churn_v1)")
    p.add_argument("--config", dest="config_path", help="Path to config YAML")
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write .meta.json next to output (input path, threshold, row count, etc.)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Only warnings and errors")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)

    try:
        run_batch_scoring(
            input_path=args.input_path,
            output_path=args.output_path,
            threshold=args.threshold,
            model_version=args.model_version,
            model_path=args.model_path,
            config_path=args.config_path,
            write_manifest=args.write_manifest,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1
    except Exception:
        logger.exception("Batch scoring failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

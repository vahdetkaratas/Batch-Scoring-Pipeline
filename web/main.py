"""
FastAPI shell for the portfolio page: templates, static assets, /api/score, /api/download/{job_id}.
"""
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.run_batch_scoring import run_batch_scoring

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERCEL_STATIC = _REPO_ROOT / "vercel_static"
_TEMPLATES = Path(__file__).resolve().parent / "templates"

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MiB
JOB_TTL_SECONDS = 3600
_PREVIEW_ROWS = 10

_TMP_PARENT = Path(tempfile.gettempdir()) / "batch_scoring_demo"
_TMP_PARENT.mkdir(parents=True, exist_ok=True)

# job_id -> metadata for a future download route (paths + timestamps)
_jobs: dict[str, dict] = {}


def _default_model_path() -> Path:
    env = os.environ.get("DEMO_MODEL_PATH", "").strip()
    if env:
        p = Path(env)
        return p if p.is_absolute() else (_REPO_ROOT / p).resolve()
    return (_REPO_ROOT / "tests" / "fixtures" / "e2e_pipeline.joblib").resolve()


def _cleanup_job_files(job_id: str) -> None:
    meta = _jobs.pop(job_id, None)
    if not meta:
        return
    for key in ("in_path", "out_path"):
        p = meta.get(key)
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def _sweep_expired_jobs() -> None:
    now = time.time()
    for jid in list(_jobs.keys()):
        meta = _jobs[jid]
        if now - meta.get("created", 0) <= JOB_TTL_SECONDS:
            continue
        for key in ("in_path", "out_path"):
            p = meta.get(key)
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
        del _jobs[jid]


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_VERCEL_STATIC)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATES))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/api/download/{job_id}")
async def api_download(job_id: str, background_tasks: BackgroundTasks):
    _sweep_expired_jobs()
    meta = _jobs.get(job_id)
    if not meta:
        return JSONResponse(
            status_code=404,
            content={
                "error": "No scored file for this job. It may have expired or already been downloaded.",
                "code": "not_found",
            },
        )
    out_path = Path(meta["out_path"])
    if not out_path.is_file():
        _jobs.pop(job_id, None)
        return JSONResponse(
            status_code=404,
            content={
                "error": "Scored output is no longer available.",
                "code": "not_found",
            },
        )

    background_tasks.add_task(_cleanup_job_files, job_id)

    return FileResponse(
        path=out_path,
        filename="scored_batch.csv",
        media_type="text/csv",
    )


@app.post("/api/score")
async def api_score(
    file: UploadFile = File(...),
    threshold: float = Form(0.35),
    model_version: str = Form("e2e_web"),
):
    _sweep_expired_jobs()

    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "A CSV file upload is required.", "code": "no_filename"},
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"error": "Uploaded file is empty.", "code": "empty"},
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": f"Upload exceeds maximum size ({MAX_UPLOAD_BYTES} bytes).",
                "code": "too_large",
                "max_bytes": MAX_UPLOAD_BYTES,
            },
        )
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={"error": "Only .csv uploads are accepted.", "code": "not_csv"},
        )

    model_path = _default_model_path()
    if not model_path.is_file():
        logger.error("Demo model missing at %s", model_path)
        raise HTTPException(
            status_code=503,
            detail={"error": "Demo model is not available on the server.", "code": "no_model"},
        )

    job_id = str(uuid.uuid4())
    in_path = _TMP_PARENT / f"{job_id}_in.csv"
    out_path = _TMP_PARENT / f"{job_id}_out.csv"

    try:
        in_path.write_bytes(raw)
    except OSError:
        logger.exception("Failed to write upload to %s", in_path)
        raise HTTPException(
            status_code=500,
            detail={"error": "Could not store the upload.", "code": "write_failed"},
        )

    try:
        out_df = run_batch_scoring(
            input_path=in_path,
            output_path=out_path,
            model_path=model_path,
            threshold=threshold,
            model_version=model_version,
            write_manifest=False,
        )
    except ValueError as e:
        logger.info("Scoring validation failed: %s", e)
        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "code": "scoring_validation"},
        ) from e
    except FileNotFoundError as e:
        logger.info("Scoring file error: %s", e)
        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "code": "file_not_found"},
        ) from e
    except Exception:
        logger.exception("Scoring failed unexpectedly")
        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail={"error": "Scoring failed.", "code": "internal"},
        ) from None

    _jobs[job_id] = {
        "created": time.time(),
        "in_path": str(in_path),
        "out_path": str(out_path),
        "model_version": model_version,
        "threshold": threshold,
    }

    preview = json.loads(
        out_df.head(_PREVIEW_ROWS).to_json(orient="records", date_format="iso")
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "row_count": int(len(out_df)),
            "columns": list(out_df.columns),
            "preview": preview,
            "model_version": model_version,
            "threshold": threshold,
        }
    )

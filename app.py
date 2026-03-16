"""
╔══════════════════════════════════════════════════════════════╗
║       USSD KPI Extractor  —  Web Application Backend        ║
║       Stack : FastAPI + PaddleOCR/Tesseract + PyGithub      ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║   GET  /                 → Web UI                            ║
║   POST /upload           → Process image → append to CSV    ║
║   GET  /results          → Return full CSV as JSON          ║
║   GET  /download         → Download CSV file                ║
║   POST /sync             → Force push CSV to GitHub         ║
║   GET  /health           → Health check                     ║
╚══════════════════════════════════════════════════════════════╝

Setup:
  1.  Copy your .env file or set environment variables:
        GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
        GITHUB_REPO=username/repo-name
        GITHUB_CSV_PATH=data/kpi_results.csv   (path inside repo)
        GITHUB_BRANCH=main                      (optional, default: main)
        PADDLE_MODEL_DIR=./paddle_models        (optional)

  2.  Run:
        uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse,
                                JSONResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles

# ── load .env if present ──────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── local modules ─────────────────────────────────────────────
import extractor as kpi

# ── GitHub integration ────────────────────────────────────────
try:
    from github import Github, GithubException
    _github_available = True
except ImportError:
    _github_available = False

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO     = os.getenv("GITHUB_REPO", "")           # e.g. "user/repo"
GITHUB_CSV_PATH = os.getenv("GITHUB_CSV_PATH",
                             "data/kpi_results.csv")
GITHUB_BRANCH   = os.getenv("GITHUB_BRANCH", "main")
PADDLE_MODEL_DIR = os.getenv("PADDLE_MODEL_DIR", None)

LOCAL_CSV       = Path("kpi_results.csv")
UPLOAD_DIR      = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CSV_COLUMNS = kpi.CSV_COLUMNS   # shared with extractor

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s │ %(message)s"
)
log = logging.getLogger("ussd_app")

# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="USSD KPI Extractor",
    description="Extract radio KPIs from USSD/ServiceMode screenshots",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════════════════════
# GitHub helper
# ══════════════════════════════════════════════════════════════

class GitHubSync:
    """Handles reading / writing the CSV to a GitHub repository."""

    def __init__(self):
        self._gh   = None
        self._repo = None
        self._sha  = None        # current file SHA (needed for updates)
        self.enabled = False
        self._connect()

    def _connect(self):
        if not _github_available:
            log.warning("PyGithub not installed — GitHub sync disabled.")
            return
        if not GITHUB_TOKEN or not GITHUB_REPO:
            log.warning("GITHUB_TOKEN / GITHUB_REPO not set — sync disabled.")
            return
        try:
            self._gh   = Github(GITHUB_TOKEN)
            self._repo = self._gh.get_repo(GITHUB_REPO)
            self.enabled = True
            log.info(f"✅ GitHub connected: {GITHUB_REPO} [{GITHUB_BRANCH}]")
        except Exception as e:
            log.error(f"GitHub connection failed: {e}")

    # ── Pull CSV from GitHub → local file ─────────────────────
    def pull(self) -> bool:
        if not self.enabled:
            return False
        try:
            contents = self._repo.get_contents(GITHUB_CSV_PATH,
                                               ref=GITHUB_BRANCH)
            self._sha = contents.sha
            decoded  = contents.decoded_content.decode("utf-8")
            LOCAL_CSV.write_text(decoded, encoding="utf-8")
            log.info(f"📥 Pulled CSV from GitHub ({len(decoded)} bytes)")
            return True
        except Exception as e:
            # File may not exist yet — that's OK
            log.info(f"GitHub pull: {e} (will create on first push)")
            self._sha = None
            return False

    # ── Push local CSV → GitHub ────────────────────────────────
    def push(self, commit_message: str = "Update KPI results") -> bool:
        if not self.enabled:
            return False
        if not LOCAL_CSV.exists():
            return False
        try:
            content = LOCAL_CSV.read_text(encoding="utf-8")
            if self._sha:
                # Update existing file
                result = self._repo.update_file(
                    path=GITHUB_CSV_PATH,
                    message=commit_message,
                    content=content,
                    sha=self._sha,
                    branch=GITHUB_BRANCH
                )
            else:
                # Create new file
                result = self._repo.create_file(
                    path=GITHUB_CSV_PATH,
                    message=commit_message,
                    content=content,
                    branch=GITHUB_BRANCH
                )
            self._sha = result["content"].sha
            log.info(f"📤 Pushed CSV to GitHub: {GITHUB_CSV_PATH}")
            return True
        except Exception as e:
            log.error(f"GitHub push failed: {e}")
            return False

    def status(self) -> dict:
        return {
            "enabled":   self.enabled,
            "repo":      GITHUB_REPO if self.enabled else None,
            "branch":    GITHUB_BRANCH if self.enabled else None,
            "csv_path":  GITHUB_CSV_PATH if self.enabled else None,
            "has_remote_file": self._sha is not None,
        }


# Global singleton
github = GitHubSync()


# ══════════════════════════════════════════════════════════════
# CSV helper
# ══════════════════════════════════════════════════════════════

def _ensure_csv():
    """Create CSV with header if it doesn't exist."""
    if not LOCAL_CSV.exists():
        with open(LOCAL_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def _append_row(record: dict):
    """Append one record to the local CSV."""
    _ensure_csv()
    with open(LOCAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(record)


def _read_csv() -> list[dict]:
    """Read all rows from the local CSV."""
    if not LOCAL_CSV.exists():
        return []
    try:
        df = pd.read_csv(LOCAL_CSV, dtype=str).fillna("N/A")
        return df.to_dict(orient="records")
    except Exception:
        return []


def _row_count() -> int:
    if not LOCAL_CSV.exists():
        return 0
    try:
        return sum(1 for _ in open(LOCAL_CSV)) - 1   # minus header
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════
# App startup
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    log.info("🚀 USSD KPI Extractor starting …")

    # Initialise PaddleOCR (or fall back to Tesseract silently)
    kpi._init_paddle(PADDLE_MODEL_DIR)

    # Pull CSV from GitHub (if configured)
    github.pull()

    # Always ensure local CSV exists
    _ensure_csv()

    log.info(f"📄 Local CSV: {LOCAL_CSV.resolve()}  ({_row_count()} existing rows)")
    log.info("✅ App ready.")


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════

# ── Home page ─────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("templates/index.html").read_text(encoding="utf-8")


# ── Health check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "ocr_engine": "PaddleOCR" if kpi._use_paddle else "Tesseract",
        "csv_rows":   _row_count(),
        "github":     github.status(),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    }


# ── Upload & process image ────────────────────────────────────
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Accept an uploaded image, run OCR + KPI extraction,
    append the result to the CSV, push to GitHub.
    Returns the extracted record as JSON.
    """
    # Validate file type
    allowed = {"image/jpeg", "image/jpg", "image/png",
               "image/webp", "image/bmp"}
    if file.content_type and file.content_type.lower() not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   "Please upload JPEG, PNG, WEBP, or BMP."
        )

    # Save to temp file
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    tmp_path = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to save upload: {e}")

    # Run KPI extraction
    try:
        # Override image_name to use original filename
        record = kpi.process_image(tmp_path)
        record["image_name"] = file.filename or tmp_path.name
    except Exception as e:
        log.error(f"Extraction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500,
                            detail=f"KPI extraction failed: {e}")
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Append to CSV
    _append_row(record)

    # Push to GitHub (async-ish: don't fail the request if push fails)
    github_pushed = False
    if github.enabled:
        commit_msg = (
            f"Add KPI result: {record['image_name']} "
            f"[{record['Technology']}] "
            f"RSRP={record['RSRP']} PCI={record['PCI']}"
        )
        github_pushed = github.push(commit_msg)

    return JSONResponse({
        "success":       True,
        "record":        record,
        "csv_row_count": _row_count(),
        "github_pushed": github_pushed,
    })


# ── Get all results ───────────────────────────────────────────
@app.get("/results")
async def get_results():
    rows = _read_csv()
    return JSONResponse({
        "total": len(rows),
        "rows":  rows
    })


# ── Download CSV ──────────────────────────────────────────────
@app.get("/download")
async def download_csv():
    if not LOCAL_CSV.exists():
        raise HTTPException(status_code=404, detail="No CSV file yet.")
    return FileResponse(
        path=str(LOCAL_CSV),
        media_type="text/csv",
        filename="kpi_results.csv"
    )


# ── Force push to GitHub ──────────────────────────────────────
@app.post("/sync")
async def sync_github():
    if not github.enabled:
        raise HTTPException(
            status_code=503,
            detail="GitHub sync is not configured. "
                   "Set GITHUB_TOKEN and GITHUB_REPO environment variables."
        )
    pushed = github.push("Manual sync from web UI")
    if not pushed:
        raise HTTPException(status_code=500, detail="GitHub push failed.")
    return {"success": True, "message": "CSV pushed to GitHub successfully."}


# ── Delete a row ──────────────────────────────────────────────
@app.delete("/results/{image_name}")
async def delete_result(image_name: str):
    if not LOCAL_CSV.exists():
        raise HTTPException(status_code=404, detail="CSV not found.")
    df = pd.read_csv(LOCAL_CSV, dtype=str).fillna("N/A")
    original_len = len(df)
    df = df[df["image_name"] != image_name]
    if len(df) == original_len:
        raise HTTPException(status_code=404,
                            detail=f"No row found for: {image_name}")
    df.to_csv(LOCAL_CSV, index=False)
    if github.enabled:
        github.push(f"Delete row: {image_name}")
    return {"success": True, "deleted": image_name,
            "remaining": len(df)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

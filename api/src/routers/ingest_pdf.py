# src/routers/ingest.py
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ingest", tags=["ingest"])

# In-memory job store (process lifetime). Swap to Postgres/Redis later if needed.
class Job(BaseModel):
    state: Literal["uploaded","parsing","chunking","embedding","indexing","done","error"]
    progress: float = 0.0
    message: str = ""

JOBS: Dict[str, Job] = {}

# Paths and env
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
PDF_DIR = DATA_DIR / "pdfs"
VENV_PY = os.getenv("VENV_PY", "/app/.venv/bin/python")  # your venv python
EMBED_DIM = os.getenv("EMBED_DIM", "768")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

PDF_DIR.mkdir(parents=True, exist_ok=True)

def _run(cmd: list[str], env: Optional[dict] = None):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    out_lines = []
    for line in proc.stdout:  # stream logs if needed
        out_lines.append(line.rstrip())
    code = proc.wait()
    return code, "\n".join(out_lines)

def _pipeline(job_id: str, saved_path: Path):
    job = JOBS[job_id]

    # 1) Parse / Chunk
    job.state, job.progress, job.message = "parsing", 0.15, "Parsing PDF"
    try:
        job.state, job.progress, job.message = "chunking", 0.35, "Chunking"
        code, out = _run([VENV_PY, "/app/scripts/chunk_pdfs.py", str(saved_path)])
        if code != 0:
            raise RuntimeError(f"chunk_pdfs failed:\n{out}")
    except Exception as e:
        job.state, job.message = "error", f"{e}"
        return

    # 2) Embeddings (+ vector indexing)
    job.state, job.progress, job.message = "embedding", 0.65, "Embedding"
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env["OLLAMA_PORT"] = OLLAMA_PORT
    env["EMBED_DIM"] = EMBED_DIM

    try:
        # If your embed script supports filtering by doc_id, use that.
        # Otherwise it will embed only pending chunks (recommended).
        code, out = _run(
            [VENV_PY, "/app/scripts/embed_chunks.py", "--embed-batch", "64"], env=env
        )
        if code != 0:
            raise RuntimeError(f"embed_chunks failed:\n{out}")
    except Exception as e:
        job.state, job.message = "error", f"{e}"
        return

    # 3) (Optional) explicit “indexing” step if you separate it; most setups index inside embed script
    job.state, job.progress, job.message = "indexing", 0.9, "Indexing vectors"
    # nothing to run if embed script already upserts to OpenSearch

    job.state, job.progress, job.message = "done", 1.0, "Completed"

@router.post("/upload")
async def upload(file: UploadFile, background: BackgroundTasks):
    # Basic validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save to /app/data/pdfs/<filename>
    safe_name = file.filename.replace("/", "_")
    target = PDF_DIR / safe_name
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Track job and kick background pipeline
    job_id = uuid.uuid4().hex
    JOBS[job_id] = Job(state="uploaded", progress=0.05, message="File saved")
    background.add_task(_pipeline, job_id=job_id, saved_path=target)
    return {"job_id": job_id}

@router.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

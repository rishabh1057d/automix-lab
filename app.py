"""Local AutoMix workbench."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MAX_FILE = 100 * 1024 * 1024
MAX_BODY = 6 * MAX_FILE + 1024 * 1024
WORKER = ThreadPoolExecutor(max_workers=1)
LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("automix")


def atomic_json(path: Path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def public_job(job):
    return {k: v for k, v in job.items() if k != "future"}


def update_job(job_id, status, progress, message, **extra):
    with LOCK:
        JOBS[job_id].update(status=status, progress=progress, message=message, **extra)
        value = public_job(JOBS[job_id])
        atomic_json(DATA / "jobs" / f"{job_id}.json", value)
    LOG.info("%s %s%% %s", job_id[:8], progress, message)


def run_job(job_id, function):
    try:
        function(job_id)
    except Exception as exc:
        LOG.exception("Job %s failed", job_id)
        update_job(job_id, "error", 0, "Could not finish this job.", error=str(exc)[:500])


def new_job(function):
    with LOCK:
        active = sum(j["status"] not in ("complete", "error") for j in JOBS.values())
        if active >= 3:
            raise HTTPException(429, "Three jobs are already queued. Wait for a render to finish.")
        ident = uuid.uuid4().hex
        JOBS[ident] = dict(id=ident, status="queued", progress=0, message="Waiting for the audio worker.", created=time.time())
    update_job(ident, "queued", 0, "Waiting for the audio worker.")
    WORKER.submit(run_job, ident, function)
    return {"id": ident}


@asynccontextmanager
async def lifespan(app):
    for name in ("uploads", "jobs", "mixes", "analysis"):
        (DATA / name).mkdir(parents=True, exist_ok=True)
    for path in (DATA / "jobs").glob("*.json"):
        try:
            job = json.loads(path.read_text())
            if job["status"] not in ("complete", "error"):
                job.update(status="error", error="Server restarted. Please build the mix again.", message="Interrupted by restart.")
            JOBS[job["id"]] = job
        except (ValueError, KeyError):
            LOG.warning("Ignoring invalid job state %s", path.name)
    yield


app = FastAPI(title="AutoMix Lab", lifespan=lifespan)


class BodyTooLarge(Exception):
    pass


class LocalBoundary:
    """Limit actual streamed bytes before multipart parsing writes to temporary storage."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode()
        if host.split(":")[0] not in ("localhost", "127.0.0.1", "testserver"):
            return await JSONResponse({"detail": "This workbench accepts local requests only."}, 403)(scope, receive, send)
        if scope["method"] in ("POST", "DELETE", "PUT", "PATCH"):
            origin = headers.get(b"origin", b"").decode()
            if origin and origin not in (f"http://{host}", f"https://{host}"):
                return await JSONResponse({"detail": "Cross-origin writes are disabled."}, 403)(scope, receive, send)
        try:
            if int(headers.get(b"content-length", b"0")) > MAX_BODY:
                raise BodyTooLarge()
        except (ValueError, BodyTooLarge):
            return await JSONResponse({"detail": "Upload exceeds the 600 MB total limit."}, 413)(scope, receive, send)
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > MAX_BODY:
                raise BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLarge:
            await JSONResponse({"detail": "Upload exceeds the 600 MB total limit."}, 413)(scope, receive, send)


app.add_middleware(LocalBoundary)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def valid_id(ident):
    if not re.fullmatch(r"[0-9a-f]{32}", ident):
        raise HTTPException(404, "Unknown mix or job.")
    return ident


def result_path(ident):
    path = DATA / "mixes" / valid_id(ident) / "result.json"
    if not path.is_file():
        raise HTTPException(404, "This mix is not ready.")
    return path


@app.get("/api/jobs/{ident}")
def get_job(ident: str):
    valid_id(ident)
    with LOCK:
        job = JOBS.get(ident)
        if not job:
            raise HTTPException(404, "Unknown job.")
        return public_job(job)


def validate_audio(path):
    try:
        info = sf.info(path)
    except Exception as exc:
        raise HTTPException(400, "This file cannot be decoded as audio.") from exc
    if info.channels not in (1, 2):
        raise HTTPException(400, "Only mono and stereo recordings are supported.")
    if info.duration < 2 or info.duration > 600:
        raise HTTPException(400, "Each track must be between 2 seconds and 10 minutes.")
    if info.samplerate < 8000 or info.samplerate > 192000:
        raise HTTPException(400, "Sample rate must be between 8 and 192 kHz.")


def make_mix(job_id, tracks, mode):
    from engine import render_mix
    folder = DATA / "mixes" / job_id
    folder.mkdir()
    atomic_json(folder / "sources.json", [{**track, "path": str(track["path"])} for track in tracks])
    result = render_mix(tracks, mode, folder, DATA / "analysis",
                        lambda stage, percent, message: update_job(
                            job_id, "rendering" if stage == "complete" else stage,
                            min(99, percent), message))
    result.update(id=job_id, created=time.time(),
                  playlist_key=hashlib.sha256("|".join(t["id"] for t in tracks).encode()).hexdigest())
    atomic_json(folder / "result.json", result)
    update_job(job_id, "complete", 100, "Your mix is ready to audition.", result_id=job_id)


@app.post("/api/mixes")
async def create_mix(request: Request):
    async with request.form(max_files=6, max_fields=8) as form:
        mode = form.get("mode", "automix")
        if mode not in ("automix", "plain"):
            raise HTTPException(400, "Mode must be automix or plain.")
        tracks = []
        if form.get("source_mix_id"):
            source_folder = result_path(str(form["source_mix_id"])).parent
            source_file = source_folder / "sources.json"
            if not source_file.is_file():
                raise HTTPException(409, "Reselect the original tracks to rebuild this session.")
            tracks = json.loads(source_file.read_text())
            if any(not Path(track["path"]).is_file() for track in tracks):
                raise HTTPException(409, "An original track is missing. Please reselect your files.")
        else:
            uploads = form.getlist("files")
            if not 2 <= len(uploads) <= 6:
                raise HTTPException(400, "Choose two to six audio files.")
            for upload in uploads:
                if not hasattr(upload, "filename"):
                    raise HTTPException(400, "Expected audio files.")
                name = upload.filename or ""
                if "/" in name or "\\" in name or not name or len(name) > 180:
                    raise HTTPException(400, "Use a simple file name without folder paths.")
                suffix = Path(name).suffix.lower()
                if suffix not in (".mp3", ".wav", ".flac"):
                    raise HTTPException(400, "Choose MP3, WAV, or FLAC files.")
                temp = DATA / "uploads" / f"{uuid.uuid4().hex}.part"
                digest = hashlib.sha256()
                size = 0
                try:
                    with temp.open("wb") as handle:
                        while chunk := await upload.read(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_FILE:
                                raise HTTPException(413, "Each file must be 100 MB or smaller.")
                            digest.update(chunk)
                            handle.write(chunk)
                    await asyncio.to_thread(validate_audio, temp)
                    ident = digest.hexdigest()
                    target = temp.parent / (ident + suffix)
                    temp.replace(target)
                    tracks.append(dict(id=ident, title=Path(name).stem, path=target))
                finally:
                    temp.unlink(missing_ok=True)
    return new_job(lambda job_id: make_mix(job_id, tracks, mode))


@app.get("/api/mixes/{ident}")
def get_mix(ident: str):
    return json.loads(result_path(ident).read_text())


@app.get("/api/mixes/{ident}/audio")
def get_audio(ident: str):
    path = result_path(ident).parent / "mix.wav"
    return FileResponse(path, media_type="audio/wav", filename=f"automix-{ident[:8]}.wav", content_disposition_type="inline")


@app.get("/api/mixes/{ident}/plan")
def get_plan(ident: str):
    return FileResponse(result_path(ident), media_type="application/json", filename="transition-plan.json")


@app.get("/api/latest")
def latest():
    results = []
    for path in (DATA / "mixes").glob("*/result.json"):
        try:
            results.append(json.loads(path.read_text()))
        except ValueError:
            pass
    results.sort(key=lambda r: r.get("created", 0), reverse=True)
    key = results[0].get("playlist_key") if results else None
    return {mode: next((r for r in results if r.get("mode") == mode and r.get("playlist_key") == key), None)
            for mode in ("automix", "plain")}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="assets")
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

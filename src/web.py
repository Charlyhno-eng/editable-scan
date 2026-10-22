# SPDX-License-Identifier: MPL-2.0
"""Local HTTP API and static React application for Editable Scan."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.convert import DocumentError, _engine, convert_pdf, inspect_document
from src.editor import EditError, EditorDraft, export_pdf, validate_draft

log = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
JOB_TTL = 3600
MAX_PENDING_JOBS = 8


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"
    done: int = 0
    total: int = 0
    error: str | None = None
    finished_at: float | None = None
    report: dict | None = None


class JobStore:
    """Bounded, single-worker queue. Documents live only in a temporary folder."""

    def __init__(self, root: Path, converter):
        self.root = root
        self.converter = converter
        self.jobs: dict[str, Job] = {}
        self.drafts: dict[str, EditorDraft] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indra")

    def cleanup(self):
        with self.lock:
            expired = [
                key
                for key, job in self.jobs.items()
                if job.finished_at and time.time() - job.finished_at > JOB_TTL
            ]
            for key in expired:
                shutil.rmtree(self.root / key, ignore_errors=True)
                del self.jobs[key]
                self.drafts.pop(key, None)

    def reserve(self, filename: str) -> Job:
        with self.lock:
            if (
                sum(
                    job.status in ("uploading", "queued", "processing")
                    for job in self.jobs.values()
                )
                >= MAX_PENDING_JOBS
            ):
                raise HTTPException(429, "The queue is full. Try again shortly.")
            job = Job(uuid4().hex, filename, status="uploading")
            self.jobs[job.id] = job
            (self.root / job.id).mkdir()
            return job

    def snapshot(self, key: str):
        with self.lock:
            if key not in self.jobs:
                raise HTTPException(404, "This file has expired or could not be found.")
            return {
                **asdict(self.jobs[key]),
                "edited": bool(self.drafts.get(key, EditorDraft()).edits),
            }

    def execute(self, key: str):
        folder = self.root / key
        with self.lock:
            self.jobs[key].status = "processing"

        def progress(done, total):
            with self.lock:
                self.jobs[key].done = done
                self.jobs[key].total = total

        try:
            report = {}
            options = {"report": report} if self.converter is convert_pdf else {}
            self.converter(folder / "input", folder / "output.pdf", progress, **options)
            with self.lock:
                self.jobs[key].report = report or None
                self.jobs[key].status = "done"
        except Exception as exc:
            from indra_ocr import ModelError

            log.exception("Conversion failed for job %s", key)
            with self.lock:
                self.jobs[key].status = "error"
                if isinstance(exc, DocumentError):
                    self.jobs[key].error = str(exc)
                elif isinstance(exc, ModelError):
                    self.jobs[
                        key
                    ].error = "The Indra models are unavailable. Check the server connection and try again."
                else:
                    self.jobs[key].error = "Conversion failed. Try another document."
            (folder / "output.pdf").unlink(missing_ok=True)
        finally:
            (folder / "input").unlink(missing_ok=True)
            with self.lock:
                self.jobs[key].finished_at = time.time()

    def warmup(self):
        """Load on the conversion worker so its thread-local models can be reused."""
        try:
            _engine()
        except Exception:
            log.warning(
                "OCR warm-up failed; the next conversion will retry", exc_info=True
            )


class RequestSizeLimit:
    """Bound uploads before the multipart parser writes them to its spool."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT"):
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD_BYTES + 1024 * 1024
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > limit:
                raise HTTPException(413, "The file exceeds 50 MB.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(*, converter=convert_pdf, static_dir: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        with TemporaryDirectory(prefix="editable-scan-") as temporary:
            store = JobStore(Path(temporary), converter)
            app.state.store = store
            if converter is convert_pdf:
                store.pool.submit(store.warmup)

            async def reap():
                while True:
                    await asyncio.sleep(60)
                    store.cleanup()

            task = asyncio.create_task(reap())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await run_in_threadpool(
                    store.pool.shutdown, wait=True, cancel_futures=True
                )

    app = FastAPI(title="Editable Scan", lifespan=lifespan)
    app.add_middleware(RequestSizeLimit)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "engine": "indra"}

    @app.post("/api/jobs", status_code=202)
    async def upload(file: UploadFile):
        store = app.state.store
        store.cleanup()
        filename = (file.filename or "document").replace("\\", "/").split("/")[-1][:180]
        job = store.reserve(filename)
        folder = store.root / job.id
        size = 0
        try:
            with (folder / "input").open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "The file exceeds 50 MB.")
                    stream.write(chunk)
            if not size:
                raise HTTPException(400, "The file is empty.")
            _, count = await run_in_threadpool(inspect_document, folder / "input")
            with store.lock:
                job.total = count
                job.status = "queued"
            store.pool.submit(store.execute, job.id)
            return store.snapshot(job.id)
        except BaseException as exc:
            with store.lock:
                store.jobs.pop(job.id, None)
            shutil.rmtree(folder, ignore_errors=True)
            if isinstance(exc, DocumentError):
                raise HTTPException(422, str(exc)) from exc
            raise
        finally:
            await file.close()

    @app.get("/api/jobs/{key}")
    def status(key: str):
        return app.state.store.snapshot(key)

    @app.get("/api/jobs/{key}/download")
    def download(key: str, original: bool = False):
        store = app.state.store
        job = store.snapshot(key)
        if job["status"] != "done":
            raise HTTPException(409, "The PDF is still being prepared.")
        stem = (
            re.sub(r"[\x00-\x1f\x7f]", "", Path(job["filename"]).stem).strip()
            or "document"
        )
        response = FileResponse(
            store.root / key / "output.pdf",
            media_type="application/pdf",
            filename=f"{stem}-text.pdf",
            headers={"Cache-Control": "no-store"},
        )
        with store.lock:
            draft = store.drafts.get(key)
        if not original and draft and draft.edits:
            return Response(
                export_pdf(editable_path(key).read_bytes(), draft),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": response.headers["content-disposition"],
                    "Cache-Control": "no-store",
                },
            )
        return response

    def editable_path(key):
        store = app.state.store
        with store.lock:
            job = store.snapshot(key)
            if job["status"] != "done":
                raise HTTPException(409, "The PDF is still being prepared.")
            # Editing renews the existing temporary-document lifetime.
            store.jobs[key].finished_at = time.time()
            return store.root / key / "output.pdf"

    @app.get("/api/jobs/{key}/editor")
    def editor(key: str, response: Response):
        editable_path(key)
        response.headers["Cache-Control"] = "no-store"
        with app.state.store.lock:
            return app.state.store.drafts.get(key, EditorDraft())

    @app.get("/api/jobs/{key}/passages")
    def passages(key: str, page: int = 1):
        from src.original import original_regions

        try:
            return original_regions(editable_path(key), page)
        except EditError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.put("/api/jobs/{key}/editor")
    def save_editor(key: str, draft: EditorDraft):
        source = editable_path(key)
        try:
            validate_draft(source, draft)
        except EditError as exc:
            raise HTTPException(422, str(exc)) from exc
        with app.state.store.lock:
            app.state.store.drafts[key] = draft
        return {"saved": True}

    @app.post("/api/jobs/{key}/export")
    def export(key: str, draft: EditorDraft):
        source = editable_path(key).read_bytes()
        try:
            data = export_pdf(source, draft)
        except EditError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(
            data, media_type="application/pdf", headers={"Cache-Control": "no-store"}
        )

    # Keep unknown API routes out of the SPA fallback.
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT"])
    def missing_api(path: str):
        raise HTTPException(404, "Route not found.")

    directory = static_dir or Path(
        os.environ.get(
            "EDITABLE_SCAN_WEB_DIR",
            Path(__file__).resolve().parents[1] / "web" / "dist",
        )
    )
    if (directory / "index.html").is_file():
        if (directory / "assets").is_dir():
            app.mount(
                "/assets", StaticFiles(directory=directory / "assets"), name="assets"
            )
        if (directory / "pdfjs").is_dir():
            app.mount(
                "/pdfjs", StaticFiles(directory=directory / "pdfjs"), name="pdfjs"
            )

        @app.get("/{path:path}")
        def frontend(path: str):
            return FileResponse(
                directory / "index.html", headers={"Cache-Control": "no-cache"}
            )

    return app


def run():
    """Serve the built React app and API on one local origin."""
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("EDITABLE_SCAN_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    run()

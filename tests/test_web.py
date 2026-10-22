# SPDX-License-Identifier: MPL-2.0
"""Upload, queue, download, failure and expiry contracts."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pikepdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from editable_scan import convert as conversion
from editable_scan import web
from editable_scan.convert import DocumentError


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(stream, "PNG")
    return stream.getvalue()


def convert(source, output, progress):
    progress(0, 1)
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(output)
    progress(1, 1)


def await_job(client, key):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{key}").json()
        if result["status"] in ("done", "error"):
            return result
        time.sleep(0.01)
    pytest.fail("Job did not finish")


def test_upload_status_download_and_cleanup():
    app = web.create_app(converter=convert)
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok", "engine": "indra"}
        response = client.post(
            "/api/jobs", files={"file": ("café.png", image_bytes(), "image/png")}
        )
        assert response.status_code == 202
        key = response.json()["id"]
        job = await_job(client, key)
        assert job["status"] == "done"
        assert (job["done"], job["total"]) == (1, 1)
        download = client.get(f"/api/jobs/{key}/download")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/pdf"
        assert "text.pdf" in download.headers["content-disposition"]
        assert download.content.startswith(b"%PDF-")
        assert not (app.state.store.root / key / "input").exists()
        root = app.state.store.root
    assert not root.exists()


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"", 400), (b"not an image", 422), (b"%PDF-1.7\nbroken", 422)],
)
def test_invalid_uploads_are_cleaned(data, expected):
    app = web.create_app(converter=convert)
    with TestClient(app) as client:
        response = client.post("/api/jobs", files={"file": ("file.pdf", data)})
        assert response.status_code == expected
        assert not app.state.store.jobs
        assert list(app.state.store.root.iterdir()) == []


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 10)
    with TestClient(web.create_app(converter=convert)) as client:
        assert (
            client.post(
                "/api/jobs", files={"file": ("large.png", image_bytes())}
            ).status_code
            == 413
        )


def test_streaming_request_size_limit(monkeypatch):
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 10)
    with TestClient(web.create_app(converter=convert)) as client:
        response = client.post(
            "/api/jobs", files={"file": ("huge.png", b"x" * (1024 * 1024 + 100))}
        )
        assert response.status_code == 413


def test_failure_never_offers_a_pdf():
    def fail(source, output, progress):
        output.write_bytes(b"partial output")
        raise DocumentError("No text detected.")

    app = web.create_app(converter=fail)
    with TestClient(app) as client:
        key = client.post(
            "/api/jobs", files={"file": ("scan.png", image_bytes())}
        ).json()["id"]
        result = await_job(client, key)
        assert result["status"] == "error"
        assert result["error"] == "No text detected."
        assert client.get(f"/api/jobs/{key}/download").status_code == 409
        assert not (app.state.store.root / key / "output.pdf").exists()


def test_queue_is_bounded_and_download_waits(monkeypatch):
    monkeypatch.setattr(web, "MAX_PENDING_JOBS", 1)
    release = threading.Event()

    def wait(source, output, progress):
        assert release.wait(5)
        convert(source, output, progress)

    app = web.create_app(converter=wait)
    with TestClient(app) as client:
        try:
            first = client.post(
                "/api/jobs", files={"file": ("scan.png", image_bytes())}
            )
            key = first.json()["id"]
            assert client.get(f"/api/jobs/{key}/download").status_code == 409
            assert (
                client.post(
                    "/api/jobs", files={"file": ("scan.png", image_bytes())}
                ).status_code
                == 429
            )
        finally:
            release.set()
        assert await_job(client, key)["status"] == "done"


def test_expiry_removes_results():
    app = web.create_app(converter=convert)
    with TestClient(app) as client:
        key = client.post(
            "/api/jobs", files={"file": ("scan.png", image_bytes())}
        ).json()["id"]
        await_job(client, key)
        with app.state.store.lock:
            app.state.store.jobs[key].finished_at = time.time() - web.JOB_TTL - 1
        app.state.store.cleanup()
        assert client.get(f"/api/jobs/{key}").status_code == 404
        assert not (app.state.store.root / key).exists()


def test_spa_deep_links_and_api_404(tmp_path):
    (tmp_path / "index.html").write_text("<html>Editable Scan</html>")
    with TestClient(web.create_app(converter=convert, static_dir=tmp_path)) as client:
        assert "Editable Scan" in client.get("/document/example").text
        assert client.get("/api/unknown").status_code == 404
        assert client.get("/api/jobs/unknown").status_code == 404


def test_models_warm_up_on_the_same_worker_as_conversions(monkeypatch):
    loaded_on = []
    monkeypatch.setattr(web, "_engine", lambda: loaded_on.append(threading.get_ident()))
    with TestClient(web.create_app()) as client:
        assert client.get("/api/health").status_code == 200
        worker = client.app.state.store.pool.submit(threading.get_ident).result(
            timeout=3
        )
        assert loaded_on == [worker]


def test_completed_job_keeps_visual_report_after_source_cleanup(monkeypatch):
    monkeypatch.setattr(web, "_engine", lambda: None)
    monkeypatch.setattr(
        conversion,
        "_engine",
        lambda: lambda _: SimpleNamespace(source="ocr", rotation=0, lines=[]),
    )
    fixture = Path(__file__).parent / "fixtures" / "scan.png"
    app = web.create_app()
    with TestClient(app) as client:
        key = client.post(
            "/api/jobs", files={"file": ("scan.png", fixture.read_bytes())}
        ).json()["id"]
        job = await_job(client, key)
        assert job["status"] == "done"
        report = job["report"]
        assert 0 <= report["similarity_percent"] <= 100
        assert len(report["pages"]) == 1
        download = client.get(f"/api/jobs/{key}/download")
        assert report["output_bytes"] == len(download.content)
        assert not (app.state.store.root / key / "input").exists()
        assert client.get(f"/api/jobs/{key}").json()["report"] == report

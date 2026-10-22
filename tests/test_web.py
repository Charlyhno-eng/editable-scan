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

from src import convert as conversion
from src import web
from src.convert import DocumentError


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


def test_editor_draft_export_download_reset_and_expiry():
    import pymupdf

    from src.editor import EditorDraft, OriginalEdit
    from tests.test_original import source

    def extracted(data):
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return doc[0].get_text()

    def editable_convert(input, output, progress):
        output.write_bytes(source())
        progress(1, 1)

    app = web.create_app(converter=editable_convert)
    with TestClient(app) as client:
        key = client.post(
            "/api/jobs", files={"file": ("scan.png", image_bytes())}
        ).json()["id"]
        await_job(client, key)
        base = f"/api/jobs/{key}"
        original = client.get(f"{base}/download").content
        assert client.get(f"{base}/editor").json() == {"edits": []}
        passage = client.get(f"{base}/passages").json()[0]
        assert passage["text"] == "Original words"
        assert client.get(f"{base}/passages?page=50").status_code == 422
        draft = EditorDraft(
            edits=[
                OriginalEdit(id=passage["id"], page=1, text="Edited words", bold=True)
            ]
        ).model_dump()
        assert client.put(f"{base}/editor", json=draft).status_code == 200
        assert client.get(f"{base}/editor").json() == draft
        # Export is a read operation on the draft, including during a newer save.
        assert client.post(f"{base}/export", json={"edits": []}).status_code == 200
        assert client.get(f"{base}/editor").json() == draft
        assert client.get(base).json()["edited"] is True
        output = client.post(f"{base}/export", json=draft)
        assert output.status_code == 200
        assert output.headers["content-type"] == "application/pdf"
        assert extracted(output.content).count("Edited words") == 1
        assert (
            extracted(client.get(f"{base}/download").content).count("Edited words") == 1
        )
        assert client.get(f"{base}/download?original=true").content == original
        invalid = {"edits": [{**draft["edits"][0], "page": 50}]}
        assert client.put(f"{base}/editor", json=invalid).status_code == 422
        assert client.post(f"{base}/export", json=invalid).status_code == 422
        assert client.get(f"{base}/editor").json() == draft
        assert client.put(f"{base}/editor", json={"edits": []}).status_code == 200
        assert client.get(f"{base}/download").content == original
        assert client.get(base).json()["edited"] is False
        app.state.store.jobs[key].finished_at = time.time() - web.JOB_TTL - 1
        app.state.store.cleanup()
        assert key not in app.state.store.drafts
        assert client.get(f"{base}/editor").status_code == 404
        assert client.post(f"{base}/export", json=draft).status_code == 404


def test_editor_reuses_render_for_preview_save_and_download(monkeypatch):
    from tests.test_original import source

    calls = []
    render = web.export_pdf

    def counted(data, draft):
        calls.append(draft.model_dump_json())
        return render(data, draft)

    monkeypatch.setattr(web, "export_pdf", counted)

    def convert(input, output, progress):
        output.write_bytes(source())
        progress(1, 1)

    app = web.create_app(converter=convert)
    with TestClient(app) as client:
        key = client.post(
            "/api/jobs", files={"file": ("scan.png", image_bytes())}
        ).json()["id"]
        await_job(client, key)
        base = f"/api/jobs/{key}"
        passage = client.get(f"{base}/passages").json()[0]
        info = client.get(f"{base}/font", params={"passage": passage["id"], "page": 1})
        assert info.status_code == 200
        assert info.json()["name"] == "Helvetica"
        assert info.json()["exact"] is True
        assert (
            client.get(f"{base}/font", params={"passage": "missing"}).status_code == 422
        )
        assert client.get(f"{base}/passages?unit=unknown").status_code == 422
        draft = {
            "edits": [
                {
                    "id": passage["id"],
                    "page": 1,
                    "text": "A longer replacement",
                    "overflow": "shrink",
                    "font": "LiberationSerif",
                }
            ]
        }
        preview = client.post(f"{base}/export", json=draft)
        assert preview.status_code == 200
        assert client.put(f"{base}/editor", json=draft).status_code == 200
        assert client.get(f"{base}/download").content == preview.content
        assert client.post(f"{base}/export", json=draft).content == preview.content
        assert len(calls) == 1
        saved = client.get(f"{base}/editor").json()["edits"][0]
        assert saved["overflow"] == "shrink"
        assert saved["font"] == "LiberationSerif"
        assert app.state.store.renders
        app.state.store.jobs[key].finished_at = time.time() - web.JOB_TTL - 1
        app.state.store.cleanup()
        assert not app.state.store.renders


def test_concurrent_editor_requests_share_one_render(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from src.editor import EditorDraft
    from tests.test_original import draft, source

    data = source()
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "output.pdf").write_bytes(data)
    store = web.JobStore(tmp_path, None)
    calls = []
    original = web.export_pdf

    def counted(data, edits):
        calls.append(1)
        return original(data, edits)

    monkeypatch.setattr(web, "export_pdf", counted)
    changes = draft(data, text="Concurrent changes")
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: store.render("job", changes), range(4)))
    assert len(calls) == 1
    assert len(set(outputs)) == 1
    for i in range(10):
        store.render("job", draft(data, text=f"Replacement {i}"))
    assert len(store.renders) == 8
    store.render("job", EditorDraft())
    assert len(store.renders) == 8
    store.pool.shutdown()

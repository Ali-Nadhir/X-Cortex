import io

from fastapi.testclient import TestClient

from xcortex.server.app import create_app


def client(settings):
    return TestClient(create_app(settings))


def png_bytes():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (64, 64), 128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_config(settings):
    api = client(settings)
    assert api.get("/api/health").json()["status"] == "ok"
    cfg = api.get("/api/config").json()
    assert cfg["mode"] == "mock"
    assert {t["id"] for t in cfg["tools"]} == {"triage", "segmentation", "rag"}


def test_index_served(settings):
    api = client(settings)
    response = api.get("/")
    assert response.status_code == 200
    assert "X-Cortex" in response.text
    assert api.get("/static/app.js").status_code == 200


def test_upload_rejects_unknown_type(settings):
    api = client(settings)
    response = api.post("/api/upload", files={"file": ("x.txt", b"nope", "text/plain")})
    assert response.status_code == 400


def test_full_analysis_stream(settings):
    api = client(settings)
    upload = api.post("/api/upload", files={"file": ("study.png", png_bytes(), "image/png")})
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]

    response = api.post("/api/analyze", json={"session_id": session_id, "history": "72yo F"})
    assert response.status_code == 200
    body = response.text
    assert "event: stage" in body
    assert "event: token" in body
    assert "event: report" in body
    assert "event: done" in body

    detail = api.get(f"/api/sessions/{session_id}").json()
    assert detail["report"]["triage_summary"]["band"] == "high"
    assert detail["verification"]["ok"] is True
    listing = api.get("/api/sessions").json()["sessions"]
    assert listing[0]["region"] == "wrist"

    preview = api.get(f"/a/{session_id}/preview.png")
    assert preview.status_code == 200
    assert api.get("/a/../etc/passwd").status_code == 404


def test_followup_stream(settings):
    api = client(settings)
    upload = api.post("/api/upload", files={"file": ("study.png", png_bytes(), "image/png")})
    session_id = upload.json()["session_id"]
    api.post("/api/analyze", json={"session_id": session_id, "history": ""})

    response = api.post("/api/ask", json={"session_id": session_id, "question": "summarize"})
    assert response.status_code == 200
    assert "event: token" in response.text

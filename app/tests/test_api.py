import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(
        settings,
        "upload_dir",
        str(tmp_path / "uploads"),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_requires_api_key(client):
    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "sample.mov",
                b"fake-video",
                "video/quicktime",
            )
        },
    )

    assert response.status_code == 401


def test_rejects_non_mov_file(client):
    response = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": "test-key"},
        files={
            "file": (
                "sample.mp4",
                b"fake-video",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 415


def test_rejects_empty_file(client):
    response = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": "test-key"},
        files={
            "file": (
                "empty.mov",
                b"",
                "video/quicktime",
            )
        },
    )

    assert response.status_code == 400


def test_upload_and_job_status(client):
    response = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": "test-key"},
        files={
            "file": (
                "sample.mov",
                b"fake-video-content",
                "video/quicktime",
            )
        },
    )

    assert response.status_code == 202

    body = response.json()
    assert body["job_id"]
    assert body["status"] == "processing"

    job_response = client.get(
        f"/api/v1/jobs/{body['job_id']}",
        headers={"X-API-Key": "test-key"},
    )

    assert job_response.status_code == 200

    result = job_response.json()
    assert result["status"] == "failed"
    assert result["error"] == "analyzer_not_configured"


def test_unknown_job_returns_404(client):
    response = client.get(
        "/api/v1/jobs/nonexistent-job",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 404
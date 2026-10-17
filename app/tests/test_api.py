import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.job_store import job_store
from app.workers.processor import process_video


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-key")
    monkeypatch.setattr(
        settings,
        "upload_dir",
        str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(
        settings,
        "database_path",
        str(tmp_path / "chiaki.db"),
    )
    monkeypatch.setattr(settings, "job_lease_seconds", 360)
    monkeypatch.setattr(settings, "max_job_attempts", 2)

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


def test_upload_creates_queued_job(client):
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
    assert body["status"] == "queued"

    job_response = client.get(
        f"/api/v1/jobs/{body['job_id']}",
        headers={"X-API-Key": "test-key"},
    )

    assert job_response.status_code == 200
    assert job_response.json()["status"] == "queued"


def test_worker_marks_unconfigured_analyzer_as_failed(
    client,
    tmp_path,
):
    video_path = tmp_path / "sample.mov"
    video_path.write_bytes(b"fake-video")

    job_id = str(uuid4())
    job_store.create(job_id, video_path)

    claimed = job_store.claim_next()

    assert claimed is not None

    claimed_id, claimed_path, token = claimed
    assert claimed_id == job_id

    process_video(claimed_id, claimed_path, token)

    result = job_store.get(job_id)

    assert result is not None
    assert result.status == "failed"
    assert result.error == "analyzer_not_configured"
    assert not video_path.exists()


def test_job_cannot_be_claimed_twice(client, tmp_path):
    video_path = tmp_path / "sample.mov"
    video_path.write_bytes(b"fake-video")

    job_store.create(str(uuid4()), video_path)

    first = job_store.claim_next()
    second = job_store.claim_next()

    assert first is not None
    assert second is None


def test_stale_claim_cannot_overwrite_new_claim(
    client,
    tmp_path,
):
    video_path = tmp_path / "sample.mov"
    video_path.write_bytes(b"fake-video")

    job_id = str(uuid4())
    job_store.create(job_id, video_path)

    first = job_store.claim_next()
    assert first is not None

    _, _, old_token = first

    # Simulate an expired lease.
    with job_store._connect() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET lease_until = ?
            WHERE job_id = ?
            """,
            (time.time() - 1, job_id),
        )

    second = job_store.claim_next()
    assert second is not None

    _, _, new_token = second
    assert old_token != new_token

    assert job_store.fail(
        job_id,
        old_token,
        "stale_worker_error",
    ) is False

    result = job_store.get(job_id)
    assert result is not None
    assert result.status == "processing"
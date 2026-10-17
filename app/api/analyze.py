from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from app.config import settings
from app.schemas.analysis import AnalysisResult
from app.services.job_store import job_store

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

ALLOWED_EXTENSION = ".mov"


def verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_video(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
) -> AnalysisResult:
    filename = file.filename or ""

    if Path(filename).suffix.lower() != ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=415,
            detail="Only .mov files are supported",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / f"{uuid4()}.mov"
    max_size = settings.max_upload_size_mb * 1024 * 1024
    size = 0

    try:
        with video_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds maximum upload size",
                    )

                destination.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty",
            )

    except Exception:
        video_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    job = job_store.create(
        job_id=str(uuid4()),
        video_path=video_path,
    )

    return job


@router.get(
    "/jobs/{job_id}",
    response_model=AnalysisResult,
)
def get_job(
    job_id: str,
    _: None = Depends(verify_api_key),
) -> AnalysisResult:
    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job
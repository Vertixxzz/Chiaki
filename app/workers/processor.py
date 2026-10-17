import logging
from pathlib import Path

from app.schemas.analysis import AnalysisResult
from app.services.analyzer import (
    Analyzer,
    AnalyzerNotConfiguredError,
)
from app.services.job_manager import job_manager

logger = logging.getLogger(__name__)

analyzer = Analyzer()


def process_video(job_id: str, video_path: Path) -> None:
    try:
        result = analyzer.analyze(video_path)

        job_manager.update(
            job_id,
            AnalysisResult(
                job_id=job_id,
                status="done",
                **result,
            ),
        )

    except AnalyzerNotConfiguredError as exc:
        job_manager.update(
            job_id,
            AnalysisResult(
                job_id=job_id,
                status="failed",
                error=str(exc),
            ),
        )

    except Exception:
        logger.exception("Analysis failed for job %s", job_id)

        job_manager.update(
            job_id,
            AnalysisResult(
                job_id=job_id,
                status="failed",
                error="analysis_failed",
            ),
        )

    finally:
        # Удаляем временное видео после завершения обработки.
        video_path.unlink(missing_ok=True)
import logging
from pathlib import Path

from app.schemas.analysis import AnalysisResult
from app.services.analyzer import (
    Analyzer,
    AnalyzerNotConfiguredError,
)
from app.services.job_store import job_store

logger = logging.getLogger(__name__)
analyzer = Analyzer()


def process_video(
    job_id: str,
    video_path: Path,
    claim_token: str,
) -> None:
    owns_job = False

    try:
        payload = analyzer.analyze(video_path)

        result = AnalysisResult(
            job_id=job_id,
            status="done",
            **payload,
        )

        if result.verdict is None:
            raise ValueError("Analyzer returned no verdict")

        if result.verdict == "no_violation" and result.violations:
            raise ValueError(
                "no_violation verdict cannot contain violations"
            )

        if result.verdict == "violation" and not result.violations:
            raise ValueError(
                "violation verdict requires at least one event"
            )

        owns_job = job_store.complete(result, claim_token)

        if not owns_job:
            logger.warning(
                "Ignoring stale result for job %s",
                job_id,
            )

    except AnalyzerNotConfiguredError as exc:
        owns_job = job_store.fail(
            job_id,
            claim_token,
            str(exc),
        )

    except Exception:
        logger.exception("Analysis failed for job %s", job_id)

        owns_job = job_store.fail(
            job_id,
            claim_token,
            "analysis_failed",
        )

    finally:
        if owns_job:
            video_path.unlink(missing_ok=True)
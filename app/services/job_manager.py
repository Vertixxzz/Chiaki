from threading import Lock
from uuid import uuid4

from app.schemas.analysis import AnalysisResult


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, AnalysisResult] = {}
        self._lock = Lock()

    def create(self) -> AnalysisResult:
        job = AnalysisResult(
            job_id=str(uuid4()),
            status="processing",
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> AnalysisResult | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, result: AnalysisResult) -> None:
        with self._lock:
            self._jobs[job_id] = result


job_manager = JobManager()
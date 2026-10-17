import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.config import settings
from app.schemas.analysis import AnalysisResult, Violation


class JobStore:
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db_path = Path(settings.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            db_path,
            timeout=10,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row

        try:
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(connection)
            yield connection

            if connection.in_transaction:
                connection.commit()

        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                verdict TEXT,
                violations_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                video_path TEXT NOT NULL,
                claimed INTEGER NOT NULL DEFAULT 0,
                claimed_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                claim_token TEXT,
                lease_until REAL,
                started_at REAL
            )
            """
        )

        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }

        migrations = {
            "attempts": (
                "ALTER TABLE jobs ADD COLUMN "
                "attempts INTEGER NOT NULL DEFAULT 0"
            ),
            "claim_token": "ALTER TABLE jobs ADD COLUMN claim_token TEXT",
            "lease_until": "ALTER TABLE jobs ADD COLUMN lease_until REAL",
            "started_at": "ALTER TABLE jobs ADD COLUMN started_at REAL",
        }

        for column, statement in migrations.items():
            if column not in existing_columns:
                connection.execute(statement)

        # Recover jobs left in processing by the older implementation,
        # which did not use claim tokens or leases.
        connection.execute(
            """
            UPDATE jobs
            SET status = 'queued',
                claimed = 0,
                claim_token = NULL,
                lease_until = NULL,
                started_at = NULL
            WHERE status = 'processing'
              AND claim_token IS NULL
              AND lease_until IS NULL
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
            ON jobs(status, created_at)
            """
        )

    def create(self, job_id: str, video_path: Path) -> AnalysisResult:
        now = time.time()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, video_path, created_at, updated_at
                )
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (job_id, str(video_path), now, now),
            )

        return AnalysisResult(job_id=job_id, status="queued")

    def get(self, job_id: str) -> AnalysisResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

        if row is None:
            return None

        return AnalysisResult(
            job_id=row["job_id"],
            status=row["status"],
            verdict=row["verdict"],
            violations=[
                Violation.model_validate(item)
                for item in json.loads(row["violations_json"])
            ],
            error=row["error"],
        )

    def claim_next(self) -> tuple[str, Path, str] | None:
        now = time.time()
        lease_seconds = settings.job_lease_seconds
        max_attempts = settings.max_job_attempts

        with self._connect() as connection:
            # Serialize selection and state changes across concurrent workers.
            connection.execute("BEGIN IMMEDIATE")

            expired = connection.execute(
                """
                SELECT job_id, attempts
                FROM jobs
                WHERE status = 'processing'
                  AND (lease_until IS NULL OR lease_until < ?)
                """,
                (now,),
            ).fetchall()

            for expired_row in expired:
                attempts = expired_row["attempts"]
                if attempts >= max_attempts:
                    connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed',
                            error = 'worker_lease_expired',
                            claimed = 0,
                            claimed_at = NULL,
                            claim_token = NULL,
                            lease_until = NULL,
                            started_at = NULL,
                            updated_at = ?
                        WHERE job_id = ?
                          AND status = 'processing'
                        """,
                        (now, expired_row["job_id"]),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'queued',
                            error = NULL,
                            claimed = 0,
                            claimed_at = NULL,
                            claim_token = NULL,
                            lease_until = NULL,
                            started_at = NULL,
                            updated_at = ?
                        WHERE job_id = ?
                          AND status = 'processing'
                        """,
                        (now, expired_row["job_id"]),
                    )

            row = connection.execute(
                """
                SELECT job_id, video_path
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                return None

            job_id = row["job_id"]
            video_path = Path(row["video_path"])
            claim_token = str(uuid4())

            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    attempts = attempts + 1,
                    claimed = 1,
                    claimed_at = ?,
                    claim_token = ?,
                    lease_until = ?,
                    started_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'queued'
                """,
                (
                    now,
                    claim_token,
                    now + lease_seconds,
                    now,
                    now,
                    job_id,
                ),
            )

            if updated.rowcount != 1:
                return None

            return job_id, video_path, claim_token

    def complete(self, result: AnalysisResult, claim_token: str) -> bool:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'done',
                    verdict = ?,
                    violations_json = ?,
                    error = NULL,
                    claimed = 0,
                    claimed_at = NULL,
                    claim_token = NULL,
                    lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'processing'
                  AND claim_token = ?
                """,
                (
                    result.verdict,
                    json.dumps(
                        [
                            item.model_dump(mode="json")
                            for item in result.violations
                        ],
                        ensure_ascii=False,
                    ),
                    time.time(),
                    result.job_id,
                    claim_token,
                ),
            )

        return updated.rowcount == 1

    def fail(self, job_id: str, claim_token: str, error: str) -> bool:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = ?,
                    claimed = 0,
                    claimed_at = NULL,
                    claim_token = NULL,
                    lease_until = NULL,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'processing'
                  AND claim_token = ?
                """,
                (error, time.time(), job_id, claim_token),
            )

        return updated.rowcount == 1


job_store = JobStore()

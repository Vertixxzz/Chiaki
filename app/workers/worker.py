import logging
import time

from app.services.job_store import job_store
from app.workers.processor import process_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Chiaki worker started")

    while True:
        try:
            task = job_store.claim_next()

            if task is None:
                time.sleep(1)
                continue

            job_id, video_path, claim_token = task
            logger.info("Claimed job %s", job_id)

            process_video(
                job_id,
                video_path,
                claim_token,
            )

        except KeyboardInterrupt:
            logger.info("Chiaki worker stopped")
            break

        except Exception:
            logger.exception("Worker loop failed")
            time.sleep(2)


if __name__ == "__main__":
    main()
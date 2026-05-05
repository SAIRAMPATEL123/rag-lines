from datetime import datetime, timezone
from threading import Timer, Lock
from typing import Dict, Any
from uuid import uuid4
from loguru import logger


class InProcessScheduler:
    """Simple in-process one-off scheduler for MVP use."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def schedule_once(self, run_at: datetime, fn, payload: Dict[str, Any]) -> str:
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)

        delay = max(0, (run_at - datetime.now(timezone.utc)).total_seconds())
        job_id = str(uuid4())

        def wrapped():
            with self._lock:
                self.jobs[job_id]["status"] = "running"
            try:
                result = fn(payload)
                with self._lock:
                    self.jobs[job_id]["status"] = "completed"
                    self.jobs[job_id]["result"] = result
            except Exception as e:
                logger.exception(f"Scheduled job failed: {job_id}")
                with self._lock:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["error"] = str(e)

        timer = Timer(delay, wrapped)
        with self._lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "run_at": run_at.isoformat(),
                "status": "scheduled",
                "payload": payload,
            }
        timer.start()
        logger.info(f"Scheduled job {job_id} to run in {delay:.2f}s")
        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.jobs.get(job_id, {"job_id": job_id, "status": "not_found"})


scheduler = InProcessScheduler()

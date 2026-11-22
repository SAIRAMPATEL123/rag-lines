from datetime import datetime, timezone
from threading import Timer, Lock
from typing import Dict, Any, Optional
from uuid import uuid4
from loguru import logger
import json
from pathlib import Path
import os
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ScheduledJob(Base):
    """SQLAlchemy model for scheduled jobs"""
    __tablename__ = "scheduled_jobs"

    job_id = Column(String(36), primary_key=True)
    run_at = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    payload = Column(Text, nullable=False)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)


class PersistentScheduler:
    """Scheduler with persistent job store using MySQL/SQLite"""

    def __init__(self, db_connection_string: str = None):
        from config.config import get_config

        self.config = get_config()
        self.db_connection_string = db_connection_string or self.config.db_connection_string

        try:
            self.engine = create_engine(self.db_connection_string, pool_pre_ping=True)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            self.use_db = True
            logger.info(f"PersistentScheduler using database: {self.config.db_type}")
        except Exception as e:
            logger.warning(f"Database connection failed, falling back to JSON: {e}")
            self.use_db = False
            self.jobs_dir = Path("./data/scheduled_jobs")
            self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_timers: Dict[str, Timer] = {}
        self._lock = Lock()
        self._load_pending_jobs()
        logger.info("PersistentScheduler initialized")

    def _load_pending_jobs(self):
        """Load pending jobs from database/disk and reschedule them"""
        try:
            if self.use_db:
                session = self.Session()
                try:
                    pending = session.query(ScheduledJob).filter(
                        ScheduledJob.status == "scheduled"
                    ).all()

                    now = datetime.now(timezone.utc)
                    for job_record in pending:
                        run_at = job_record.run_at
                        if run_at.tzinfo is None:
                            run_at = run_at.replace(tzinfo=timezone.utc)

                        if run_at > now:
                            delay = (run_at - now).total_seconds()
                            job_dict = {
                                "job_id": job_record.job_id,
                                "run_at": job_record.run_at.isoformat(),
                                "status": job_record.status,
                                "payload": json.loads(job_record.payload)
                            }
                            self.jobs[job_record.job_id] = job_dict

                            def make_callback(jid):
                                def callback():
                                    self._mark_job_running(jid)
                                return callback

                            timer = Timer(delay, make_callback(job_record.job_id))
                            self.active_timers[job_record.job_id] = timer
                            timer.start()
                            logger.info(f"Rescheduled pending job {job_record.job_id} to run in {delay:.1f}s")

                    logger.info(f"Loaded {len(pending)} pending jobs from database")
                finally:
                    session.close()
            else:
                # Fallback to JSON files
                for job_file in self.jobs_dir.glob("*.json"):
                    try:
                        with open(job_file, 'r') as f:
                            job = json.load(f)

                        if job.get("status") == "scheduled":
                            run_at = datetime.fromisoformat(job["run_at"])
                            now = datetime.now(timezone.utc)

                            if run_at > now:
                                delay = (run_at - now).total_seconds()
                                self.jobs[job["job_id"]] = job

                                def make_callback(jid):
                                    def callback():
                                        self._mark_job_running(jid)
                                    return callback

                                timer = Timer(delay, make_callback(job["job_id"]))
                                self.active_timers[job["job_id"]] = timer
                                timer.start()
                                logger.info(f"Rescheduled pending job {job['job_id']} to run in {delay:.1f}s")

                    except Exception as e:
                        logger.error(f"Error loading job: {e}")

        except Exception as e:
            logger.error(f"Error loading pending jobs: {e}")

    def _save_job(self, job: Dict[str, Any]):
        """Save job to database/disk"""
        if not self.use_db:
            # Fallback to JSON
            job_file = self.jobs_dir / f"{job['job_id']}.json"
            try:
                with open(job_file, 'w') as f:
                    json.dump(job, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving job {job['job_id']}: {e}")
            return

        session = self.Session()
        try:
            existing = session.query(ScheduledJob).filter(
                ScheduledJob.job_id == job["job_id"]
            ).first()

            if existing:
                existing.status = job.get("status")
                existing.result = json.dumps(job.get("result")) if job.get("result") else None
                existing.error = job.get("error")
                existing.started_at = job.get("started_at")
                existing.completed_at = job.get("completed_at")
                existing.failed_at = job.get("failed_at")
                existing.cancelled_at = job.get("cancelled_at")
            else:
                db_job = ScheduledJob(
                    job_id=job["job_id"],
                    run_at=datetime.fromisoformat(job["run_at"]),
                    status=job.get("status"),
                    created_at=datetime.fromisoformat(job.get("created_at", datetime.now(timezone.utc).isoformat())),
                    payload=json.dumps(job.get("payload", {})),
                    result=json.dumps(job.get("result")) if job.get("result") else None,
                    error=job.get("error")
                )
                session.add(db_job)

            session.commit()
        except Exception as e:
            logger.error(f"Error saving job {job['job_id']}: {e}")
            session.rollback()
        finally:
            session.close()

    def _mark_job_running(self, job_id: str):
        """Mark job as running and execute (if callback available)"""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "running"
                self.jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
                self._save_job(self.jobs[job_id])

    def schedule_once(self, run_at: datetime, fn, payload: Dict[str, Any], job_id: Optional[str] = None) -> str:
        """Schedule a one-off job that persists across restarts"""
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)

        delay = max(0, (run_at - datetime.now(timezone.utc)).total_seconds())
        job_id = job_id or str(uuid4())

        def wrapped():
            try:
                with self._lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]["status"] = "running"
                        self.jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
                        self._save_job(self.jobs[job_id])

                result = fn(payload)

                with self._lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]["status"] = "completed"
                        self.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                        self.jobs[job_id]["result"] = result
                        self._save_job(self.jobs[job_id])

                logger.info(f"Job {job_id} completed successfully")

            except Exception as e:
                logger.exception(f"Scheduled job {job_id} failed: {e}")
                with self._lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]["status"] = "failed"
                        self.jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
                        self.jobs[job_id]["error"] = str(e)
                        self._save_job(self.jobs[job_id])

        job_data = {
            "job_id": job_id,
            "run_at": run_at.isoformat(),
            "status": "scheduled",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        with self._lock:
            self.jobs[job_id] = job_data
            self._save_job(job_data)

        if delay > 0:
            timer = Timer(delay, wrapped)
            self.active_timers[job_id] = timer
            timer.start()
            logger.info(f"Scheduled job {job_id} to run in {delay:.2f}s")
        else:
            timer = Timer(0, wrapped)
            self.active_timers[job_id] = timer
            timer.start()
            logger.info(f"Job {job_id} scheduled to run immediately")

        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job details from database/memory/disk"""
        with self._lock:
            if job_id in self.jobs:
                return self.jobs[job_id]

        if self.use_db:
            session = self.Session()
            try:
                db_job = session.query(ScheduledJob).filter(
                    ScheduledJob.job_id == job_id
                ).first()

                if db_job:
                    return {
                        "job_id": db_job.job_id,
                        "run_at": db_job.run_at.isoformat(),
                        "status": db_job.status,
                        "created_at": db_job.created_at.isoformat() if db_job.created_at else None,
                        "started_at": db_job.started_at.isoformat() if db_job.started_at else None,
                        "completed_at": db_job.completed_at.isoformat() if db_job.completed_at else None,
                        "result": json.loads(db_job.result) if db_job.result else None,
                        "error": db_job.error
                    }
            except Exception as e:
                logger.error(f"Error fetching job {job_id}: {e}")
            finally:
                session.close()
        else:
            # Fallback to JSON
            job_file = self.jobs_dir / f"{job_id}.json"
            if job_file.exists():
                try:
                    with open(job_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading job {job_id}: {e}")

        return {"job_id": job_id, "status": "not_found"}

    def list_jobs(self, status: Optional[str] = None) -> list:
        """List all jobs, optionally filtered by status"""
        with self._lock:
            jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.get("status") == status]

        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        with self._lock:
            if job_id in self.active_timers:
                self.active_timers[job_id].cancel()
                del self.active_timers[job_id]
                logger.info(f"Cancelled timer for job {job_id}")

            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "cancelled"
                self.jobs[job_id]["cancelled_at"] = datetime.now(timezone.utc).isoformat()
                self._save_job(self.jobs[job_id])
                logger.info(f"Job {job_id} marked as cancelled")
                return True

        return False


scheduler = PersistentScheduler()

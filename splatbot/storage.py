from __future__ import annotations

import sqlite3
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

from .config import ScanMode
from .models import (
    ArtifactKind,
    JobArtifact,
    JobStatus,
    MediaItem,
    MediaKind,
    ScanJob,
    UploadSession,
    utcnow,
)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  telegram_user_id INTEGER NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media_items (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  local_path TEXT NOT NULL,
  remote_key TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  telegram_user_id INTEGER NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_artifacts (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  local_path TEXT NOT NULL,
  remote_key TEXT,
  url TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(telegram_user_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(telegram_user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_job ON job_artifacts(job_id, kind);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_job_kind_unique ON job_artifacts(job_id, kind);
"""


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _dt_optional(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _session(row: sqlite3.Row) -> UploadSession:
    return UploadSession(
        id=row["id"],
        telegram_user_id=row["telegram_user_id"],
        mode=ScanMode(row["mode"]),
        status=JobStatus(row["status"]),
        created_at=_dt(row["created_at"]),
        updated_at=_dt(row["updated_at"]),
    )


def _media(row: sqlite3.Row) -> MediaItem:
    return MediaItem(
        id=row["id"],
        session_id=row["session_id"],
        kind=MediaKind(row["kind"]),
        local_path=row["local_path"],
        remote_key=row["remote_key"],
        created_at=_dt(row["created_at"]),
    )


def _job(row: sqlite3.Row) -> ScanJob:
    return ScanJob(
        id=row["id"],
        session_id=row["session_id"],
        telegram_user_id=row["telegram_user_id"],
        mode=ScanMode(row["mode"]),
        status=JobStatus(row["status"]),
        error=row["error"],
        created_at=_dt(row["created_at"]),
        updated_at=_dt(row["updated_at"]),
        runpod_pod_id=row["runpod_pod_id"] if "runpod_pod_id" in row.keys() else None,
        claimed_at=_dt_optional(row["claimed_at"] if "claimed_at" in row.keys() else None),
        heartbeat_at=_dt_optional(row["heartbeat_at"] if "heartbeat_at" in row.keys() else None),
    )


def _artifact(row: sqlite3.Row) -> JobArtifact:
    return JobArtifact(
        id=row["id"],
        job_id=row["job_id"],
        kind=ArtifactKind(row["kind"]),
        local_path=row["local_path"],
        remote_key=row["remote_key"],
        url=row["url"],
        created_at=_dt(row["created_at"]),
    )


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA)
            await db.execute("PRAGMA busy_timeout=30000")
            await self._migrate(db)
            await db.commit()

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in await cursor.fetchall()}
        for name in ("runpod_pod_id", "claimed_at", "heartbeat_at"):
            if name not in columns:
                await db.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")
        await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_session_unique ON jobs(session_id)")

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        db = await aiosqlite.connect(self.path)
        db.row_factory = sqlite3.Row
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
        finally:
            await db.close()

    async def create_session(self, telegram_user_id: int, mode: ScanMode) -> UploadSession:
        now = utcnow().isoformat()
        session_id = uuid.uuid4().hex
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO sessions (id, telegram_user_id, mode, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, telegram_user_id, mode.value, JobStatus.COLLECTING.value, now, now),
            )
            await db.commit()
        session = await self.get_session(session_id)
        assert session is not None
        return session

    async def get_active_session(self, telegram_user_id: int) -> UploadSession | None:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT * FROM sessions
                WHERE telegram_user_id = ? AND status = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (telegram_user_id, JobStatus.COLLECTING.value),
            )
            row = await cursor.fetchone()
        return _session(row) if row else None

    async def get_session(self, session_id: str) -> UploadSession | None:
        async with self._connect() as db:
            cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
        return _session(row) if row else None

    async def set_session_status(self, session_id: str, status: JobStatus) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, utcnow().isoformat(), session_id),
            )
            await db.commit()

    async def set_session_mode(self, session_id: str, mode: ScanMode) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE sessions SET mode = ?, updated_at = ? WHERE id = ?",
                (mode.value, utcnow().isoformat(), session_id),
            )
            await db.commit()

    async def add_media(
        self,
        session_id: str,
        kind: MediaKind,
        local_path: Path,
        remote_key: str | None = None,
    ) -> MediaItem:
        media_id = uuid.uuid4().hex
        now = utcnow().isoformat()
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO media_items (id, session_id, kind, local_path, remote_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (media_id, session_id, kind.value, str(local_path), remote_key, now),
            )
            await db.commit()
        items = await self.list_media(session_id)
        return next(item for item in items if item.id == media_id)

    async def list_media(self, session_id: str) -> list[MediaItem]:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT * FROM media_items WHERE session_id = ? ORDER BY created_at, id",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [_media(row) for row in rows]

    async def create_job(self, session: UploadSession) -> ScanJob:
        job_id = uuid.uuid4().hex
        now = utcnow().isoformat()
        async with self._connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (JobStatus.QUEUED.value, now, session.id, JobStatus.COLLECTING.value),
            )
            if cursor.rowcount != 1:
                await db.rollback()
                existing = await self.get_job_for_session(session.id)
                if existing is not None:
                    return existing
                raise RuntimeError("session is not collecting")
            await db.execute(
                """
                INSERT INTO jobs (id, session_id, telegram_user_id, mode, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    job_id,
                    session.id,
                    session.telegram_user_id,
                    session.mode.value,
                    JobStatus.QUEUED.value,
                    now,
                    now,
                ),
            )
            await db.commit()
        job = await self.get_job(job_id)
        assert job is not None
        return job

    async def get_job_for_session(self, session_id: str) -> ScanJob | None:
        async with self._connect() as db:
            cursor = await db.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,))
            row = await cursor.fetchone()
        return _job(row) if row else None

    async def get_job(self, job_id: str) -> ScanJob | None:
        async with self._connect() as db:
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
        return _job(row) if row else None

    async def get_latest_job_for_user(self, telegram_user_id: int) -> ScanJob | None:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE telegram_user_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (telegram_user_id,),
            )
            row = await cursor.fetchone()
        return _job(row) if row else None

    async def terminal_jobs_with_runpod_pods(self) -> list[ScanJob]:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE runpod_pod_id IS NOT NULL
                  AND status IN (?, ?, ?)
                ORDER BY updated_at
                """,
                (
                    JobStatus.DONE.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ),
            )
            rows = await cursor.fetchall()
        return [_job(row) for row in rows]

    async def next_queued_job(self) -> ScanJob | None:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            )
            row = await cursor.fetchone()
        return _job(row) if row else None

    async def claim_next_queued_job(self) -> ScanJob | None:
        now = utcnow().isoformat()
        async with self._connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            cursor = await db.execute(
                """
                UPDATE jobs
                SET status = ?, error = NULL, updated_at = ?, claimed_at = ?, heartbeat_at = ?
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE status = ?
                    ORDER BY created_at
                    LIMIT 1
                )
                RETURNING *
                """,
                (JobStatus.PREPARING.value, now, now, now, JobStatus.QUEUED.value),
            )
            row = await cursor.fetchone()
            await db.commit()
        return _job(row) if row else None

    async def set_job_runpod_pod_id(self, job_id: str, pod_id: str | None) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE jobs SET runpod_pod_id = ?, heartbeat_at = ?, updated_at = ? WHERE id = ?",
                (pod_id, utcnow().isoformat(), utcnow().isoformat(), job_id),
            )
            await db.commit()

    async def heartbeat_job(self, job_id: str) -> None:
        async with self._connect() as db:
            await db.execute(
                "UPDATE jobs SET heartbeat_at = ?, updated_at = ? WHERE id = ?",
                (utcnow().isoformat(), utcnow().isoformat(), job_id),
            )
            await db.commit()

    async def set_job_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        async with self._connect() as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ? AND status NOT IN (?, ?, ?)
                """,
                (
                    status.value,
                    error,
                    utcnow().isoformat(),
                    job_id,
                    JobStatus.DONE.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ),
            )
            await db.commit()

    async def set_job_failed_unless_terminal(self, job_id: str, error: str) -> ScanJob | None:
        async with self._connect() as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ? AND status NOT IN (?, ?, ?)
                """,
                (
                    JobStatus.FAILED.value,
                    error,
                    utcnow().isoformat(),
                    job_id,
                    JobStatus.DONE.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ),
            )
            await db.commit()
        return await self.get_job(job_id)

    async def interrupted_jobs(self, grace_seconds: int) -> list[ScanJob]:
        cutoff = (utcnow() - timedelta(seconds=grace_seconds)).isoformat()
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE status IN (?, ?, ?, ?, ?, ?)
                  AND COALESCE(heartbeat_at, updated_at) < ?
                ORDER BY updated_at
                """,
                (
                    JobStatus.PREPARING.value,
                    JobStatus.PREPROCESSING.value,
                    JobStatus.COLMAP.value,
                    JobStatus.TRAINING.value,
                    JobStatus.EXPORTING.value,
                    JobStatus.RENDERING.value,
                    cutoff,
                ),
            )
            rows = await cursor.fetchall()
        return [_job(row) for row in rows]

    async def fail_interrupted_jobs(self, error: str, grace_seconds: int = 0) -> list[ScanJob]:
        interrupted = await self.interrupted_jobs(grace_seconds)
        if not interrupted:
            return []
        job_ids = [job.id for job in interrupted]
        placeholders = ",".join("?" for _ in job_ids)
        async with self._connect() as db:
            await db.execute(
                f"""
                UPDATE jobs
                SET status = ?, error = ?, updated_at = ?, runpod_pod_id = NULL
                WHERE id IN ({placeholders})
                """,
                (JobStatus.FAILED.value, error, utcnow().isoformat(), *job_ids),
            )
            await db.commit()
        return interrupted

    async def add_artifact(
        self,
        job_id: str,
        kind: ArtifactKind,
        local_path: Path,
        remote_key: str | None = None,
        url: str | None = None,
    ) -> JobArtifact:
        artifact_id = uuid.uuid4().hex
        now = utcnow().isoformat()
        async with self._connect() as db:
            await db.execute("DELETE FROM job_artifacts WHERE job_id = ? AND kind = ?", (job_id, kind.value))
            await db.execute(
                """
                INSERT INTO job_artifacts (id, job_id, kind, local_path, remote_key, url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, job_id, kind.value, str(local_path), remote_key, url, now),
            )
            await db.commit()
        artifact = await self.get_artifact(artifact_id)
        assert artifact is not None
        return artifact

    async def get_artifact(self, artifact_id: str) -> JobArtifact | None:
        async with self._connect() as db:
            cursor = await db.execute("SELECT * FROM job_artifacts WHERE id = ?", (artifact_id,))
            row = await cursor.fetchone()
        return _artifact(row) if row else None

    async def list_artifacts(self, job_id: str) -> list[JobArtifact]:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT * FROM job_artifacts WHERE job_id = ? ORDER BY created_at, id",
                (job_id,),
            )
            rows = await cursor.fetchall()
        return [_artifact(row) for row in rows]

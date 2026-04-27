from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import timedelta

from .config import Settings
from .models import JobStatus, utcnow
from .notifications import TelegramNotifier
from .pipeline import PipelineOutputs
from .publishing import publish_job_artifacts
from .storage import Store

LOGGER = logging.getLogger(__name__)


async def set_status(job_id: str, status: JobStatus, error: str | None) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    await store.set_job_status(job_id, status, error)


async def heartbeat(job_id: str) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    await store.heartbeat_job(job_id)


async def complete(job_id: str, notify: bool) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    job = await store.get_job(job_id)
    if job is None:
        raise SystemExit(f"job not found: {job_id}")
    if job.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}:
        raise SystemExit(f"job {job_id} is already terminal: {job.status.value}")
    ply = settings.job_dir(job_id) / "export" / "cleaned_splat.ply"
    preview = settings.job_dir(job_id) / "renders" / "turntable.mp4"
    preview_output = preview if preview.exists() else None
    metrics = settings.job_dir(job_id) / "metrics.json"
    artifacts = await publish_job_artifacts(
        settings,
        store,
        job,
        PipelineOutputs(ply, preview_output, metrics if metrics.exists() else None),
    )
    await store.set_job_status(job_id, JobStatus.DONE)
    updated = await store.get_job(job_id)
    if updated is None:
        raise SystemExit(f"job disappeared before completion could be recorded: {job_id}")
    if updated.status != JobStatus.DONE:
        raise SystemExit(f"job {job_id} was not marked done: {updated.status.value}")
    if notify and settings.telegram_token:
        try:
            await TelegramNotifier(settings.telegram_token).job_done(updated, artifacts)
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to send completion notification for job %s", job_id)


async def fail(job_id: str, error: str, notify: bool) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    job = await store.set_job_failed_unless_terminal(job_id, error)
    if notify and settings.telegram_token and job and job.status == JobStatus.FAILED and job.error == error:
        try:
            await TelegramNotifier(settings.telegram_token).job_failed(job, error)
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to send failure notification for job %s", job_id)


async def cleanup() -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    cutoff = utcnow() - timedelta(days=settings.job_retention_days)
    deleted = 0
    async with store._connect() as db:
        cursor = await db.execute("SELECT id, session_id, updated_at FROM jobs WHERE updated_at < ?", (cutoff.isoformat(),))
        rows = await cursor.fetchall()
        for row in rows:
            shutil.rmtree(settings.job_dir(row["id"]), ignore_errors=True)
            shutil.rmtree(settings.public_results_dir / row["id"], ignore_errors=True)
            shutil.rmtree(settings.data_dir / "sessions" / row["session_id"], ignore_errors=True)
            await db.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))
            await db.execute("DELETE FROM sessions WHERE id = ?", (row["session_id"],))
            deleted += 1
        await db.commit()
    print(f"deleted {deleted} expired job(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Splatbot job state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser("set-status")
    set_parser.add_argument("job_id")
    set_parser.add_argument("status", choices=[status.value for status in JobStatus])
    set_parser.add_argument("--error")

    heartbeat_parser = subparsers.add_parser("heartbeat")
    heartbeat_parser.add_argument("job_id")

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("job_id")
    complete_parser.add_argument("--notify", action="store_true")

    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("job_id")
    fail_parser.add_argument("--error", required=True)
    fail_parser.add_argument("--notify", action="store_true")

    subparsers.add_parser("cleanup")

    args = parser.parse_args()
    if args.command == "set-status":
        asyncio.run(set_status(args.job_id, JobStatus(args.status), args.error))
    elif args.command == "heartbeat":
        asyncio.run(heartbeat(args.job_id))
    elif args.command == "complete":
        asyncio.run(complete(args.job_id, args.notify))
    elif args.command == "fail":
        asyncio.run(fail(args.job_id, args.error, args.notify))
    elif args.command == "cleanup":
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()

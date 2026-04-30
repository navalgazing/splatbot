from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import json
from datetime import timedelta
from pathlib import Path

from .artifact_manifest import ARTIFACT_DIR_NAME, MANIFEST_NAME, write_artifact_manifest
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
    mesh = settings.job_dir(job_id) / "export" / (settings.mesh_export_filename.strip() or "mesh.glb")
    mesh_output = mesh if mesh.exists() else None
    preview = settings.job_dir(job_id) / "renders" / "turntable.mp4"
    preview_output = preview if preview.exists() else None
    metrics = settings.job_dir(job_id) / "metrics.json"
    raw_ply = settings.job_dir(job_id) / "export" / "raw_splat.ply"
    quality_report = settings.job_dir(job_id) / "quality_report.json"
    candidate_report = settings.job_dir(job_id) / "candidate_report.json"
    manifest = write_artifact_manifest(settings.job_dir(job_id), job_id=job_id)
    artifacts = await publish_job_artifacts(
        settings,
        store,
        job,
        PipelineOutputs(
            ply,
            preview_output,
            metrics if metrics.exists() else None,
            raw_ply=raw_ply if raw_ply.exists() else None,
            mesh_path=mesh_output,
            quality_report_path=quality_report if quality_report.exists() else None,
            candidate_report_path=candidate_report if candidate_report.exists() else None,
            artifact_manifest_path=manifest if manifest.exists() else None,
        ),
    )
    await store.set_job_status(job_id, JobStatus.DONE)
    updated = await store.get_job(job_id)
    if updated is None:
        raise SystemExit(f"job disappeared before completion could be recorded: {job_id}")
    if updated.status != JobStatus.DONE:
        raise SystemExit(f"job {job_id} was not marked done: {updated.status.value}")
    if notify and settings.telegram_token_value:
        try:
            await TelegramNotifier(settings.telegram_token_value).job_done(updated, artifacts)
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to send completion notification for job %s", job_id)


async def fail(job_id: str, error: str, notify: bool) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    job = await store.set_job_failed_unless_terminal(job_id, error)
    if notify and settings.telegram_token_value and job and job.status == JobStatus.FAILED and job.error == error:
        try:
            await TelegramNotifier(settings.telegram_token_value).job_failed(
                job,
                error,
                metrics_path=settings.job_dir(job_id) / "metrics.json",
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to send failure notification for job %s", job_id)


async def cleanup() -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    now = utcnow()
    public_cutoff = now - timedelta(days=settings.job_retention_days)
    heavy_cutoff = now - timedelta(days=settings.heavy_artifact_retention_days)
    debug_cutoff = now - timedelta(days=max(settings.job_retention_days, settings.debug_artifact_retention_days))
    heavy_pruned = 0
    public_pruned = 0
    deleted = 0
    async with store._connect() as db:
        cursor = await db.execute(
            "SELECT id, session_id, updated_at FROM jobs WHERE updated_at < ?",
            (heavy_cutoff.isoformat(),),
        )
        rows = await cursor.fetchall()
        for row in rows:
            job_dir = settings.job_dir(row["id"])
            if is_matrix_job(job_dir):
                continue
            if prune_heavy_artifacts(job_dir):
                heavy_pruned += 1
                write_artifact_manifest(job_dir, job_id=row["id"])

        cursor = await db.execute(
            "SELECT id, session_id, updated_at FROM jobs WHERE updated_at < ?",
            (public_cutoff.isoformat(),),
        )
        rows = await cursor.fetchall()
        for row in rows:
            if is_matrix_job(settings.job_dir(row["id"])):
                continue
            public_dir = settings.public_results_dir / row["id"]
            session_dir = settings.data_dir / "sessions" / row["session_id"]
            if public_dir.exists() or session_dir.exists():
                public_pruned += 1
            shutil.rmtree(public_dir, ignore_errors=True)
            shutil.rmtree(session_dir, ignore_errors=True)

        cursor = await db.execute(
            "SELECT id, session_id, updated_at FROM jobs WHERE updated_at < ?",
            (debug_cutoff.isoformat(),),
        )
        rows = await cursor.fetchall()
        for row in rows:
            if is_matrix_job(settings.job_dir(row["id"])):
                continue
            shutil.rmtree(settings.job_dir(row["id"]), ignore_errors=True)
            shutil.rmtree(settings.public_results_dir / row["id"], ignore_errors=True)
            shutil.rmtree(settings.data_dir / "sessions" / row["session_id"], ignore_errors=True)
            await db.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))
            await db.execute("DELETE FROM sessions WHERE id = ?", (row["session_id"],))
            deleted += 1
        await db.commit()
    print(
        f"pruned heavy artifacts for {heavy_pruned} job(s); "
        f"pruned public/session files for {public_pruned} job(s); "
        f"deleted {deleted} expired job(s)"
    )


def is_matrix_job(job_dir: Path) -> bool:
    metrics_path = job_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(metrics.get("matrix_run"))


def prune_heavy_artifacts(job_dir: Path) -> bool:
    targets = [
        job_dir / "source_media",
        job_dir / "candidate_frames",
        job_dir / "images",
        job_dir / "object_images",
        job_dir / "mask_artifacts",
        job_dir / "nerfstudio",
        job_dir / ARTIFACT_DIR_NAME / "source_media",
        job_dir / ARTIFACT_DIR_NAME / "frames",
        job_dir / ARTIFACT_DIR_NAME / "masks",
        job_dir / ARTIFACT_DIR_NAME / "training",
    ]
    removed = False
    for target in targets:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed = True
    return removed


async def artifacts(job_id: str, refresh: bool) -> None:
    settings = Settings()
    job_dir = settings.job_dir(job_id)
    if refresh:
        write_artifact_manifest(job_dir, job_id=job_id)
    manifest_path = job_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"artifact manifest not found for job {job_id}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"job {job_id}: {manifest.get('artifact_count', 0)} artifact(s)")
    for entry in manifest.get("artifacts", []):
        public = "public" if entry.get("public") else "private"
        print(
            f"{entry.get('size_bytes', 0):>12} {public:<7} "
            f"{entry.get('retention_tier', ''):<16} {entry.get('category', ''):<16} {entry.get('path')}"
        )


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

    artifacts_parser = subparsers.add_parser("artifacts")
    artifacts_parser.add_argument("job_id")
    artifacts_parser.add_argument("--refresh", action="store_true")

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
    elif args.command == "artifacts":
        asyncio.run(artifacts(args.job_id, args.refresh))


if __name__ == "__main__":
    main()

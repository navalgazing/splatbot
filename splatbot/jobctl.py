from __future__ import annotations

import argparse
import asyncio

from .config import Settings
from .models import ArtifactKind, JobStatus
from .notifications import TelegramNotifier
from .pipeline import PipelineOutputs
from .storage import Store
from .viewer import publish_viewer


async def set_status(job_id: str, status: JobStatus, error: str | None) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    await store.set_job_status(job_id, status, error)


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
    viewer_path = publish_viewer(settings, job_id, PipelineOutputs(ply, preview_output))
    artifacts = [
        await store.add_artifact(job_id, ArtifactKind.PLY, ply),
    ]
    if preview_output is not None:
        artifacts.append(await store.add_artifact(job_id, ArtifactKind.PREVIEW, preview_output))
    artifacts.append(
        await store.add_artifact(
            job_id,
            ArtifactKind.VIEWER,
            viewer_path,
            url=settings.public_job_url(job_id) or None,
        )
    )
    await store.set_job_status(job_id, JobStatus.DONE)
    updated = await store.get_job(job_id)
    if notify and settings.telegram_token and updated:
        await TelegramNotifier(settings.telegram_token).job_done(updated, artifacts)


async def fail(job_id: str, error: str, notify: bool) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    await store.set_job_failed_unless_terminal(job_id, error)
    job = await store.get_job(job_id)
    if notify and settings.telegram_token and job:
        await TelegramNotifier(settings.telegram_token).job_failed(job, error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Splatbot job state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser("set-status")
    set_parser.add_argument("job_id")
    set_parser.add_argument("status", choices=[status.value for status in JobStatus])
    set_parser.add_argument("--error")

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("job_id")
    complete_parser.add_argument("--notify", action="store_true")

    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("job_id")
    fail_parser.add_argument("--error", required=True)
    fail_parser.add_argument("--notify", action="store_true")

    args = parser.parse_args()
    if args.command == "set-status":
        asyncio.run(set_status(args.job_id, JobStatus(args.status), args.error))
    elif args.command == "complete":
        asyncio.run(complete(args.job_id, args.notify))
    elif args.command == "fail":
        asyncio.run(fail(args.job_id, args.error, args.notify))


if __name__ == "__main__":
    main()

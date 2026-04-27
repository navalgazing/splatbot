#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import shutil
import uuid
from pathlib import Path

from splatbot.config import ScanMode, Settings
from splatbot.media import classify_path
from splatbot.models import JobStatus, MediaKind, ScanJob, utcnow
from splatbot.runpod_backend import RunPodClient, RunPodLauncher
from splatbot.storage import Store


async def create_preparing_job(
    settings: Settings,
    store: Store,
    source: Path,
    telegram_user_id: int,
    mode: ScanMode,
) -> ScanJob:
    session = await store.create_session(telegram_user_id, mode)
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / source.name
    shutil.copy2(source, dest)
    kind = classify_path(dest)
    if kind not in {MediaKind.PHOTO, MediaKind.VIDEO}:
        raise ValueError(f"unsupported source media: {source}")
    await store.add_media(session.id, kind, dest)

    job_id = uuid.uuid4().hex
    now = utcnow().isoformat()
    async with store._connect() as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (JobStatus.QUEUED.value, now, session.id),
        )
        await db.execute(
            """
            INSERT INTO jobs (id, session_id, telegram_user_id, mode, status, error, created_at, updated_at,
                              claimed_at, heartbeat_at)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                job_id,
                session.id,
                telegram_user_id,
                mode.value,
                JobStatus.PREPARING.value,
                now,
                now,
                now,
                now,
            ),
        )
        await db.commit()
    job = await store.get_job(job_id)
    assert job is not None
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one controlled RunPod job from a local media file.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--telegram-user-id", required=True, type=int)
    parser.add_argument("--mode", choices=[mode.value for mode in ScanMode], default=ScanMode.OBJECT.value)
    parser.add_argument("--colmap-use-gpu", action="store_true")
    parser.add_argument("--colmap-bin", default="colmap")
    args = parser.parse_args()

    settings = Settings()
    settings.runpod_image_name = args.image
    settings.runpod_venv = "/opt/splatbot/venv"
    settings.runpod_bootstrap_command = ""
    settings.runpod_setup_command = ""
    settings.colmap_use_gpu = args.colmap_use_gpu
    settings.colmap_bin = args.colmap_bin

    store = Store(settings.database_path)
    asyncio.run(store.init())
    job = asyncio.run(
        create_preparing_job(
            settings,
            store,
            args.source,
            args.telegram_user_id,
            ScanMode(args.mode),
        )
    )
    print(f"created controlled job {job.id} session={job.session_id}", flush=True)

    def record_pod_id(pod_id: str | None) -> None:
        asyncio.run(store.set_job_runpod_pod_id(job.id, pod_id))

    launcher = RunPodLauncher(settings, client=RunPodClient(settings.runpod_api_key))
    try:
        pod = launcher.launch(job, record_pod_id)
        print(f"completed controlled job {job.id} on pod {pod.id}", flush=True)
    except Exception as exc:
        asyncio.run(store.set_job_failed_unless_terminal(job.id, str(exc)))
        raise


if __name__ == "__main__":
    main()

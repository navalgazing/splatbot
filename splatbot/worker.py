from __future__ import annotations

import argparse
import asyncio
import logging

from .config import Settings
from .models import ArtifactKind, JobStatus
from .pipeline import ScanPipeline
from .storage import Store

LOGGER = logging.getLogger(__name__)


async def run_job(job_id: str) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    job = await store.get_job(job_id)
    if job is None:
        raise SystemExit(f"job not found: {job_id}")
    media = await store.list_media(job.session_id)
    pipeline = ScanPipeline(settings)
    try:
        await store.set_job_status(job.id, JobStatus.PREPARING)
        outputs = await pipeline.run(job.id, job.mode, media, store.set_job_status)
        LOGGER.info("job %s complete: %s %s", job.id, outputs.cleaned_ply, outputs.preview_mp4)
        await store.add_artifact(job.id, ArtifactKind.PLY, outputs.cleaned_ply)
        await store.add_artifact(job.id, ArtifactKind.PREVIEW, outputs.preview_mp4)
        await store.set_job_status(job.id, JobStatus.DONE)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("job failed")
        await store.set_job_status(job.id, JobStatus.FAILED, str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Splatbot GPU job.")
    parser.add_argument("job_id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_job(args.job_id))

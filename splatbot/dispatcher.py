from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .artifacts import ArtifactRef, ArtifactStore
from .config import Settings
from .models import ArtifactKind, JobArtifact, JobStatus, ScanJob
from .notifications import TelegramNotifier
from .pipeline import PipelineOutputs, ScanPipeline
from .storage import Store

LOGGER = logging.getLogger(__name__)


class Dispatcher:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        pipeline: ScanPipeline | None = None,
        artifact_store: ArtifactStore | None = None,
        notifier: TelegramNotifier | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = pipeline or ScanPipeline(settings)
        self.artifact_store = artifact_store or ArtifactStore(settings)
        self.notifier = notifier

    def _upload_artifact(self, job: ScanJob, kind: ArtifactKind, path: Path) -> ArtifactRef | None:
        if not self.artifact_store.enabled:
            return None
        suffix = path.suffix or f".{kind.value}"
        return self.artifact_store.upload(path, f"jobs/{job.id}/{kind.value}{suffix}")

    async def _publish_artifacts(self, job: ScanJob, outputs: PipelineOutputs) -> list[JobArtifact]:
        published: list[JobArtifact] = []
        for kind, path in (
            (ArtifactKind.PLY, outputs.cleaned_ply),
            (ArtifactKind.PREVIEW, outputs.preview_mp4),
        ):
            ref = self._upload_artifact(job, kind, path)
            published.append(
                await self.store.add_artifact(
                    job.id,
                    kind,
                    path,
                    remote_key=ref.key if ref else None,
                    url=ref.url if ref else None,
                )
            )
        return published

    async def run_once(self) -> bool:
        job = await self.store.next_queued_job()
        if job is None:
            return False
        media = await self.store.list_media(job.session_id)
        try:
            await self.store.set_job_status(job.id, JobStatus.PREPARING)
            outputs = await self.pipeline.run(job.id, job.mode, media, self.store.set_job_status)
            artifacts = await self._publish_artifacts(job, outputs)
            LOGGER.info("job %s done: ply=%s preview=%s", job.id, outputs.cleaned_ply, outputs.preview_mp4)
            await self.store.set_job_status(job.id, JobStatus.DONE)
            if self.notifier:
                await self.notifier.job_done(job, artifacts)
        except Exception as exc:  # noqa: BLE001 - user-visible job failures should be persisted.
            LOGGER.exception("job %s failed", job.id)
            await self.store.set_job_status(job.id, JobStatus.FAILED, str(exc))
            if self.notifier:
                await self.notifier.job_failed(job, str(exc))
        return True

    async def run_forever(self, interval_seconds: float = 5.0) -> None:
        while True:
            worked = await self.run_once()
            if not worked:
                await asyncio.sleep(interval_seconds)


async def amain() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    notifier = TelegramNotifier(settings.telegram_token) if settings.telegram_token else None
    await Dispatcher(settings, store, notifier=notifier).run_forever()


def main() -> None:
    asyncio.run(amain())

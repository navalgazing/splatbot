from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .artifacts import ArtifactRef, ArtifactStore
from .config import Settings, WorkerBackend
from .logging_config import configure_logging
from .models import ArtifactKind, JobArtifact, JobStatus, ScanJob
from .notifications import TelegramNotifier
from .pipeline import PipelineOutputs, ScanPipeline
from .runpod_backend import RunPodLauncher
from .storage import Store
from .viewer import publish_viewer

LOGGER = logging.getLogger(__name__)


class Dispatcher:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        pipeline: ScanPipeline | None = None,
        artifact_store: ArtifactStore | None = None,
        notifier: TelegramNotifier | None = None,
        runpod_launcher: RunPodLauncher | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = pipeline or ScanPipeline(settings)
        self.artifact_store = artifact_store or ArtifactStore(settings)
        self.notifier = notifier
        self.runpod_launcher = runpod_launcher or RunPodLauncher(settings)

    def _upload_artifact(self, job: ScanJob, kind: ArtifactKind, path: Path) -> ArtifactRef | None:
        if not self.artifact_store.enabled:
            return None
        suffix = path.suffix or f".{kind.value}"
        return self.artifact_store.upload(path, f"jobs/{job.id}/{kind.value}{suffix}")

    async def _publish_artifacts(self, job: ScanJob, outputs: PipelineOutputs) -> list[JobArtifact]:
        published: list[JobArtifact] = []
        viewer_path = publish_viewer(self.settings, job.id, outputs)
        viewer_url = self.settings.public_job_url(job.id)
        artifacts_to_publish = [(ArtifactKind.PLY, outputs.cleaned_ply)]
        if outputs.preview_mp4 is not None and outputs.preview_mp4.exists():
            artifacts_to_publish.append((ArtifactKind.PREVIEW, outputs.preview_mp4))
        for kind, path in artifacts_to_publish:
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
        published.append(
            await self.store.add_artifact(
                job.id,
                ArtifactKind.VIEWER,
                viewer_path,
                url=viewer_url or None,
            )
        )
        return published

    async def run_once(self) -> bool:
        job = await self.store.claim_next_queued_job()
        if job is None:
            return False
        if self.settings.worker_backend == WorkerBackend.RUNPOD:
            try:
                pod = await asyncio.to_thread(self.runpod_launcher.launch, job)
                LOGGER.info("finished RunPod pod %s for job %s", pod.id, job.id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("RunPod job %s failed", job.id)
                updated = await self.store.set_job_failed_unless_terminal(job.id, str(exc))
                if self.notifier and updated and updated.status == JobStatus.FAILED and updated.error == str(exc):
                    await self.notifier.job_failed(updated, str(exc))
            return True
        media = await self.store.list_media(job.session_id)
        try:
            outputs = await self.pipeline.run(job.id, job.mode, media, self.store.set_job_status)
            artifacts = await self._publish_artifacts(job, outputs)
            LOGGER.info("job %s done: ply=%s preview=%s", job.id, outputs.cleaned_ply, outputs.preview_mp4)
            await self.store.set_job_status(job.id, JobStatus.DONE)
            if self.notifier:
                await self.notifier.job_done(job, artifacts)
        except Exception as exc:  # noqa: BLE001 - user-visible job failures should be persisted.
            LOGGER.exception("job %s failed", job.id)
            updated = await self.store.set_job_failed_unless_terminal(job.id, str(exc))
            if self.notifier and updated and updated.status == JobStatus.FAILED:
                await self.notifier.job_failed(updated, str(exc))
        return True

    async def run_forever(self, interval_seconds: float = 5.0) -> None:
        while True:
            worked = await self.run_once()
            if not worked:
                await asyncio.sleep(interval_seconds)


async def amain() -> None:
    configure_logging()
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    notifier = TelegramNotifier(settings.telegram_token) if settings.telegram_token else None
    interrupted = await store.fail_interrupted_jobs("Job interrupted by bot restart; please resubmit.")
    if interrupted:
        LOGGER.warning("marked %s interrupted job(s) failed on dispatcher startup", interrupted)
    await Dispatcher(settings, store, notifier=notifier).run_forever()


def main() -> None:
    asyncio.run(amain())

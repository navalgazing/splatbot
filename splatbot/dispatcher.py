from __future__ import annotations

import asyncio
import logging

from .artifacts import ArtifactStore
from .config import Settings, WorkerBackend
from .logging_config import configure_logging
from .models import JobArtifact, JobStatus, ScanJob
from .notifications import TelegramNotifier
from .pipeline import ScanPipeline
from .publishing import publish_job_artifacts
from .runpod_backend import RunPodLauncher
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
        runpod_launcher: RunPodLauncher | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = pipeline or ScanPipeline(settings)
        self.artifact_store = artifact_store or ArtifactStore(settings)
        self.notifier = notifier
        self.runpod_launcher = runpod_launcher or RunPodLauncher(settings)

    async def _notify_done(self, job: ScanJob, artifacts: list[JobArtifact]) -> None:
        if not self.notifier:
            return
        try:
            await self.notifier.job_done(job, artifacts)
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to notify user about completed job %s", job.id)

    async def _notify_failed(self, job: ScanJob, error: str) -> None:
        if not self.notifier:
            return
        try:
            await self.notifier.job_failed(job, error)
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to notify user about failed job %s", job.id)

    async def run_once(self) -> bool:
        job = await self.store.claim_next_queued_job()
        if job is None:
            return False
        if self.settings.worker_backend == WorkerBackend.RUNPOD:
            try:
                loop = asyncio.get_running_loop()

                def record_pod_id(pod_id: str | None) -> None:
                    future = asyncio.run_coroutine_threadsafe(
                        self.store.set_job_runpod_pod_id(job.id, pod_id),
                        loop,
                    )
                    future.result(timeout=10)

                pod = await asyncio.to_thread(self.runpod_launcher.launch, job, record_pod_id)
                LOGGER.info("finished RunPod pod %s for job %s", pod.id, job.id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("RunPod job %s failed", job.id)
                updated = await self.store.set_job_failed_unless_terminal(job.id, str(exc))
                if self.notifier and updated and updated.status == JobStatus.FAILED and updated.error == str(exc):
                    await self._notify_failed(updated, str(exc))
            return True
        if self.settings.worker_backend != WorkerBackend.LOCAL:
            updated = await self.store.set_job_failed_unless_terminal(
                job.id,
                f"Worker backend is not implemented: {self.settings.worker_backend.value}",
            )
            if self.notifier and updated and updated.status == JobStatus.FAILED:
                await self._notify_failed(updated, updated.error or "worker backend is not implemented")
            return True
        media = await self.store.list_media(job.session_id)
        try:
            outputs = await self.pipeline.run(job.id, job.mode, media, self.store.set_job_status)
            artifacts = await publish_job_artifacts(
                self.settings,
                self.store,
                job,
                outputs,
                self.artifact_store,
            )
            LOGGER.info("job %s done: ply=%s preview=%s", job.id, outputs.cleaned_ply, outputs.preview_mp4)
            await self.store.set_job_status(job.id, JobStatus.DONE)
            updated = await self.store.get_job(job.id)
            if updated and updated.status == JobStatus.DONE:
                await self._notify_done(updated, artifacts)
        except Exception as exc:  # noqa: BLE001 - user-visible job failures should be persisted.
            LOGGER.exception("job %s failed", job.id)
            updated = await self.store.set_job_failed_unless_terminal(job.id, str(exc))
            if self.notifier and updated and updated.status == JobStatus.FAILED:
                await self._notify_failed(updated, str(exc))
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
    interrupted = await recover_interrupted_jobs(settings, store, notifier)
    if interrupted:
        LOGGER.warning("marked %s interrupted job(s) failed on dispatcher startup", interrupted)
    await Dispatcher(settings, store, notifier=notifier).run_forever()


async def recover_interrupted_jobs(
    settings: Settings,
    store: Store,
    notifier: TelegramNotifier | None,
) -> int:
    interrupted = await store.fail_interrupted_jobs(
        "Job interrupted by bot restart; please resubmit.",
        grace_seconds=settings.interrupted_job_grace_seconds,
    )
    if settings.worker_backend == WorkerBackend.RUNPOD and settings.runpod_api_key:
        launcher = RunPodLauncher(settings)
        for job in interrupted:
            if job.runpod_pod_id:
                try:
                    await asyncio.to_thread(launcher.client.delete_pod, job.runpod_pod_id)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("failed to delete interrupted RunPod pod %s", job.runpod_pod_id)
        for job in await store.terminal_jobs_with_runpod_pods():
            if job.runpod_pod_id:
                try:
                    await asyncio.to_thread(launcher.client.delete_pod, job.runpod_pod_id)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("failed to delete terminal RunPod pod %s", job.runpod_pod_id)
                else:
                    await store.set_job_runpod_pod_id(job.id, None)
    if notifier:
        for job in interrupted:
            try:
                await notifier.job_failed(job, "Job interrupted by bot restart; please resubmit.")
            except Exception:  # noqa: BLE001
                LOGGER.exception("failed to notify user about interrupted job %s", job.id)
    return len(interrupted)


def main() -> None:
    asyncio.run(amain())

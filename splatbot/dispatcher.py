from __future__ import annotations

import asyncio
import logging
import signal

from .artifacts import ArtifactStore
from .config import Settings, WorkerBackend
from .job_overrides import settings_for_job
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
        self.pipeline = pipeline
        self.artifact_store = artifact_store
        self.notifier = notifier
        self.runpod_launcher = runpod_launcher

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
        try:
            job_settings = settings_for_job(self.settings, job)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("job %s has invalid settings overrides", job.id)
            updated = await self.store.set_job_failed_unless_terminal(job.id, str(exc))
            if self.notifier and updated and updated.status == JobStatus.FAILED:
                await self._notify_failed(updated, str(exc))
            return True
        if self.settings.worker_backend == WorkerBackend.RUNPOD:
            try:
                loop = asyncio.get_running_loop()

                def record_pod_id(pod_id: str | None) -> None:
                    future = asyncio.run_coroutine_threadsafe(
                        self.store.set_job_runpod_pod_id(job.id, pod_id),
                        loop,
                    )
                    future.result(timeout=10)

                launcher = self.runpod_launcher or RunPodLauncher(job_settings)
                pod = await asyncio.to_thread(launcher.launch, job, record_pod_id)
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
            pipeline = self.pipeline or ScanPipeline(job_settings)
            artifact_store = self.artifact_store or ArtifactStore(job_settings)
            outputs = await pipeline.run(job.id, job.mode, media, self.store.set_job_status, job.preset)
            artifacts = await publish_job_artifacts(
                job_settings,
                self.store,
                job,
                outputs,
                artifact_store,
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

    async def run_forever(self, interval_seconds: float = 5.0, stop_event: asyncio.Event | None = None) -> None:
        while stop_event is None or not stop_event.is_set():
            worked = await self.run_once()
            if not worked:
                try:
                    if stop_event:
                        await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                    else:
                        await asyncio.sleep(interval_seconds)
                except TimeoutError:
                    pass


async def amain() -> None:
    configure_logging()
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    notifier = TelegramNotifier(settings.telegram_token_value) if settings.telegram_token_value else None
    interrupted = await recover_interrupted_jobs(settings, store, notifier)
    if interrupted:
        LOGGER.warning("marked %s interrupted job(s) failed on dispatcher startup", interrupted)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass
    await Dispatcher(settings, store, notifier=notifier).run_forever(stop_event=stop_event)


async def recover_interrupted_jobs(
    settings: Settings,
    store: Store,
    notifier: TelegramNotifier | None,
) -> int:
    interrupted = await store.fail_interrupted_jobs(
        "Job interrupted by bot restart; please resubmit.",
        grace_seconds=0,
    )
    if settings.worker_backend == WorkerBackend.RUNPOD and settings.runpod_api_key_value:
        launcher = RunPodLauncher(settings)
        for job in interrupted:
            if job.runpod_pod_id:
                try:
                    await asyncio.to_thread(launcher.client.delete_pod, job.runpod_pod_id)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("failed to delete interrupted RunPod pod %s", job.runpod_pod_id)
                else:
                    await store.set_job_runpod_pod_id(job.id, None)
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

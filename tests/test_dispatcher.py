from datetime import UTC, datetime
from pathlib import Path

from splatbot.artifacts import ArtifactRef
from splatbot.config import ScanMode, Settings
from splatbot.dispatcher import Dispatcher, recover_interrupted_jobs
from splatbot.models import (
    ArtifactKind,
    JobArtifact,
    JobStatus,
    MediaItem,
    MediaKind,
    ScanJob,
)
from splatbot.pipeline import PipelineOutputs
from splatbot.storage import Store
from splatbot.viewer import VIEWER_PAGE_VERSION


class FakePipeline:
    async def run(self, job_id: str, mode: ScanMode, media: list[MediaItem], on_status=None, preset=None) -> PipelineOutputs:
        if on_status:
            await on_status(job_id, JobStatus.TRAINING)
        root = Path(media[0].local_path).parent
        ply = root / "clean.ply"
        preview = root / "preview.mp4"
        ply.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 1",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        preview.write_bytes(b"fake")
        return PipelineOutputs(cleaned_ply=ply, preview_mp4=preview)


class FakeArtifactStore:
    enabled = True

    def upload(self, path: Path, key: str) -> ArtifactRef:
        return ArtifactRef(key=key, url=f"https://example.test/{key}")


class FakeNotifier:
    def __init__(self) -> None:
        self.done: list[tuple[ScanJob, list[JobArtifact]]] = []
        self.failed: list[tuple[ScanJob, str]] = []

    async def job_done(self, job: ScanJob, artifacts: list[JobArtifact]) -> None:
        self.done.append((job, artifacts))

    async def job_failed(self, job: ScanJob, error: str) -> None:
        self.failed.append((job, error))


class TerminalRacePipeline:
    def __init__(self, store: Store) -> None:
        self.store = store

    async def run(self, job_id: str, mode: ScanMode, media: list[MediaItem], on_status=None, preset=None) -> PipelineOutputs:
        root = Path(media[0].local_path).parent
        ply = root / "clean.ply"
        ply.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 1",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        await self.store.set_job_failed_unless_terminal(job_id, "cancelled elsewhere")
        return PipelineOutputs(cleaned_ply=ply, preview_mp4=None)


async def test_dispatcher_runs_next_job(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=1, mode=ScanMode.SCENE)
    await store.add_media(session.id, MediaKind.VIDEO, tmp_path / "scan.mp4")
    job = await store.create_job(session)

    notifier = FakeNotifier()
    dispatcher = Dispatcher(
        Settings(
            data_dir=tmp_path,
            database_path=tmp_path / "splatbot.sqlite3",
            public_results_dir=tmp_path / "public",
            public_base_url="https://example.test",
        ),
        store,
        FakePipeline(),
        FakeArtifactStore(),
        notifier,
    )

    assert await dispatcher.run_once() is True
    assert (await store.get_job(job.id)).status.value == "done"
    artifacts = await store.list_artifacts(job.id)
    assert [artifact.kind for artifact in artifacts] == [
        ArtifactKind.PLY,
        ArtifactKind.PREVIEW,
        ArtifactKind.VIEWER,
    ]
    assert artifacts[0].url == f"https://example.test/jobs/{job.id}/ply.ply"
    assert artifacts[2].url == f"https://example.test/results/{job.id}/?v={VIEWER_PAGE_VERSION}"
    assert len(notifier.done) == 1


class FailingRunPodLauncher:
    def launch(self, job: ScanJob):
        raise RuntimeError("ssh died after remote completion")


async def test_dispatcher_does_not_overwrite_completed_runpod_job(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=1, mode=ScanMode.SCENE)
    job = await store.create_job(session)

    class CompletingThenFailingRunPodLauncher:
        def launch(self, job: ScanJob, on_pod_id=None):
            import asyncio

            if on_pod_id:
                on_pod_id("pod123")
            asyncio.run(store.set_job_status(job.id, JobStatus.DONE))
            raise RuntimeError("ssh died after remote completion")

    notifier = FakeNotifier()
    dispatcher = Dispatcher(
        Settings(
            data_dir=tmp_path,
            database_path=tmp_path / "splatbot.sqlite3",
            worker_backend="runpod",
            runpod_api_key="x",
        ),
        store,
        notifier=notifier,
        runpod_launcher=CompletingThenFailingRunPodLauncher(),
    )

    assert await dispatcher.run_once() is True
    assert (await store.get_job(job.id)).status == JobStatus.DONE
    assert notifier.failed == []


async def test_dispatcher_does_not_notify_done_if_terminal_state_wins_race(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=1, mode=ScanMode.SCENE)
    await store.add_media(session.id, MediaKind.PHOTO, tmp_path / "scan.jpg")
    job = await store.create_job(session)

    notifier = FakeNotifier()
    dispatcher = Dispatcher(
        Settings(
            data_dir=tmp_path,
            database_path=tmp_path / "splatbot.sqlite3",
            public_results_dir=tmp_path / "public",
            public_base_url="https://example.test",
        ),
        store,
        TerminalRacePipeline(store),
        FakeArtifactStore(),
        notifier,
    )

    assert await dispatcher.run_once() is True
    updated = await store.get_job(job.id)
    assert updated is not None
    assert updated.status == JobStatus.FAILED
    assert notifier.done == []


async def test_recover_interrupted_jobs_fails_fresh_claimed_jobs_on_startup(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=1, mode=ScanMode.SCENE)
    job = await store.create_job(session)
    claimed = await store.claim_next_queued_job()
    assert claimed is not None
    assert claimed.id == job.id

    notifier = FakeNotifier()
    interrupted = await recover_interrupted_jobs(
        Settings(
            data_dir=tmp_path,
            database_path=tmp_path / "splatbot.sqlite3",
            interrupted_job_grace_seconds=600,
        ),
        store,
        notifier,
    )

    assert interrupted == 1
    updated = await store.get_job(job.id)
    assert updated is not None
    assert updated.status == JobStatus.FAILED
    assert "interrupted by bot restart" in (updated.error or "")
    assert len(notifier.failed) == 1

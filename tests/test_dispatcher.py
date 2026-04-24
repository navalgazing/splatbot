from datetime import UTC, datetime
from pathlib import Path

from splatbot.artifacts import ArtifactRef
from splatbot.config import ScanMode, Settings
from splatbot.dispatcher import Dispatcher
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


class FakePipeline:
    async def run(self, job_id: str, mode: ScanMode, media: list[MediaItem], on_status=None) -> PipelineOutputs:
        if on_status:
            await on_status(job_id, JobStatus.TRAINING)
        return PipelineOutputs(cleaned_ply=Path("/tmp/clean.ply"), preview_mp4=Path("/tmp/preview.mp4"))


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


async def test_dispatcher_runs_next_job(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=1, mode=ScanMode.SCENE)
    await store.add_media(session.id, MediaKind.VIDEO, tmp_path / "scan.mp4")
    job = await store.create_job(session)

    notifier = FakeNotifier()
    dispatcher = Dispatcher(
        Settings(data_dir=tmp_path, database_path=tmp_path / "splatbot.sqlite3"),
        store,
        FakePipeline(),
        FakeArtifactStore(),
        notifier,
    )

    assert await dispatcher.run_once() is True
    assert (await store.get_job(job.id)).status.value == "done"
    artifacts = await store.list_artifacts(job.id)
    assert [artifact.kind for artifact in artifacts] == [ArtifactKind.PLY, ArtifactKind.PREVIEW]
    assert artifacts[0].url == f"https://example.test/jobs/{job.id}/ply.ply"
    assert len(notifier.done) == 1

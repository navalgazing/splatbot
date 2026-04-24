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
    assert artifacts[2].url == f"https://example.test/results/{job.id}/"
    assert len(notifier.done) == 1

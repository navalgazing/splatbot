from splatbot.config import ScanMode
from splatbot.models import ArtifactKind, JobStatus, MediaKind
from splatbot.storage import Store


async def test_session_media_and_job_lifecycle(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()

    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    active = await store.get_active_session(42)
    assert active == session

    media = await store.add_media(session.id, MediaKind.PHOTO, tmp_path / "a.jpg")
    assert media.kind == MediaKind.PHOTO
    assert len(await store.list_media(session.id)) == 1

    job = await store.create_job(session)
    assert job.status == JobStatus.QUEUED
    assert await store.get_active_session(42) is None
    assert (await store.next_queued_job()).id == job.id

    await store.set_job_status(job.id, JobStatus.DONE)
    assert (await store.get_job(job.id)).status == JobStatus.DONE
    assert (await store.get_latest_job_for_user(42)).id == job.id

    artifact = await store.add_artifact(
        job.id,
        ArtifactKind.PLY,
        tmp_path / "cleaned_splat.ply",
        remote_key="jobs/1/ply.ply",
        url="https://example.test/ply",
    )
    assert artifact.kind == ArtifactKind.PLY
    assert (await store.list_artifacts(job.id))[0].url == "https://example.test/ply"


async def test_claim_next_queued_job_is_atomic(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)

    claimed = await store.claim_next_queued_job()

    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == JobStatus.PREPARING
    assert await store.claim_next_queued_job() is None


async def test_set_job_failed_unless_terminal_preserves_done(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)
    await store.set_job_status(job.id, JobStatus.DONE)

    updated = await store.set_job_failed_unless_terminal(job.id, "late ssh failure")

    assert updated is not None
    assert updated.status == JobStatus.DONE
    assert updated.error is None

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


async def test_set_job_status_does_not_overwrite_terminal_job(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)
    await store.set_job_status(job.id, JobStatus.DONE)

    await store.set_job_status(job.id, JobStatus.COLMAP)
    updated = await store.get_job(job.id)

    assert updated is not None
    assert updated.status == JobStatus.DONE


async def test_fail_interrupted_jobs_marks_running_states_failed(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)
    await store.claim_next_queued_job()

    changed = await store.fail_interrupted_jobs("restart")

    assert [job.id for job in changed] == [job.id]
    updated = await store.get_job(job.id)
    assert updated is not None
    assert updated.status == JobStatus.FAILED
    assert updated.error == "restart"


async def test_create_job_is_idempotent_for_session(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)

    first = await store.create_job(session)
    second = await store.create_job(session)

    assert second.id == first.id


async def test_runpod_pod_id_can_be_recorded(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)

    await store.set_job_runpod_pod_id(job.id, "pod123")
    updated = await store.get_job(job.id)

    assert updated is not None
    assert updated.runpod_pod_id == "pod123"


async def test_terminal_jobs_with_runpod_pods(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)
    await store.set_job_runpod_pod_id(job.id, "pod123")
    await store.set_job_status(job.id, JobStatus.DONE)

    terminal = await store.terminal_jobs_with_runpod_pods()

    assert [job.id for job in terminal] == [job.id]


async def test_heartbeat_updates_job(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)

    await store.heartbeat_job(job.id)
    updated = await store.get_job(job.id)

    assert updated is not None
    assert updated.heartbeat_at is not None


async def test_add_artifact_replaces_existing_kind(tmp_path) -> None:
    store = Store(tmp_path / "splatbot.sqlite3")
    await store.init()
    session = await store.create_session(telegram_user_id=42, mode=ScanMode.OBJECT)
    job = await store.create_job(session)

    await store.add_artifact(job.id, ArtifactKind.PLY, tmp_path / "old.ply", url="old")
    await store.add_artifact(job.id, ArtifactKind.PLY, tmp_path / "new.ply", url="new")
    artifacts = await store.list_artifacts(job.id)

    assert len(artifacts) == 1
    assert artifacts[0].url == "new"

import json
from datetime import UTC, datetime

import pytest

from splatbot.config import Settings
from splatbot.job_overrides import job_overrides_path, settings_for_job
from splatbot.models import JobStatus, ScanJob, ScanMode


def make_job() -> ScanJob:
    now = datetime.now(UTC)
    return ScanJob(
        id="job123",
        session_id="session123",
        telegram_user_id=1,
        mode=ScanMode.OBJECT,
        status=JobStatus.QUEUED,
        error=None,
        created_at=now,
        updated_at=now,
    )


def test_settings_for_job_applies_matrix_overrides(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    job = make_job()
    path = job_overrides_path(settings, job.id)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "matrix_run": {"id": "matrix1", "row": "pose-vggt-colmap"},
                "settings": {
                    "best_pose_backends": "vggt-colmap",
                    "best_pose_required_backends": "vggt-colmap",
                },
            }
        ),
        encoding="utf-8",
    )

    overridden = settings_for_job(settings, job)

    assert overridden.best_pose_backends == "vggt-colmap"
    assert overridden.best_pose_required_backends == "vggt-colmap"
    assert json.loads(overridden.matrix_run_metadata)["row"] == "pose-vggt-colmap"
    assert settings.best_pose_backends != overridden.best_pose_backends


def test_settings_for_job_rejects_unknown_overrides(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    job = make_job()
    path = job_overrides_path(settings, job.id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"settings": {"does_not_exist": True}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown job settings override"):
        settings_for_job(settings, job)

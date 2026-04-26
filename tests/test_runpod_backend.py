import os
from datetime import UTC, datetime

import pytest

from splatbot.config import Settings
from splatbot.models import JobStatus, ScanJob, ScanMode
from splatbot.runpod_backend import (
    RunPodError,
    RunPodLauncher,
    RunPodSshTarget,
    render_remote_worker_command,
)


def make_runpod_settings(tmp_path, pod_key=None, vps_key=None) -> Settings:
    pod_key = pod_key or tmp_path / "pod_key"
    vps_key = vps_key or tmp_path / "vps_key"
    for key in (pod_key, vps_key):
        key.write_text("private", encoding="utf-8")
        os.chmod(key, 0o600)
    pod_key.with_suffix(".pub").write_text("public", encoding="utf-8")
    return Settings(
        default_scan_mode=ScanMode.SCENE,
        worker_backend="runpod",
        runpod_api_key="runpod-token",
        runpod_vps_host="203.0.113.10",
        runpod_vps_ssh_key=vps_key,
        runpod_pod_ssh_key=pod_key,
    )


def test_runpod_validate_requires_readable_vps_key(tmp_path) -> None:
    settings = make_runpod_settings(tmp_path)
    os.chmod(settings.runpod_vps_ssh_key, 0o000)

    with pytest.raises(RunPodError, match="SPLATBOT_RUNPOD_VPS_SSH_KEY is not readable"):
        RunPodLauncher(settings)._validate()


def test_runpod_validate_requires_readable_pod_key(tmp_path) -> None:
    settings = make_runpod_settings(tmp_path)
    os.chmod(settings.runpod_pod_ssh_key, 0o000)

    with pytest.raises(RunPodError, match="SPLATBOT_RUNPOD_POD_SSH_KEY is not readable"):
        RunPodLauncher(settings)._validate()


def test_pod_ssh_args_are_noninteractive_and_ephemeral(tmp_path) -> None:
    settings = make_runpod_settings(tmp_path)

    args = RunPodLauncher(settings)._pod_ssh_args(RunPodSshTarget("198.51.100.2", 30022))

    assert "BatchMode=yes" in args
    assert "IdentitiesOnly=yes" in args
    assert "StrictHostKeyChecking=no" in args
    assert "UserKnownHostsFile=/dev/null" in args
    assert "ServerAliveInterval=30" in args
    assert "ServerAliveCountMax=6" in args
    assert args[-2:] == ["30022", "root@198.51.100.2"]


def test_remote_worker_command_exports_pipeline_settings(tmp_path) -> None:
    settings = make_runpod_settings(tmp_path)
    job = ScanJob(
        id="job123",
        session_id="session123",
        telegram_user_id=42,
        mode=ScanMode.OBJECT,
        status=JobStatus.QUEUED,
        error=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    command = render_remote_worker_command(settings, job, "pod123", "a2V5")

    assert "export SPLATBOT_MAX_VIDEO_FRAMES=140" in command
    assert "export SPLATBOT_FFPROBE_BIN=ffprobe" in command
    assert "export SPLATBOT_TRAIN_MAX_ITERATIONS=10000" in command

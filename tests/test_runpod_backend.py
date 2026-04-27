import os
from datetime import UTC, datetime

import pytest

from splatbot.config import Settings
from splatbot.models import JobStatus, ScanJob, ScanMode
from splatbot.runpod_backend import (
    RunPodClient,
    RunPodError,
    RunPodLauncher,
    RunPodPod,
    RunPodSshUnavailableError,
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
    assert "export SPLATBOT_MAX_VIDEO_CANDIDATE_FPS=30.0" in command
    assert "export SPLATBOT_SCAN_PRESET=balanced" in command
    assert "export SPLATBOT_FFPROBE_BIN=ffprobe" in command
    assert "export SPLATBOT_ADAPTIVE_FRAME_SELECTION=true" in command
    assert "export SPLATBOT_FRAME_QUALITY_REJECT_THRESHOLD=35.0" in command
    assert "export SPLATBOT_OBJECT_COLMAP_ORIGINAL_POSE_FALLBACK=true" in command
    assert "export SPLATBOT_TRAIN_MAX_ITERATIONS=10000" in command
    assert "export SPLATBOT_COLMAP_USE_GPU=false" in command
    assert "export SPLATBOT_REMBG_REQUIRE_GPU=false" in command
    assert "export SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION=splatbot-runtime-2026-04-26-v1" in command
    assert "export SPLATBOT_RUNPOD_RUNTIME_CACHE_MARKER=/workspace/.splatbot-runtime-cache-version" in command
    assert "export SPLATBOT_LOG_COMMAND_OUTPUT=true" in command
    assert "SPLATBOT_RUNPOD_API_KEY" not in command


def test_launch_recycles_pods_without_public_ssh_endpoint(tmp_path) -> None:
    settings = make_runpod_settings(tmp_path)
    settings.runpod_no_endpoint_timeout_seconds = 0
    settings.runpod_launch_attempts = 2

    class EndpointlessClient:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.deleted: list[str] = []

        def create_ssh_pod(self, settings, job, public_key):
            pod_id = f"pod{len(self.created) + 1}"
            self.created.append(pod_id)
            return RunPodPod(id=pod_id, image_name=settings.runpod_image_name)

        def get_pod(self, pod_id: str) -> dict:
            return {"desiredStatus": "RUNNING", "publicIp": "", "portMappings": None}

        def delete_pod(self, pod_id: str) -> None:
            self.deleted.append(pod_id)

    job = ScanJob(
        id="job123",
        session_id="session123",
        telegram_user_id=42,
        mode=ScanMode.SCENE,
        status=JobStatus.QUEUED,
        error=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    client = EndpointlessClient()
    pod_ids: list[str | None] = []

    with pytest.raises(RunPodSshUnavailableError, match="never received a public SSH endpoint"):
        RunPodLauncher(settings, client=client).launch(job, pod_ids.append)

    assert client.created == ["pod1", "pod2"]
    assert client.deleted == ["pod1", "pod2"]
    assert pod_ids == ["pod1", None, "pod2", None]


def test_create_pod_uses_network_volume_and_datacenter_filters(tmp_path) -> None:
    settings = make_runpod_settings(tmp_path)
    settings.runpod_network_volume_id = "vol123"
    settings.runpod_data_center_ids = "EU-RO-1, EUR-IS-2"
    job = ScanJob(
        id="job123456789",
        session_id="session123",
        telegram_user_id=42,
        mode=ScanMode.SCENE,
        status=JobStatus.QUEUED,
        error=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    class CapturingClient(RunPodClient):
        def __init__(self) -> None:
            self.payload = None

        def request(self, method, path, payload=None):
            self.payload = payload
            return {"id": "pod123", "imageName": payload["imageName"]}

    client = CapturingClient()

    pod = client.create_ssh_pod(settings, job, "public")

    assert pod.id == "pod123"
    assert client.payload["networkVolumeId"] == "vol123"
    assert "volumeInGb" not in client.payload
    assert client.payload["dataCenterIds"] == ["EU-RO-1", "EUR-IS-2"]
    assert client.payload["dataCenterPriority"] == "availability"

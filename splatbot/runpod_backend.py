from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings
from .models import ScanJob


LOGGER = logging.getLogger(__name__)


class RunPodError(RuntimeError):
    pass


class RunPodApiError(RunPodError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"RunPod REST API failed: {status_code} {body}")


class RunPodSshUnavailableError(RunPodError):
    pass


@dataclass(frozen=True)
class RunPodPod:
    id: str
    image_name: str


@dataclass(frozen=True)
class RunPodSshTarget:
    host: str
    port: int


class RunPodClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def request(self, method: str, path: str, payload: dict | None = None) -> dict | list:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"https://rest.runpod.io/v1{path}",
            data=data,
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
                "user-agent": "splatbot/0.1 (+https://github.com/navalgazing/splatbot)",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RunPodApiError(exc.code, body) from exc

    def create_ssh_pod(self, settings: Settings, job: ScanJob, public_key: str) -> RunPodPod:
        cloud_type = settings.runpod_cloud_type
        if cloud_type == "ALL":
            cloud_type = "SECURE"
        payload = {
            "name": "splatbot-" + job.id[:12],
            "imageName": settings.runpod_image_name,
            "gpuTypeIds": [settings.runpod_gpu_type_id],
            "gpuCount": 1,
            "cloudType": cloud_type,
            "computeType": "GPU",
            "containerDiskInGb": settings.runpod_container_disk_gb,
            "volumeInGb": settings.runpod_volume_gb,
            "volumeMountPath": settings.runpod_volume_mount_path,
            "vcpuCount": settings.runpod_min_vcpu_count,
            "minRAMPerGPU": settings.runpod_min_memory_gb,
            "allowedCudaVersions": ["12.8", "12.9", "13.0"],
            "supportPublicIp": True,
            "ports": [settings.runpod_ports],
            "env": {"PUBLIC_KEY": public_key},
        }
        if settings.runpod_network_volume_id:
            payload["networkVolumeId"] = settings.runpod_network_volume_id
            payload.pop("volumeInGb", None)
        data_center_ids = [item.strip() for item in settings.runpod_data_center_ids.split(",") if item.strip()]
        if data_center_ids:
            payload["dataCenterIds"] = data_center_ids
            payload["dataCenterPriority"] = "availability"
        pod = self.request(
            "POST",
            "/pods",
            payload,
        )
        assert isinstance(pod, dict)
        return RunPodPod(id=pod["id"], image_name=pod["imageName"])

    def get_pod(self, pod_id: str) -> dict:
        pod = self.request("GET", f"/pods/{pod_id}")
        assert isinstance(pod, dict)
        return pod

    def delete_pod(self, pod_id: str) -> None:
        try:
            self.request("DELETE", f"/pods/{pod_id}")
        except RunPodApiError as exc:
            if exc.status_code == 404:
                return
            raise


class RunPodLauncher:
    def __init__(self, settings: Settings, client: RunPodClient | None = None) -> None:
        self.settings = settings
        self.client = client or RunPodClient(settings.runpod_api_key)

    def launch(self, job: ScanJob, on_pod_id: Callable[[str | None], None] | None = None) -> RunPodPod:
        self._validate()
        public_key = self.settings.runpod_pod_ssh_key.with_suffix(".pub").read_text().strip()
        attempts = max(1, self.settings.runpod_launch_attempts)
        last_error: RunPodSshUnavailableError | None = None
        for attempt in range(1, attempts + 1):
            pod = self.client.create_ssh_pod(self.settings, job, public_key)
            LOGGER.info("created RunPod pod %s for job %s attempt %s/%s", pod.id, job.id, attempt, attempts)
            if on_pod_id:
                on_pod_id(pod.id)
            try:
                target = self.wait_for_ssh(pod.id)
                self.run_worker(job, pod.id, target)
                return pod
            except RunPodSshUnavailableError as exc:
                last_error = exc
                LOGGER.warning("RunPod pod %s did not become SSH-ready: %s", pod.id, exc)
                if attempt >= attempts:
                    raise
            finally:
                try:
                    self.client.delete_pod(pod.id)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("failed to delete RunPod pod %s", pod.id)
                else:
                    if on_pod_id:
                        try:
                            on_pod_id(None)
                        except Exception:  # noqa: BLE001
                            LOGGER.exception("failed to clear RunPod pod id for job %s", job.id)
        assert last_error is not None
        raise last_error

    def _validate(self) -> None:
        if not self.settings.runpod_api_key:
            raise RunPodError("SPLATBOT_RUNPOD_API_KEY is required for RunPod backend")
        if not self.settings.runpod_vps_host:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_HOST is required for RunPod backend")
        if self.settings.runpod_vps_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_SSH_KEY is required for RunPod backend")
        if self.settings.runpod_pod_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_POD_SSH_KEY is required for RunPod backend")
        self._validate_readable_file(self.settings.runpod_vps_ssh_key, "SPLATBOT_RUNPOD_VPS_SSH_KEY")
        self._validate_readable_file(self.settings.runpod_pod_ssh_key, "SPLATBOT_RUNPOD_POD_SSH_KEY")
        self._validate_readable_file(self.settings.runpod_pod_ssh_key.with_suffix(".pub"), "pod SSH public key")

    @staticmethod
    def _validate_readable_file(path: Path, label: str) -> None:
        if not path.exists():
            raise RunPodError(f"{label} missing: {path}")
        if not path.is_file():
            raise RunPodError(f"{label} is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise RunPodError(f"{label} is not readable by this process: {path}")

    def wait_for_ssh(self, pod_id: str) -> RunPodSshTarget:
        started = time.monotonic()
        deadline = time.monotonic() + self.settings.runpod_ssh_ready_timeout_seconds
        last_seen = ""
        while time.monotonic() < deadline:
            pod = self.client.get_pod(pod_id)
            host = pod.get("publicIp") or ""
            port = (pod.get("portMappings") or {}).get("22")
            last_seen = f"host={host!r} port={port!r} status={pod.get('desiredStatus')!r}"
            if host and port:
                target = RunPodSshTarget(host=host, port=int(port))
                ready, ssh_error = self._ssh_ready(target)
                if ready:
                    return target
                if ssh_error:
                    last_seen = f"{last_seen} ssh_error={ssh_error!r}"
            elif time.monotonic() - started >= self.settings.runpod_no_endpoint_timeout_seconds:
                raise RunPodSshUnavailableError(
                    f"RunPod pod never received a public SSH endpoint before recycle timeout: {last_seen}"
                )
            time.sleep(10)
        raise RunPodSshUnavailableError(f"RunPod pod SSH was not ready before timeout: {last_seen}")

    def _ssh_ready(self, target: RunPodSshTarget) -> tuple[bool, str]:
        result = subprocess.run(
            [
                "ssh",
                *self._pod_ssh_args(target),
                "echo",
                "ready",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        error = result.stderr.decode(errors="replace").strip()[-500:]
        return result.returncode == 0, error

    def _pod_ssh_args(self, target: RunPodSshTarget) -> list[str]:
        return [
            "-i",
            str(self.settings.runpod_pod_ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            "-p",
            str(target.port),
            f"{self.settings.runpod_ssh_user}@{target.host}",
        ]

    def run_worker(self, job: ScanJob, pod_id: str, target: RunPodSshTarget) -> None:
        log_path = self.settings.job_dir(job.id) / "runpod-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        key_b64 = base64.b64encode(self.settings.runpod_vps_ssh_key.read_bytes()).decode()
        command = render_remote_worker_command(self.settings, job, pod_id, key_b64)
        ssh_command = ["ssh", *self._pod_ssh_args(target), "bash", "-s"]
        with log_path.open("ab") as log:
            log.write(f"\n--- RunPod worker {pod_id} on {target.host}:{target.port} ---\n".encode())
            result = subprocess.run(
                ssh_command,
                input=command.encode(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=self.settings.runpod_worker_timeout_seconds,
                check=False,
            )
        if result.returncode != 0:
            raise RunPodError(f"RunPod worker exited with {result.returncode}; see {log_path}")


def render_remote_worker_command(settings: Settings, job: ScanJob, pod_id: str, key_b64: str) -> str:
    host = shlex.quote(settings.runpod_vps_host)
    user = shlex.quote(settings.runpod_vps_user)
    bootstrap_command = shlex.quote(settings.runpod_bootstrap_command.strip())
    setup_command = shlex.quote(settings.runpod_setup_command.strip())
    runtime_cache_version = shlex.quote(settings.runpod_runtime_cache_version.strip())
    runtime_cache_marker = shlex.quote(settings.runpod_runtime_cache_marker.strip())
    venv_export = (
        f"export SPLATBOT_RUNPOD_VENV={shlex.quote(settings.runpod_venv.strip())}"
        if settings.runpod_venv.strip()
        else ""
    )
    pipeline_exports = "\n".join(
        f"export {name}={shlex.quote(str(value))}"
        for name, value in {
            "SPLATBOT_MAX_IMAGES": settings.max_images,
            "SPLATBOT_MAX_VIDEO_FRAMES": settings.max_video_frames,
            "SPLATBOT_MAX_VIDEO_SECONDS": settings.max_video_seconds,
            "SPLATBOT_MAX_VIDEO_SAMPLE_FPS": settings.max_video_sample_fps,
            "SPLATBOT_FFMPEG_BIN": settings.ffmpeg_bin,
            "SPLATBOT_FFPROBE_BIN": settings.ffprobe_bin,
            "SPLATBOT_COLMAP_BIN": settings.colmap_bin,
            "SPLATBOT_NS_PROCESS_DATA_BIN": settings.ns_process_data_bin,
            "SPLATBOT_NS_TRAIN_BIN": settings.ns_train_bin,
            "SPLATBOT_NS_EXPORT_BIN": settings.ns_export_bin,
            "SPLATBOT_NS_RENDER_BIN": settings.ns_render_bin,
            "SPLATBOT_REMBG_BIN": settings.rembg_bin,
            "SPLATBOT_COLMAP_USE_GPU": str(settings.colmap_use_gpu).lower(),
            "SPLATBOT_COMMAND_TIMEOUT_SECONDS": settings.command_timeout_seconds,
            "SPLATBOT_COMMAND_TAIL_BYTES": settings.command_tail_bytes,
            "SPLATBOT_TRAIN_MAX_ITERATIONS": settings.train_max_iterations,
            "SPLATBOT_TRAIN_STEPS_PER_SAVE": settings.train_steps_per_save,
            "SPLATBOT_RENDER_PREVIEW": str(settings.render_preview).lower(),
            "SPLATBOT_LOG_COMMAND_OUTPUT": "true",
        }.items()
    )
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /root/.ssh /workspace/splatbot-app
if ! command -v ssh >/dev/null || ! command -v rsync >/dev/null || ! command -v python3 >/dev/null; then
  apt-get update
  apt-get install -y openssh-client rsync python3
fi
printf %s {shlex.quote(key_b64)} | base64 -d > /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519
SSH_OPTS="-i /root/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
rsync -r --delete --no-perms --no-owner --no-group --omit-dir-times --exclude "__pycache__" --exclude "*.egg-info" -e "ssh $SSH_OPTS" {user}@{host}:/opt/splatbot/app/pyproject.toml /workspace/splatbot-app/
rsync -r --delete --no-perms --no-owner --no-group --omit-dir-times --exclude "__pycache__" --exclude "*.egg-info" -e "ssh $SSH_OPTS" {user}@{host}:/opt/splatbot/app/splatbot/ /workspace/splatbot-app/splatbot/
rsync -r --delete --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" {user}@{host}:/opt/splatbot/app/scripts/ /workspace/splatbot-app/scripts/
chmod +x /workspace/splatbot-app/scripts/runpod_worker.sh
export SPLATBOT_JOB_ID={shlex.quote(job.id)}
export SPLATBOT_SESSION_ID={shlex.quote(job.session_id)}
export SPLATBOT_SCAN_MODE={shlex.quote(job.mode.value)}
export SPLATBOT_RUNPOD_API_KEY={shlex.quote(settings.runpod_api_key)}
export RUNPOD_POD_ID={shlex.quote(pod_id)}
export SPLATBOT_VPS_HOST={host}
export SPLATBOT_VPS_USER={user}
{venv_export}
{pipeline_exports}
export SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND={bootstrap_command}
export SPLATBOT_RUNPOD_SETUP_COMMAND={setup_command}
export SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION={runtime_cache_version}
export SPLATBOT_RUNPOD_RUNTIME_CACHE_MARKER={runtime_cache_marker}
exec /workspace/splatbot-app/scripts/runpod_worker.sh
"""

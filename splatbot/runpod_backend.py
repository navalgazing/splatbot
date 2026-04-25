from __future__ import annotations

import base64
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .models import ScanJob


class RunPodError(RuntimeError):
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
            raise RunPodError(f"RunPod REST API failed: {exc.code} {body}") from exc

    def create_ssh_pod(self, settings: Settings, job: ScanJob, public_key: str) -> RunPodPod:
        cloud_type = settings.runpod_cloud_type
        if cloud_type == "ALL":
            cloud_type = "SECURE"
        pod = self.request(
            "POST",
            "/pods",
            {
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
            },
        )
        assert isinstance(pod, dict)
        return RunPodPod(id=pod["id"], image_name=pod["imageName"])

    def get_pod(self, pod_id: str) -> dict:
        pod = self.request("GET", f"/pods/{pod_id}")
        assert isinstance(pod, dict)
        return pod

    def delete_pod(self, pod_id: str) -> None:
        self.request("DELETE", f"/pods/{pod_id}")


class RunPodLauncher:
    def __init__(self, settings: Settings, client: RunPodClient | None = None) -> None:
        self.settings = settings
        self.client = client or RunPodClient(settings.runpod_api_key)

    def launch(self, job: ScanJob) -> RunPodPod:
        self._validate()
        public_key = self.settings.runpod_pod_ssh_key.with_suffix(".pub").read_text().strip()
        pod = self.client.create_ssh_pod(self.settings, job, public_key)
        try:
            target = self.wait_for_ssh(pod.id)
            self.run_worker(job, pod.id, target)
        finally:
            try:
                self.client.delete_pod(pod.id)
            except Exception:
                pass
        return pod

    def _validate(self) -> None:
        if not self.settings.runpod_api_key:
            raise RunPodError("SPLATBOT_RUNPOD_API_KEY is required for RunPod backend")
        if not self.settings.runpod_vps_host:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_HOST is required for RunPod backend")
        if self.settings.runpod_vps_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_SSH_KEY is required for RunPod backend")
        if self.settings.runpod_pod_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_POD_SSH_KEY is required for RunPod backend")
        if not self.settings.runpod_pod_ssh_key.exists():
            raise RunPodError(f"pod SSH key missing: {self.settings.runpod_pod_ssh_key}")
        if not self.settings.runpod_pod_ssh_key.with_suffix(".pub").exists():
            raise RunPodError(f"pod SSH public key missing: {self.settings.runpod_pod_ssh_key.with_suffix('.pub')}")

    def wait_for_ssh(self, pod_id: str) -> RunPodSshTarget:
        deadline = time.monotonic() + self.settings.runpod_ssh_ready_timeout_seconds
        last_seen = ""
        while time.monotonic() < deadline:
            pod = self.client.get_pod(pod_id)
            host = pod.get("publicIp") or ""
            port = (pod.get("portMappings") or {}).get("22")
            last_seen = f"host={host!r} port={port!r} status={pod.get('desiredStatus')!r}"
            if host and port:
                target = RunPodSshTarget(host=host, port=int(port))
                if self._ssh_ready(target):
                    return target
            time.sleep(10)
        raise RunPodError(f"RunPod pod SSH was not ready before timeout: {last_seen}")

    def _ssh_ready(self, target: RunPodSshTarget) -> bool:
        result = subprocess.run(
            [
                "ssh",
                *self._pod_ssh_args(target),
                "echo",
                "ready",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        return result.returncode == 0

    def _pod_ssh_args(self, target: RunPodSshTarget) -> list[str]:
        return [
            "-i",
            str(self.settings.runpod_pod_ssh_key),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(target.port),
            f"{self.settings.runpod_ssh_user}@{target.host}",
        ]

    def run_worker(self, job: ScanJob, pod_id: str, target: RunPodSshTarget) -> None:
        log_path = self.settings.job_dir(job.id) / "runpod-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        key_b64 = base64.b64encode(self.settings.runpod_vps_ssh_key.read_bytes()).decode()
        command = render_remote_worker_command(self.settings, job, pod_id, key_b64)
        ssh_command = ["ssh", *self._pod_ssh_args(target), "bash", "-lc", shlex.quote(command)]
        with log_path.open("ab") as log:
            log.write(f"\n--- RunPod worker {pod_id} on {target.host}:{target.port} ---\n".encode())
            result = subprocess.run(ssh_command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RunPodError(f"RunPod worker exited with {result.returncode}; see {log_path}")


def render_remote_worker_command(settings: Settings, job: ScanJob, pod_id: str, key_b64: str) -> str:
    host = shlex.quote(settings.runpod_vps_host)
    user = shlex.quote(settings.runpod_vps_user)
    setup_command = shlex.quote(settings.runpod_setup_command.strip())
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /root/.ssh /workspace/splatbot-app
apt-get update
apt-get install -y openssh-client rsync python3
printf %s {shlex.quote(key_b64)} | base64 -d > /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519
SSH_OPTS="-i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
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
export SPLATBOT_RUNPOD_SETUP_COMMAND={setup_command}
exec /workspace/splatbot-app/scripts/runpod_worker.sh
"""

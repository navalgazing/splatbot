from __future__ import annotations

import base64
import json
import shlex
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Settings
from .models import ScanJob


class RunPodError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPodPod:
    id: str
    image_name: str


class RunPodClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def graphql(self, query: str) -> dict:
        url = f"https://api.runpod.io/graphql?api_key={self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps({"query": query}).encode(),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "user-agent": "splatbot/0.1 (+https://github.com/navalgazing/splatbot)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RunPodError(f"RunPod API failed: {exc.code} {body}") from exc
        if payload.get("errors"):
            raise RunPodError(json.dumps(payload["errors"]))
        return payload["data"]

    def create_pod(self, settings: Settings, job: ScanJob, docker_args: str) -> RunPodPod:
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
            "ports": [settings.runpod_ports],
            "env": {
                "SPLATBOT_JOB_ID": job.id,
                "SPLATBOT_SESSION_ID": job.session_id,
                "SPLATBOT_SCAN_MODE": job.mode.value,
                "SPLATBOT_RUNPOD_API_KEY": self.api_key,
            },
            "dockerEntrypoint": [],
            "dockerStartCmd": ["bash", "-lc", docker_args],
        }
        request = urllib.request.Request(
            "https://rest.runpod.io/v1/pods",
            data=json.dumps(payload).encode(),
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
                "user-agent": "splatbot/0.1 (+https://github.com/navalgazing/splatbot)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                pod = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RunPodError(f"RunPod REST API failed: {exc.code} {body}") from exc
        return RunPodPod(id=pod["id"], image_name=pod["imageName"])


class RunPodLauncher:
    def __init__(self, settings: Settings, client: RunPodClient | None = None) -> None:
        self.settings = settings
        self.client = client or RunPodClient(settings.runpod_api_key)

    def launch(self, job: ScanJob) -> RunPodPod:
        if not self.settings.runpod_api_key:
            raise RunPodError("SPLATBOT_RUNPOD_API_KEY is required for RunPod backend")
        if not self.settings.runpod_vps_host:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_HOST is required for RunPod backend")
        if self.settings.runpod_vps_ssh_key is None:
            raise RunPodError("SPLATBOT_RUNPOD_VPS_SSH_KEY is required for RunPod backend")
        key_b64 = base64.b64encode(self.settings.runpod_vps_ssh_key.read_bytes()).decode()
        command = render_start_command(self.settings, key_b64)
        return self.client.create_pod(self.settings, job, command)


def render_start_command(settings: Settings, key_b64: str) -> str:
    host = shlex.quote(settings.runpod_vps_host)
    user = shlex.quote(settings.runpod_vps_user)
    repo = shlex.quote(settings.runpod_repo_url)
    setup_command = settings.runpod_setup_command.strip()
    quoted_setup = setup_command if setup_command else "true"
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git openssh-client rsync ffmpeg colmap python3 python3-venv python3-pip build-essential
mkdir -p /root/.ssh /workspace/input-media /workspace/results
printf %s {shlex.quote(key_b64)} | base64 -d > /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519
ssh-keyscan -H {host} >> /root/.ssh/known_hosts
git clone --depth=1 {repo} /workspace/splatbot-app
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install --upgrade pip
/workspace/venv/bin/pip install -e /workspace/splatbot-app
/workspace/venv/bin/pip install boto3
{quoted_setup}
ssh -i /root/.ssh/id_ed25519 {user}@{host} "/opt/splatbot/venv/bin/splatbot-jobctl set-status $SPLATBOT_JOB_ID preparing"
rsync -az -e "ssh -i /root/.ssh/id_ed25519" {user}@{host}:/var/lib/splatbot/sessions/$SPLATBOT_SESSION_ID/ /workspace/input-media/
set +e
/workspace/venv/bin/splatbot-run-job-dir "$SPLATBOT_JOB_ID" "$SPLATBOT_SCAN_MODE" /workspace/input-media /workspace/results
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  ssh -i /root/.ssh/id_ed25519 {user}@{host} "mkdir -p /var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/export /var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/renders"
  rsync -az -e "ssh -i /root/.ssh/id_ed25519" /workspace/results/cleaned_splat.ply {user}@{host}:/var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/export/cleaned_splat.ply
  rsync -az -e "ssh -i /root/.ssh/id_ed25519" /workspace/results/turntable.mp4 {user}@{host}:/var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/renders/turntable.mp4
  ssh -i /root/.ssh/id_ed25519 {user}@{host} "/opt/splatbot/venv/bin/splatbot-jobctl complete $SPLATBOT_JOB_ID --notify"
else
  ssh -i /root/.ssh/id_ed25519 {user}@{host} "/opt/splatbot/venv/bin/splatbot-jobctl fail $SPLATBOT_JOB_ID --error 'RunPod worker failed with exit code $rc' --notify"
  exit "$rc"
fi
if [ -n "${{RUNPOD_POD_ID:-}}" ]; then
  curl -fsS --request DELETE --header "Authorization: Bearer $SPLATBOT_RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" || true
fi
"""

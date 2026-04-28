#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from splatbot.config import ScanMode, Settings
from splatbot.models import JobStatus, ScanJob
from splatbot.runpod_backend import RunPodClient, RunPodLauncher


SMOKE_SCRIPT = r"""
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PATH=/opt/splatbot/venv/bin:/usr/local/cuda/bin:$PATH
export SPLATBOT_SAM2_CHECKPOINT=/opt/splatbot/models/sam2.1_hiera_large.pt
export SPLATBOT_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
echo image smoke
command -v colmap
colmap -h > /tmp/colmap-help.txt 2>&1
sed -n '1,5p' /tmp/colmap-help.txt
grep -q 'with CUDA' /tmp/colmap-help.txt
colmap feature_extractor -h > /tmp/feature-help.txt 2>&1
grep -m1 'SiftExtraction.use_gpu' /tmp/feature-help.txt
/opt/splatbot/venv/bin/pip install --no-build-isolation --no-deps -e /workspace/splatbot-app
/opt/splatbot/venv/bin/python - <<'PY'
import importlib.metadata as metadata
import importlib
import importlib.util
from pathlib import Path
import onnxruntime as ort
import torch

def require_import(module: str) -> None:
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(f"{module} is not importable: {exc}") from exc
    print(f"{module}=importable")

def optional_import(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        print(f"{module}=missing")
        return
    require_import(module)

print("torch_cuda_available=" + str(torch.cuda.is_available()))
for package in (
    "nerfstudio",
    "gsplat",
    "rembg",
    "onnxruntime",
    "onnxruntime-gpu",
    "depth-anything-3",
    "xformers",
):
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "missing"
    print(f"{package}={version}")
require_import("sam2")
require_import("depth_anything_3")
require_import("xformers")
optional_import("sam3")
checkpoint = Path("/opt/splatbot/models/sam2.1_hiera_large.pt")
if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
    raise SystemExit(f"SAM2 checkpoint is missing: {checkpoint}")
print("onnxruntime_device=" + ort.get_device())
print("onnxruntime_providers=" + str(ort.get_available_providers()))
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("onnxruntime CUDAExecutionProvider is not available")
PY
for cmd in splatbot-segment splatbot-pose splatbot-depth splatbot-train splatbot-mesh splatbot-da3; do
  command -v "$cmd"
  "$cmd" --help >/tmp/"$cmd"-help.txt
done
splatbot-segment --backend sam2 --self-test
ns-train splatfacto-big --help | grep -q -- "--pipeline.model.strategy"
splatbot-da3 --help >/tmp/splatbot-da3-help.txt
if /opt/splatbot/venv/bin/python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("sam3") is not None else 1)
PY
then
  splatbot-segment --backend sam3 --self-test
fi
"""


def sync_app_to_pod(repo_root: Path, launcher: RunPodLauncher, target) -> None:
    ssh_args = launcher._pod_ssh_args(target)
    ssh_transport = shlex.join(["ssh", *ssh_args[:-1]])
    remote = ssh_args[-1]
    subprocess.run(
        ["ssh", *ssh_args, "mkdir", "-p", "/workspace/splatbot-app"],
        check=True,
    )
    rsync_base = [
        "rsync",
        "-r",
        "--delete",
        "--no-perms",
        "--no-owner",
        "--no-group",
        "--omit-dir-times",
        "--exclude",
        "__pycache__",
        "--exclude",
        "*.egg-info",
        "-e",
        ssh_transport,
    ]
    for item in ("pyproject.toml", "splatbot", "scripts"):
        subprocess.run(
            [*rsync_base, str(repo_root / item), f"{remote}:/workspace/splatbot-app/"],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a RunPod Splatbot image over SSH.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    settings = Settings()
    settings.runpod_image_name = args.image
    settings.runpod_venv = "/opt/splatbot/venv"
    settings.runpod_bootstrap_command = ""
    settings.runpod_setup_command = ""
    settings.colmap_use_gpu = True

    job = ScanJob(
        id="smokecudacolmap",
        session_id="smoke",
        telegram_user_id=0,
        mode=ScanMode.SCENE,
        status=JobStatus.QUEUED,
        error=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    client = RunPodClient(settings.runpod_api_key_value)
    launcher = RunPodLauncher(settings, client=client)
    public_key = settings.runpod_pod_ssh_key.with_suffix(".pub").read_text().strip()
    pod = None
    try:
        pod = client.create_ssh_pod(settings, job, public_key)
        print(f"created pod {pod.id} image={pod.image_name}", flush=True)
        target = launcher.wait_for_ssh(pod.id)
        print(f"ssh ready {target.host}:{target.port}", flush=True)
        sync_app_to_pod(Path(__file__).resolve().parents[1], launcher, target)
        result = subprocess.run(
            ["ssh", *launcher._pod_ssh_args(target), "bash", "-s"],
            input=SMOKE_SCRIPT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_seconds,
            check=False,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            raise SystemExit(f"smoke command failed: {result.returncode}")
    finally:
        if pod is not None:
            client.delete_pod(pod.id)
            print(f"deleted pod {pod.id}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from splatbot.config import ScanMode, Settings
from splatbot.models import JobStatus, ScanJob
from splatbot.runpod_backend import RunPodClient, RunPodError, RunPodLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm and verify the Splatbot RunPod network-volume runtime cache.")
    parser.add_argument("--timeout", type=int, default=7200, help="remote warm command timeout in seconds")
    parser.add_argument("--log-path", type=Path, default=None, help="local path for warm/smoke logs")
    parser.add_argument("--keep-pod-on-failure", action="store_true", help="leave the smoke pod running when checks fail")
    return parser.parse_args()


def shell_export(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def remote_script(settings: Settings) -> str:
    venv_dir = settings.runpod_venv.strip() or "/workspace/venv"
    cache_marker = settings.runpod_runtime_cache_marker.strip() or "/workspace/.splatbot-runtime-cache-version"
    return f"""#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export QT_QPA_PLATFORM="${{QT_QPA_PLATFORM:-offscreen}}"
export CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda}}"
export PATH="$CUDA_HOME/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/cuda/lib64:${{LD_LIBRARY_PATH:-}}"
export TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST:-8.9}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}}"
export TORCH_EXTENSIONS_DIR="${{TORCH_EXTENSIONS_DIR:-/workspace/torch_extensions}}"
export TORCHINDUCTOR_CACHE_DIR="${{TORCHINDUCTOR_CACHE_DIR:-/workspace/torch_inductor}}"
export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-/workspace/triton_cache}}"
export CUDA_CACHE_PATH="${{CUDA_CACHE_PATH:-/workspace/cuda_cache}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/workspace/.cache}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export U2NET_HOME="${{U2NET_HOME:-/workspace/.u2net}}"
{shell_export("VENV_DIR", venv_dir)}
{shell_export("CACHE_MARKER", cache_marker)}
{shell_export("CACHE_VERSION", settings.runpod_runtime_cache_version.strip())}
{shell_export("BOOTSTRAP_COMMAND", settings.runpod_bootstrap_command.strip())}
{shell_export("SETUP_COMMAND", settings.runpod_setup_command.strip())}
{shell_export("SPLATBOT_COLMAP_USE_GPU", str(settings.colmap_use_gpu).lower())}
{shell_export("SPLATBOT_REMBG_REQUIRE_GPU", str(settings.rembg_require_gpu).lower())}

echo "warming runtime cache version: $CACHE_VERSION"
echo "venv: $VENV_DIR"
echo "cache marker: $CACHE_MARKER"

if [ -n "$BOOTSTRAP_COMMAND" ]; then
  bash -lc "$BOOTSTRAP_COMMAND"
fi

mkdir -p "$TORCH_EXTENSIONS_DIR" "$U2NET_HOME"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
fi

CACHE_READY=false
if [ -n "$CACHE_VERSION" ] && [ -r "$CACHE_MARKER" ] && [ "$(cat "$CACHE_MARKER")" = "$CACHE_VERSION" ]; then
  CACHE_READY=true
fi

if [ "$CACHE_READY" != true ]; then
  if [ -z "$SETUP_COMMAND" ]; then
    echo "runtime cache is missing or stale, but SPLATBOT_RUNPOD_SETUP_COMMAND is empty" >&2
    exit 2
  fi
  bash -lc "$SETUP_COMMAND"
fi

export PATH="$VENV_DIR/bin:$PATH"

python - <<'PY'
import os
from pathlib import Path
import urllib.request

model = Path(os.environ["U2NET_HOME"]) / "u2net.onnx"
model.parent.mkdir(parents=True, exist_ok=True)
if not model.exists() or model.stat().st_size < 1_000_000:
    urllib.request.urlretrieve(
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        model,
    )
print(f"rembg model bytes={{model.stat().st_size}}")
PY

command -v ffmpeg
command -v ffprobe
command -v colmap
command -v nvcc
command -v ns-process-data
command -v ns-train
command -v ns-export
command -v ns-render
command -v rembg
for cmd in ns-process-data ns-train ns-export ns-render rembg; do
  "$cmd" --help >/tmp/"$cmd"-help.txt
done

truthy() {{
  case "${{1:-}}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}}

COLMAP_HELP="$(colmap -h 2>&1)"
printf '%s\n' "$COLMAP_HELP" | sed -n '1p'
if truthy "$SPLATBOT_COLMAP_USE_GPU" && ! printf '%s\n' "$COLMAP_HELP" | grep -q "with CUDA"; then
  echo "SPLATBOT_COLMAP_USE_GPU=true requires a CUDA-enabled COLMAP build" >&2
  exit 2
fi

python - <<'PY'
import importlib
import importlib.util
import os

import torch
import onnxruntime as ort

def require_import(module: str) -> None:
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(f"{{module}} is not importable: {{exc}}") from exc
    print(f"{{module}}=importable")

def optional_import(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        print(f"{{module}}=missing")
        return
    require_import(module)

print(f"torch={{torch.__version__}} cuda_available={{torch.cuda.is_available()}}")
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA is not available")

from gsplat.cuda import _backend

print(f"gsplat_backend={{_backend._C is not None}}")
if _backend._C is None:
    raise SystemExit("gsplat CUDA extension is not available")

providers = ort.get_available_providers()
print(f"onnxruntime_device={{ort.get_device()}} providers={{providers}}")
require_rembg_gpu = os.getenv("SPLATBOT_REMBG_REQUIRE_GPU", "").lower() in {"1", "true", "yes", "on"}
if require_rembg_gpu and "CUDAExecutionProvider" not in providers:
    raise SystemExit("SPLATBOT_REMBG_REQUIRE_GPU=true but ONNX Runtime has no CUDAExecutionProvider")
require_import("sam2")
optional_import("sam3")
PY

if [ -n "$CACHE_VERSION" ]; then
  mkdir -p "$(dirname "$CACHE_MARKER")"
  printf '%s\\n' "$CACHE_VERSION" > "$CACHE_MARKER"
fi

du -sh "$VENV_DIR" "$TORCH_EXTENSIONS_DIR" "$U2NET_HOME" || true
echo "runtime cache verified"
"""


def warm_volume(settings: Settings, args: argparse.Namespace) -> int:
    if not settings.runpod_network_volume_id:
        raise RunPodError("SPLATBOT_RUNPOD_NETWORK_VOLUME_ID is required to warm a persistent volume")

    log_path = args.log_path or settings.data_dir / "network-volume-warm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    client = RunPodClient(settings.runpod_api_key)
    launcher = RunPodLauncher(settings, client=client)
    launcher._validate()

    now = datetime.now(UTC)
    job = ScanJob(
        id="warm" + now.strftime("%Y%m%d%H%M%S"),
        session_id="warm",
        telegram_user_id=0,
        mode=ScanMode.SCENE,
        status=JobStatus.QUEUED,
        error=None,
        created_at=now,
        updated_at=now,
    )

    public_key = settings.runpod_pod_ssh_key.with_suffix(".pub").read_text().strip()
    pod = client.create_ssh_pod(settings, job, public_key)
    print(f"created warm pod {pod.id} image={pod.image_name}", flush=True)
    delete_pod = True
    try:
        started = time.monotonic()
        target = launcher.wait_for_ssh(pod.id)
        print(f"ssh ready {target.host}:{target.port} after={time.monotonic() - started:.1f}s", flush=True)
        command = ["ssh", *launcher._pod_ssh_args(target), "bash", "-s"]
        with log_path.open("ab") as log:
            log.write(f"\n--- RunPod runtime warm {pod.id} {now.isoformat()} ---\n".encode())
            result = subprocess.run(
                command,
                input=remote_script(settings).encode(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        print(f"warm command exit={result.returncode}; log={log_path}", flush=True)
        if result.returncode != 0:
            delete_pod = not args.keep_pod_on_failure
        return result.returncode
    finally:
        if delete_pod:
            client.delete_pod(pod.id)
            print(f"deleted warm pod {pod.id}", flush=True)
        else:
            print(f"kept warm pod {pod.id} for debugging", flush=True)


def main() -> None:
    args = parse_args()
    settings = Settings()
    try:
        raise SystemExit(warm_volume(settings, args))
    except RunPodError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

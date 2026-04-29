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
DEFAULT_TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
detect_cuda_arch_list() {{
  local compute_cap
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if printf '%s' "$compute_cap" | grep -Eq '^[0-9]+\\.[0-9]+$'; then
    printf '%s\n' "$compute_cap"
    return 0
  fi
  python3 - <<'PY' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{{major}}.{{minor}}")
PY
}}
if [ -z "${{TORCH_CUDA_ARCH_LIST:-}}" ] || [ "${{TORCH_CUDA_ARCH_LIST:-}}" = "8.9" ] || [ "${{TORCH_CUDA_ARCH_LIST:-}}" = "$DEFAULT_TORCH_CUDA_ARCH_LIST" ]; then
  detected_arch="$(detect_cuda_arch_list | head -n 1)"
  export TORCH_CUDA_ARCH_LIST="${{detected_arch:-$DEFAULT_TORCH_CUDA_ARCH_LIST}}"
else
  export TORCH_CUDA_ARCH_LIST
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}}"
export TORCH_EXTENSIONS_DIR="${{TORCH_EXTENSIONS_DIR:-/workspace/torch_extensions}}"
export TORCHINDUCTOR_CACHE_DIR="${{TORCHINDUCTOR_CACHE_DIR:-/workspace/torch_inductor}}"
export TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-/workspace/triton_cache}}"
export CUDA_CACHE_PATH="${{CUDA_CACHE_PATH:-/workspace/cuda_cache}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/workspace/.cache}}"
{shell_export("DEFAULT_TORCH_HOME", settings.torch_home)}
export TORCH_HOME="${{TORCH_HOME:-$DEFAULT_TORCH_HOME}}"
export U2NET_HOME="${{U2NET_HOME:-/opt/splatbot/models/rembg}}"
{shell_export("VENV_DIR", venv_dir)}
{shell_export("CACHE_MARKER", cache_marker)}
{shell_export("CACHE_VERSION", settings.runpod_runtime_cache_version.strip())}
{shell_export("BOOTSTRAP_COMMAND", settings.runpod_bootstrap_command.strip())}
{shell_export("SETUP_COMMAND", settings.runpod_setup_command.strip())}
{shell_export("SPLATBOT_COLMAP_USE_GPU", str(settings.colmap_use_gpu).lower())}
{shell_export("SPLATBOT_REMBG_REQUIRE_GPU", str(settings.rembg_require_gpu).lower())}
{shell_export("SPLATBOT_MAST3R_USE_GLOMAP", str(settings.mast3r_use_glomap).lower())}
{shell_export("SPLATBOT_GLOMAP_BIN", settings.glomap_bin)}
{shell_export("SPLATBOT_GLOMAP_MAPPER_ARGS", settings.glomap_mapper_args)}
{shell_export("SPLATBOT_VGGT_WEIGHTS", settings.vggt_weights)}
{shell_export("SPLATBOT_VGGSFM_TRACKER_WEIGHTS", settings.vggsfm_tracker_weights)}
{shell_export("SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD", str(settings.vggt_allow_weight_download).lower())}
{shell_export("SPLATBOT_MAST3R_WEIGHTS", settings.mast3r_weights)}
{shell_export("SPLATBOT_MAST3R_WEIGHTS_URL", settings.mast3r_weights_url)}
{shell_export("SPLATBOT_MAST3R_ALLOW_WEIGHT_DOWNLOAD", str(settings.mast3r_allow_weight_download).lower())}
{shell_export("SPLATBOT_DA3_MODEL", settings.da3_model)}
{shell_export("SPLATBOT_DA3_MODEL_CACHE_DIR", settings.da3_model_cache_dir)}
{shell_export("SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD", str(settings.da3_allow_model_download).lower())}
{shell_export("SPLATBOT_DA3_MODEL_DOWNLOAD_ATTEMPTS", str(settings.da3_model_download_attempts))}
{shell_export("SPLATBOT_TORCHVISION_ALLOW_WEIGHT_DOWNLOAD", str(settings.torchvision_allow_weight_download).lower())}

echo "warming runtime cache version: $CACHE_VERSION"
echo "venv: $VENV_DIR"
echo "cache marker: $CACHE_MARKER"

if [ -n "$BOOTSTRAP_COMMAND" ]; then
  bash -lc "$BOOTSTRAP_COMMAND"
fi

mkdir -p "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$U2NET_HOME"
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
import shutil
import urllib.request

model = Path(os.environ["U2NET_HOME"]) / "u2net.onnx"
model.parent.mkdir(parents=True, exist_ok=True)
if not model.exists() or model.stat().st_size < 1_000_000:
    for candidate in (Path("/opt/splatbot/models/rembg/u2net.onnx"), Path("/root/.u2net/u2net.onnx")):
        if candidate.exists() and candidate.stat().st_size >= 1_000_000:
            shutil.copy2(candidate, model)
            break
    else:
        urllib.request.urlretrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            model,
        )
print(f"rembg model bytes={{model.stat().st_size}}")

mast3r_weights = Path(os.environ["SPLATBOT_MAST3R_WEIGHTS"])
mast3r_weights.parent.mkdir(parents=True, exist_ok=True)
if mast3r_weights.exists() and mast3r_weights.stat().st_size >= 1_000_000_000:
    print(f"mast3r weights bytes={{mast3r_weights.stat().st_size}}")
elif os.environ.get("SPLATBOT_MAST3R_ALLOW_WEIGHT_DOWNLOAD", "true").lower() in {"1", "true", "yes", "on"}:
    partial = mast3r_weights.with_suffix(mast3r_weights.suffix + ".part")
    if partial.exists():
        partial.unlink()
    urllib.request.urlretrieve(os.environ["SPLATBOT_MAST3R_WEIGHTS_URL"], partial)
    if partial.stat().st_size < 1_000_000_000:
        raise SystemExit(f"downloaded MASt3R weights are too small: {{partial}}")
    partial.replace(mast3r_weights)
    print(f"mast3r weights bytes={{mast3r_weights.stat().st_size}}")
else:
    raise SystemExit(f"MASt3R weights are missing at {{mast3r_weights}} and downloads are disabled")

torch_home = Path(os.environ["TORCH_HOME"])
alexnet = torch_home / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
if alexnet.exists() and alexnet.stat().st_size >= 100_000_000:
    print(f"alexnet checkpoint bytes={{alexnet.stat().st_size}}")
elif os.environ.get("SPLATBOT_TORCHVISION_ALLOW_WEIGHT_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"}:
    from torch.hub import download_url_to_file

    alexnet.parent.mkdir(parents=True, exist_ok=True)
    partial = alexnet.with_suffix(alexnet.suffix + ".part")
    if partial.exists():
        partial.unlink()
    download_url_to_file(
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        str(partial),
        hash_prefix="7be5be79",
        progress=True,
    )
    partial.replace(alexnet)
    print(f"alexnet checkpoint bytes={{alexnet.stat().st_size}}")
else:
    raise SystemExit(f"AlexNet LPIPS checkpoint is missing at {{alexnet}} and downloads are disabled")

vggt_weights = Path(os.environ["SPLATBOT_VGGT_WEIGHTS"])
if vggt_weights.exists() and vggt_weights.stat().st_size >= 4_000_000_000:
    print(f"vggt weights bytes={{vggt_weights.stat().st_size}}")
elif os.environ.get("SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"}:
    from torch.hub import download_url_to_file

    vggt_weights.parent.mkdir(parents=True, exist_ok=True)
    partial = vggt_weights.with_suffix(vggt_weights.suffix + ".part")
    if partial.exists():
        partial.unlink()
    download_url_to_file(
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        str(partial),
        progress=True,
    )
    partial.replace(vggt_weights)
    print(f"vggt weights bytes={{vggt_weights.stat().st_size}}")
else:
    raise SystemExit(f"VGGT weights are missing at {{vggt_weights}} and downloads are disabled")

vggsfm_tracker = Path(os.environ["SPLATBOT_VGGSFM_TRACKER_WEIGHTS"])
if vggsfm_tracker.exists() and vggsfm_tracker.stat().st_size >= 100_000_000:
    print(f"vggsfm tracker bytes={{vggsfm_tracker.stat().st_size}}")
elif os.environ.get("SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD", "false").lower() in {"1", "true", "yes", "on"}:
    from torch.hub import download_url_to_file

    vggsfm_tracker.parent.mkdir(parents=True, exist_ok=True)
    partial = vggsfm_tracker.with_suffix(vggsfm_tracker.suffix + ".part")
    if partial.exists():
        partial.unlink()
    download_url_to_file(
        "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt",
        str(partial),
        progress=True,
    )
    partial.replace(vggsfm_tracker)
    print(f"vggsfm tracker bytes={{vggsfm_tracker.stat().st_size}}")
else:
    raise SystemExit(f"VGGSfM tracker weights are missing at {{vggsfm_tracker}} and downloads are disabled")
PY

python - <<'PY'
import os
import re
from pathlib import Path
from huggingface_hub import snapshot_download

model = os.environ["SPLATBOT_DA3_MODEL"]
safe_model = re.sub(r"[^A-Za-z0-9._-]+", "__", model)
cache_dir = Path(os.environ["SPLATBOT_DA3_MODEL_CACHE_DIR"]) / safe_model
has_model = cache_dir.joinpath("config.json").exists() and any(
    path.is_file() and path.stat().st_size > 1_000_000
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
    for path in cache_dir.glob(pattern)
)
if has_model:
    print(f"da3 model cache ready: {{cache_dir}}")
elif os.environ.get("SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD", "true").lower() in {"1", "true", "yes", "on"}:
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model, local_dir=str(cache_dir), resume_download=True)
    print(f"da3 model cache ready: {{cache_dir}}")
else:
    raise SystemExit(f"DA3 model cache is missing at {{cache_dir}} and downloads are disabled")
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
if [ "$SPLATBOT_MAST3R_USE_GLOMAP" = "true" ]; then
  command -v "$SPLATBOT_GLOMAP_BIN"
  "$SPLATBOT_GLOMAP_BIN" -h >/tmp/glomap-help.txt
fi
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

du -sh "$VENV_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$U2NET_HOME" "$(dirname "$SPLATBOT_MAST3R_WEIGHTS")" "$SPLATBOT_DA3_MODEL_CACHE_DIR" || true
echo "runtime cache verified"
"""


def warm_volume(settings: Settings, args: argparse.Namespace) -> int:
    if not settings.runpod_network_volume_id:
        raise RunPodError("SPLATBOT_RUNPOD_NETWORK_VOLUME_ID is required to warm a persistent volume")

    log_path = args.log_path or settings.data_dir / "network-volume-warm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    client = RunPodClient(settings.runpod_api_key_value)
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

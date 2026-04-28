#!/usr/bin/env bash
set -euo pipefail

: "${SPLATBOT_JOB_ID:?}"
: "${SPLATBOT_SESSION_ID:?}"
: "${SPLATBOT_SCAN_MODE:?}"
: "${SPLATBOT_SCAN_PRESET:=balanced}"
: "${SPLATBOT_VPS_HOST:?}"
: "${SPLATBOT_VPS_USER:=root}"

export DEBIAN_FRONTEND=noninteractive
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/triton_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/workspace/cuda_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export U2NET_HOME="${U2NET_HOME:-/workspace/.u2net}"
export SPLATBOT_LOG_COMMAND_OUTPUT="${SPLATBOT_LOG_COMMAND_OUTPUT:-1}"
KNOWN_HOSTS="${SPLATBOT_VPS_KNOWN_HOSTS_FILE:-/root/.ssh/known_hosts}"
mkdir -p "$(dirname "$KNOWN_HOSTS")"
if [ ! -s "$KNOWN_HOSTS" ]; then
  if [ -n "${SPLATBOT_VPS_KNOWN_HOSTS:-}" ]; then
    printf '%s\n' "$SPLATBOT_VPS_KNOWN_HOSTS" > "$KNOWN_HOSTS"
  else
    ssh-keyscan -T 15 -H "$SPLATBOT_VPS_HOST" > "$KNOWN_HOSTS"
  fi
  chmod 600 "$KNOWN_HOSTS"
fi
SSH_OPTS="-i /root/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN_HOSTS -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
VENV_DIR="${SPLATBOT_RUNPOD_VENV:-/workspace/venv}"
CACHE_MARKER="${SPLATBOT_RUNPOD_RUNTIME_CACHE_MARKER:-/workspace/.splatbot-runtime-cache-version}"
VPS_APP_DIR="${SPLATBOT_VPS_APP_DIR:-/opt/splatbot/app}"
VPS_DATA_DIR="${SPLATBOT_VPS_DATA_DIR:-/var/lib/splatbot}"
WORKER_DATA_DIR="${SPLATBOT_WORKER_DATA_DIR:-/workspace/splatbot-data}"
RUNTIME_CACHE_ENABLED=false
case "$VENV_DIR" in
  /workspace/*)
    RUNTIME_CACHE_ENABLED=true
    ;;
esac
VPS_JOBCTL="cd $VPS_APP_DIR && /opt/splatbot/venv/bin/splatbot-jobctl"

fail_job() {
  rc="$?"
  failed_command="${BASH_COMMAND:-unknown}"
  error="RunPod worker failed before completion with exit code $rc while running: $failed_command"
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "$VPS_JOBCTL fail $SPLATBOT_JOB_ID --error $(printf %q "$error") --notify" || true
  exit "$rc"
}

trap fail_job ERR

heartbeat_loop() {
  while true; do
    ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
      "$VPS_JOBCTL heartbeat $SPLATBOT_JOB_ID" || true
    sleep 60
  done
}

heartbeat_loop &
HEARTBEAT_PID="$!"

cleanup_worker() {
  if [ -n "${HEARTBEAT_PID:-}" ]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
  fi
  rm -f /root/.ssh/id_ed25519
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

check_colmap_cuda() {
  local colmap_help
  colmap_help="$(colmap -h 2>&1)"
  printf '%s\n' "$colmap_help" | sed -n '1p'
  if truthy "${SPLATBOT_COLMAP_USE_GPU:-false}" && ! printf '%s\n' "$colmap_help" | grep -q "with CUDA"; then
    echo "SPLATBOT_COLMAP_USE_GPU=true requires a CUDA-enabled COLMAP build" >&2
    exit 2
  fi
}

check_rembg_cuda() {
  "$VENV_DIR/bin/python" - <<'PY'
import os

import torch  # Preload CUDA/cuDNN libraries before ONNX Runtime initializes.
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

providers = ort.get_available_providers()
device = ort.get_device()
print(f"  onnxruntime_device={device}")
print(f"  onnxruntime_providers={providers}")

require_gpu = os.getenv("SPLATBOT_REMBG_REQUIRE_GPU", "").lower() in {"1", "true", "yes", "on"}
if not require_gpu:
    raise SystemExit(0)

if "CUDAExecutionProvider" not in providers:
    raise SystemExit("SPLATBOT_REMBG_REQUIRE_GPU=true but ONNX Runtime has no CUDAExecutionProvider")

from rembg import new_session

session = new_session("u2net")
active_providers = session.inner_session.get_providers()
print(f"  rembg_session_providers={active_providers}")
if "CUDAExecutionProvider" not in active_providers:
    raise SystemExit(
        "SPLATBOT_REMBG_REQUIRE_GPU=true but rembg did not create a CUDAExecutionProvider session"
    )
PY
}

print_runtime_diagnostics() {
  echo "runtime diagnostics:"
  echo "  job=$SPLATBOT_JOB_ID mode=$SPLATBOT_SCAN_MODE preset=$SPLATBOT_SCAN_PRESET"
  echo "  venv=$VENV_DIR cache_enabled=$RUNTIME_CACHE_ENABLED cache_ready=$RUNTIME_CACHE_READY cache_marker=$CACHE_MARKER"
  echo "  python=$(command -v python || true)"
  echo "  colmap=$(command -v colmap || true)"
  colmap -h 2>&1 | sed -n '1p' || true
  ffmpeg -version 2>&1 | sed -n '1p' || true
  "$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch

for package in ("nerfstudio", "gsplat", "rembg", "onnxruntime", "onnxruntime-gpu", "torch", "numpy", "opencv-python", "dn-splatter"):
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "missing"
    print(f"  {package}={version}")
print(f"  torch_cuda_available={torch.cuda.is_available()}")
for module in ("sam2", "sam3", "dn_splatter"):
    try:
        __import__(module)
    except Exception as exc:
        print(f"  {module}=unavailable ({exc})")
    else:
        print(f"  {module}=available")
PY
}

trap cleanup_worker EXIT

ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
  "$VPS_JOBCTL set-status $SPLATBOT_JOB_ID preparing"

if [ -n "${SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND:-}" ]; then
  bash -lc "$SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND"
fi

RUNTIME_CACHE_READY=false
if [ "$RUNTIME_CACHE_ENABLED" = true ] && [ -n "${SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION:-}" ] && [ -r "$CACHE_MARKER" ]; then
  if [ "$(cat "$CACHE_MARKER")" = "$SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION" ]; then
    RUNTIME_CACHE_READY=true
    echo "RunPod runtime cache is ready: $SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION"
  fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  RUNTIME_CACHE_READY=false
fi

"$VENV_DIR/bin/pip" install --no-build-isolation --no-deps -e /workspace/splatbot-app
"$VENV_DIR/bin/python" - <<'PY' || "$VENV_DIR/bin/pip" install boto3
import boto3  # noqa: F401
PY
if [ -n "${SPLATBOT_RUNPOD_SETUP_COMMAND:-}" ] && [ "$RUNTIME_CACHE_READY" != true ]; then
  bash -lc "$SPLATBOT_RUNPOD_SETUP_COMMAND"
fi
export PATH="$VENV_DIR/bin:$PATH"

command -v ns-process-data >/dev/null
command -v ns-train >/dev/null
command -v ns-export >/dev/null
command -v ns-render >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
command -v colmap >/dev/null
command -v rembg >/dev/null
command -v splatbot-segment >/dev/null
command -v splatbot-pose >/dev/null
command -v splatbot-depth >/dev/null
command -v splatbot-train >/dev/null
command -v splatbot-mesh >/dev/null
command -v splatbot-da3 >/dev/null
for cmd in splatbot-segment splatbot-pose splatbot-depth splatbot-train splatbot-mesh splatbot-da3; do
  "$cmd" --help >/tmp/"$cmd"-help.txt
done
IFS=',' read -ra SEGMENT_BACKENDS <<< "${SPLATBOT_BEST_SEGMENTATION_BACKENDS:-${SPLATBOT_SEGMENTATION_BACKEND:-rembg}}"
IFS=',' read -ra REQUIRED_SEGMENT_BACKENDS <<< "${SPLATBOT_BEST_SEGMENTATION_REQUIRED_BACKENDS:-}"
if [ "${SPLATBOT_SCAN_PRESET:-balanced}" != "best" ] && [ -n "${SPLATBOT_SEGMENTATION_BACKEND:-}" ]; then
  IFS=',' read -ra SEGMENT_BACKENDS <<< "$SPLATBOT_SEGMENTATION_BACKEND"
fi
if [ "${SPLATBOT_SCAN_MODE:-scene}" = "object" ]; then
  for backend in "${SEGMENT_BACKENDS[@]}"; do
    backend="$(printf '%s' "$backend" | xargs)"
    if [ -n "$backend" ] && [ "$backend" != "rembg" ]; then
      if splatbot-segment --backend "$backend" --self-test; then
        continue
      fi
      required=false
      if [ "${SPLATBOT_SCAN_PRESET:-balanced}" = "best" ]; then
        for required_backend in "${REQUIRED_SEGMENT_BACKENDS[@]}"; do
          required_backend="$(printf '%s' "$required_backend" | xargs)"
          if [ "$backend" = "$required_backend" ]; then
            required=true
            break
          fi
        done
      fi
      if [ "$required" = true ]; then
        echo "best preset requires segmentation backend '$backend', but its self-test failed" >&2
        exit 2
      fi
      echo "optional segmentation backend '$backend' self-test failed; fallback remains available" >&2
    fi
  done
fi
command -v nvcc >/dev/null
check_colmap_cuda
check_rembg_cuda
"$VENV_DIR/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch CUDA is not available")
from gsplat.cuda import _backend

if _backend._C is None:
    raise SystemExit("gsplat CUDA extension is not available")
PY
if [ "${SPLATBOT_SCAN_PRESET:-balanced}" = "best" ]; then
  "$VENV_DIR/bin/python" - <<'PY'
import importlib.util

if importlib.util.find_spec("depth_anything_3") is None:
    raise SystemExit("best preset requires Depth Anything 3, but depth_anything_3 is not installed")
PY
  ns-train splatfacto-big --help | grep -q -- "--pipeline.model.strategy" || {
    echo "best preset requires Nerfstudio splatfacto MCMC strategy support" >&2
    exit 2
  }
fi
if [ "$RUNTIME_CACHE_ENABLED" = true ] && [ -n "${SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION:-}" ]; then
  mkdir -p "$(dirname "$CACHE_MARKER")"
  printf '%s\n' "$SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION" > "$CACHE_MARKER"
fi
print_runtime_diagnostics

rm -rf /workspace/input-media /workspace/results
mkdir -p /workspace/input-media /workspace/results
cat > /workspace/splatbot-set-status <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${SPLATBOT_VPS_USER:=root}"
: "${SPLATBOT_VPS_HOST:?}"
job_id="$1"
status="$2"
ssh -i /root/.ssh/id_ed25519 \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="${SPLATBOT_VPS_KNOWN_HOSTS_FILE:-/root/.ssh/known_hosts}" \
  -o LogLevel=ERROR \
  -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
  "cd $VPS_APP_DIR && /opt/splatbot/venv/bin/splatbot-jobctl set-status $job_id $status"
SH
chmod +x /workspace/splatbot-set-status
export SPLATBOT_STATUS_COMMAND=/workspace/splatbot-set-status

rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
  "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:$VPS_DATA_DIR/sessions/$SPLATBOT_SESSION_ID/" \
  /workspace/input-media/
find /workspace/input-media -maxdepth 1 -type f -printf 'input media: %f %s bytes\n' | sort

if "$VENV_DIR/bin/splatbot-run-job-dir" "$SPLATBOT_JOB_ID" "$SPLATBOT_SCAN_MODE" --preset "$SPLATBOT_SCAN_PRESET" /workspace/input-media /workspace/results; then
  if [ -f /workspace/results/cleaned_splat.ply ]; then
    "$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path

path = Path("/workspace/results/cleaned_splat.ply")
fmt = vertices = None
with path.open("rb") as handle:
    for raw_line in handle:
        line = raw_line.decode("ascii", errors="ignore").strip()
        if line.startswith("format "):
            fmt = line
        elif line.startswith("element vertex "):
            vertices = line.rsplit(" ", 1)[-1]
        elif line == "end_header":
            break
print(f"worker result ply: size={path.stat().st_size} format={fmt} vertices={vertices}")
PY
  fi
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "mkdir -p $VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/export $VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/renders"
  rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
    /workspace/results/cleaned_splat.ply \
    "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:$VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/export/cleaned_splat.ply"
  for mesh in /workspace/results/mesh.glb /workspace/results/mesh.gltf /workspace/results/mesh.obj; do
    if [ -f "$mesh" ]; then
      rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
        "$mesh" \
        "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:$VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/export/$(basename "$mesh")"
    fi
  done
  if [ -f /workspace/results/turntable.mp4 ]; then
    rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
      /workspace/results/turntable.mp4 \
      "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:$VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/renders/turntable.mp4"
  fi
  if [ -f /workspace/results/metrics.json ]; then
    rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
      /workspace/results/metrics.json \
      "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:$VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/metrics.json"
  fi
  for report in quality_report.json candidate_report.json; do
    if [ -f "/workspace/results/$report" ]; then
      rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
        "/workspace/results/$report" \
        "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:$VPS_DATA_DIR/jobs/$SPLATBOT_JOB_ID/$report"
    fi
  done
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "$VPS_JOBCTL complete $SPLATBOT_JOB_ID --notify"
else
  rc="$?"
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "$VPS_JOBCTL fail $SPLATBOT_JOB_ID --error 'RunPod worker failed with exit code $rc' --notify"
  exit "$rc"
fi

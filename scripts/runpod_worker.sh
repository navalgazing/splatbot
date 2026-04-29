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
VENV_DIR="${SPLATBOT_RUNPOD_VENV:-/workspace/venv}"
DEFAULT_TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
detect_cuda_arch_list() {
  local compute_cap
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  if printf '%s' "$compute_cap" | grep -Eq '^[0-9]+\.[0-9]+$'; then
    printf '%s\n' "$compute_cap"
    return 0
  fi
  local python_bin="$VENV_DIR/bin/python"
  if [ ! -x "$python_bin" ]; then
    python_bin=python3
  fi
  "$python_bin" - <<'PY' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
}
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] || [ "${TORCH_CUDA_ARCH_LIST:-}" = "8.9" ] || [ "${TORCH_CUDA_ARCH_LIST:-}" = "$DEFAULT_TORCH_CUDA_ARCH_LIST" ]; then
  detected_arch="$(detect_cuda_arch_list | head -n 1)"
  export TORCH_CUDA_ARCH_LIST="${detected_arch:-$DEFAULT_TORCH_CUDA_ARCH_LIST}"
else
  export TORCH_CUDA_ARCH_LIST
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/triton_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/workspace/cuda_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TORCH_HOME="${TORCH_HOME:-/opt/splatbot/models/torch}"
export U2NET_HOME="${U2NET_HOME:-/opt/splatbot/models/rembg}"
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
CACHE_MARKER="${SPLATBOT_RUNPOD_RUNTIME_CACHE_MARKER:-/workspace/.splatbot-runtime-cache-version}"
VPS_APP_DIR="${SPLATBOT_VPS_APP_DIR:-/opt/splatbot/app}"
VPS_DATA_DIR="${SPLATBOT_VPS_DATA_DIR:-/var/lib/splatbot}"
WORKER_DATA_DIR="${SPLATBOT_WORKER_DATA_DIR:-/workspace/splatbot-data}"
export VPS_APP_DIR VPS_DATA_DIR
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
      "$VPS_JOBCTL heartbeat $SPLATBOT_JOB_ID" >/dev/null 2>&1 || true
    sleep 60 &
    wait "$!" || true
  done
}

heartbeat_loop &
HEARTBEAT_PID="$!"

cleanup_worker() {
  if [ -n "${HEARTBEAT_PID:-}" ]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
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
from pathlib import Path

import torch  # Preload CUDA/cuDNN libraries before ONNX Runtime initializes.
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

providers = ort.get_available_providers()
device = ort.get_device()
print(f"  onnxruntime_device={device}")
print(f"  onnxruntime_providers={providers}")

u2net_model = Path(os.environ.get("U2NET_HOME", "/opt/splatbot/models/rembg")) / "u2net.onnx"
if not u2net_model.exists() or u2net_model.stat().st_size < 1_000_000:
    raise SystemExit(
        f"rembg u2net model is not baked into the worker image at {u2net_model}; "
        "refusing runtime download during production job"
    )
print(f"  rembg_u2net_model={u2net_model} bytes={u2net_model.stat().st_size}")

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

check_required_glomap() {
  if [ "${SPLATBOT_SCAN_PRESET:-balanced}" != "best" ]; then
    return 0
  fi
  if ! truthy "${SPLATBOT_MAST3R_USE_GLOMAP:-true}"; then
    return 0
  fi
  local glomap_cmd="${SPLATBOT_GLOMAP_BIN:-glomap}"
  if ! command -v "$glomap_cmd" >/dev/null; then
    echo "best preset requires MASt3R+GLOMAP, but '$glomap_cmd' is not installed or not in PATH" >&2
    exit 2
  fi
  "$glomap_cmd" -h >/tmp/glomap-help.txt 2>&1 || {
    echo "best preset requires MASt3R+GLOMAP, but '$glomap_cmd -h' failed" >&2
    tail -n 80 /tmp/glomap-help.txt >&2
    exit 2
  }
}

pose_backends_include_mast3r() {
  local configured="${SPLATBOT_BEST_POSE_BACKENDS:-${SPLATBOT_POSE_BACKENDS:-}}"
  if [ "${SPLATBOT_SCAN_PRESET:-balanced}" != "best" ]; then
    configured="${SPLATBOT_POSE_BACKENDS:-$configured}"
  fi
  IFS=',' read -ra POSE_BACKENDS_FOR_CHECK <<< "$configured"
  for backend in "${POSE_BACKENDS_FOR_CHECK[@]}"; do
    backend="$(printf '%s' "$backend" | xargs)"
    if [ "$backend" = "mast3r-sfm" ]; then
      return 0
    fi
  done
  return 1
}

pose_backends_include_vggt() {
  local configured="${SPLATBOT_BEST_POSE_BACKENDS:-${SPLATBOT_POSE_BACKENDS:-}}"
  if [ "${SPLATBOT_SCAN_PRESET:-balanced}" != "best" ]; then
    configured="${SPLATBOT_POSE_BACKENDS:-$configured}"
  fi
  IFS=',' read -ra POSE_BACKENDS_FOR_CHECK <<< "$configured"
  for backend in "${POSE_BACKENDS_FOR_CHECK[@]}"; do
    backend="$(printf '%s' "$backend" | xargs)"
    if [ "$backend" = "vggt-colmap" ]; then
      return 0
    fi
  done
  return 1
}

check_vggt_weights_config() {
  if ! pose_backends_include_vggt; then
    return 0
  fi
  local weights="${SPLATBOT_VGGT_WEIGHTS:-${TORCH_HOME:-/opt/splatbot/models/torch}/hub/checkpoints/model.pt}"
  if [ -s "$weights" ]; then
    local bytes
    bytes="$(stat -c%s "$weights")"
    if [ "$bytes" -gt 4000000000 ]; then
      echo "  vggt_weights=$weights bytes=$bytes"
    else
      echo "VGGT pose backend is configured but weights at $weights are too small: $bytes bytes" >&2
      exit 2
    fi
  elif truthy "${SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD:-false}"; then
    echo "  vggt_weights=$weights missing; VGGT may download facebook/VGGT-1B during pose estimation"
  else
    echo "VGGT pose backend is configured but weights are missing at $weights and downloads are disabled" >&2
    exit 2
  fi

  local tracker="${SPLATBOT_VGGSFM_TRACKER_WEIGHTS:-${TORCH_HOME:-/opt/splatbot/models/torch}/hub/checkpoints/vggsfm_v2_tracker.pt}"
  if [ -s "$tracker" ]; then
    local tracker_bytes
    tracker_bytes="$(stat -c%s "$tracker")"
    if [ "$tracker_bytes" -gt 100000000 ]; then
      echo "  vggsfm_tracker_weights=$tracker bytes=$tracker_bytes"
    else
      echo "VGGT pose backend is configured but VGGSfM tracker weights at $tracker are too small: $tracker_bytes bytes" >&2
      exit 2
    fi
  elif truthy "${SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD:-false}"; then
    echo "  vggsfm_tracker_weights=$tracker missing; VGGT may download facebook/VGGSfM during pose estimation"
  else
    echo "VGGT pose backend is configured but VGGSfM tracker weights are missing at $tracker and downloads are disabled" >&2
    exit 2
  fi

  local dinov2_repo="${SPLATBOT_DINOV2_HUB_REPO:-${TORCH_HOME:-/opt/splatbot/models/torch}/hub/facebookresearch_dinov2_main}"
  if [ -r "$dinov2_repo/hubconf.py" ] && [ -d "$dinov2_repo/dinov2" ]; then
    echo "  dinov2_hub_repo=$dinov2_repo"
  elif truthy "${SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD:-false}"; then
    echo "  dinov2_hub_repo=$dinov2_repo missing; VGGT may download facebookresearch/dinov2 during pose estimation"
  else
    echo "VGGT pose backend is configured but DINOv2 torch hub repo is missing at $dinov2_repo and downloads are disabled" >&2
    exit 2
  fi

  local dinov2_weights="${SPLATBOT_DINOV2_VITB14_REG_WEIGHTS:-${TORCH_HOME:-/opt/splatbot/models/torch}/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth}"
  if [ -s "$dinov2_weights" ]; then
    local dinov2_bytes
    dinov2_bytes="$(stat -c%s "$dinov2_weights")"
    if [ "$dinov2_bytes" -gt 300000000 ]; then
      echo "  dinov2_vitb14_reg_weights=$dinov2_weights bytes=$dinov2_bytes"
    else
      echo "VGGT pose backend is configured but DINOv2 ViT-B/14 reg weights at $dinov2_weights are too small: $dinov2_bytes bytes" >&2
      exit 2
    fi
  elif truthy "${SPLATBOT_VGGT_ALLOW_WEIGHT_DOWNLOAD:-false}"; then
    echo "  dinov2_vitb14_reg_weights=$dinov2_weights missing; VGGT may download DINOv2 weights during pose estimation"
  else
    echo "VGGT pose backend is configured but DINOv2 ViT-B/14 reg weights are missing at $dinov2_weights and downloads are disabled" >&2
    exit 2
  fi
}

check_mast3r_weights_config() {
  if ! pose_backends_include_mast3r; then
    return 0
  fi
  local weights="${SPLATBOT_MAST3R_WEIGHTS:-/opt/splatbot/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
  if [ -s "$weights" ]; then
    echo "  mast3r_weights=$weights bytes=$(stat -c%s "$weights")"
    return 0
  fi
  if truthy "${SPLATBOT_MAST3R_ALLOW_WEIGHT_DOWNLOAD:-true}"; then
    echo "  mast3r_weights=$weights missing; splatbot-mast3r will download the configured checkpoint if selected"
    return 0
  fi
  echo "MASt3R pose backend is configured but weights are missing at $weights and downloads are disabled" >&2
  exit 2
}

depth_backends_include_da3() {
  local configured="${SPLATBOT_BEST_DEPTH_BACKENDS:-${SPLATBOT_DEPTH_BACKENDS:-}}"
  if [ "${SPLATBOT_SCAN_PRESET:-balanced}" != "best" ]; then
    configured="${SPLATBOT_DEPTH_BACKENDS:-$configured}"
  fi
  IFS=',' read -ra DEPTH_BACKENDS_FOR_CHECK <<< "$configured"
  for backend in "${DEPTH_BACKENDS_FOR_CHECK[@]}"; do
    backend="$(printf '%s' "$backend" | xargs)"
    if [ "$backend" = "da3" ]; then
      return 0
    fi
  done
  return 1
}

check_da3_model_config() {
  if ! depth_backends_include_da3; then
    return 0
  fi
  local cache_root="${SPLATBOT_DA3_MODEL_CACHE_DIR:-/opt/splatbot/models/da3}"
  local model="${SPLATBOT_DA3_MODEL:-depth-anything/DA3-LARGE-1.1}"
  local safe_model
  safe_model="$(printf '%s' "$model" | sed -E 's/[^A-Za-z0-9._-]+/__/g')"
  local cached="$cache_root/$safe_model"
  if [ -f "$model" ] || [ -f "$cached/config.json" ]; then
    echo "  da3_model=$model cache=$cached"
    return 0
  fi
  if truthy "${SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD:-true}"; then
    echo "  da3_model=$model missing from $cached; splatbot-da3 will download/cache it if selected"
    return 0
  fi
  echo "DA3 depth backend is configured but model cache is missing at $cached and downloads are disabled" >&2
  exit 2
}

check_torchvision_weights_config() {
  local torch_home="${TORCH_HOME:-/opt/splatbot/models/torch}"
  local checkpoint="$torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth"
  if [ -s "$checkpoint" ]; then
    echo "  torch_home=$torch_home alexnet_checkpoint=$checkpoint bytes=$(stat -c%s "$checkpoint")"
    return 0
  fi
  if truthy "${SPLATBOT_TORCHVISION_ALLOW_WEIGHT_DOWNLOAD:-false}"; then
    echo "  alexnet_checkpoint=$checkpoint missing; LPIPS/torchvision may download it during training"
    return 0
  fi
  echo "LPIPS AlexNet checkpoint is missing at $checkpoint and downloads are disabled" >&2
  exit 2
}

print_runtime_diagnostics() {
  echo "runtime diagnostics:"
  echo "  job=$SPLATBOT_JOB_ID mode=$SPLATBOT_SCAN_MODE preset=$SPLATBOT_SCAN_PRESET"
  echo "  venv=$VENV_DIR cache_enabled=$RUNTIME_CACHE_ENABLED cache_ready=$RUNTIME_CACHE_READY cache_marker=$CACHE_MARKER"
  echo "  python=$(command -v python || true)"
  echo "  colmap=$(command -v colmap || true)"
  echo "  glomap=$(command -v "${SPLATBOT_GLOMAP_BIN:-glomap}" || true)"
  echo "  torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-}"
  colmap -h 2>&1 | sed -n '1p' || true
  "${SPLATBOT_GLOMAP_BIN:-glomap}" -h 2>&1 | sed -n '1p' || true
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
for module in ("sam2", "sam3", "dn_splatter", "vggt", "mast3r", "dust3r", "lightglue"):
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
command -v splatbot-colmap-pose-adapter >/dev/null
command -v splatbot-vggt >/dev/null
command -v splatbot-mast3r >/dev/null
command -v splatbot-depth >/dev/null
command -v splatbot-train >/dev/null
command -v splatbot-mesh >/dev/null
command -v splatbot-da3 >/dev/null
for cmd in splatbot-segment splatbot-pose splatbot-colmap-pose-adapter splatbot-vggt splatbot-mast3r splatbot-depth splatbot-train splatbot-mesh splatbot-da3; do
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
check_required_glomap
check_vggt_weights_config
check_mast3r_weights_config
check_da3_model_config
check_torchvision_weights_config
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
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.models.splatfacto import SplatfactoModelConfig

if importlib.util.find_spec("depth_anything_3") is None:
    raise SystemExit("best preset requires Depth Anything 3, but depth_anything_3 is not installed")
if "splatfacto-mcmc" not in method_configs:
    raise SystemExit("best preset requires Nerfstudio splatfacto-mcmc, but it is not registered")
if getattr(SplatfactoModelConfig(), "strategy", None) != "default":
    raise SystemExit("best preset requires Nerfstudio Splatfacto MCMC strategy support")
PY
  ns-train --help >/tmp/ns-train-methods.txt 2>&1
  grep -q "splatfacto-mcmc" /tmp/ns-train-methods.txt || {
    echo "best preset requires Nerfstudio splatfacto-mcmc in ns-train method list" >&2
    tail -n 80 /tmp/ns-train-methods.txt >&2
    exit 2
  }
  ns-train splatfacto-mcmc --max-num-iterations 1 --steps-per-save 1 --viewer.quit-on-train-completion True --help >/tmp/ns-train-splatfacto-mcmc-help.txt 2>&1 || {
    echo "best preset requires Nerfstudio splatfacto-mcmc options used by Splatbot" >&2
    tail -n 80 /tmp/ns-train-splatfacto-mcmc-help.txt >&2
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

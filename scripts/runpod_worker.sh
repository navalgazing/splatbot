#!/usr/bin/env bash
set -euo pipefail

: "${SPLATBOT_JOB_ID:?}"
: "${SPLATBOT_SESSION_ID:?}"
: "${SPLATBOT_SCAN_MODE:?}"
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
export U2NET_HOME="${U2NET_HOME:-/workspace/.u2net}"
SSH_OPTS="-i /root/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
VENV_DIR="${SPLATBOT_RUNPOD_VENV:-/workspace/venv}"
CACHE_MARKER="${SPLATBOT_RUNPOD_RUNTIME_CACHE_MARKER:-/workspace/.splatbot-runtime-cache-version}"
VPS_JOBCTL="cd /opt/splatbot/app && /opt/splatbot/venv/bin/splatbot-jobctl"

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

trap cleanup_worker EXIT

ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
  "$VPS_JOBCTL set-status $SPLATBOT_JOB_ID preparing"

if [ -n "${SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND:-}" ]; then
  bash -lc "$SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND"
fi

RUNTIME_CACHE_READY=false
if [ -n "${SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION:-}" ] && [ -r "$CACHE_MARKER" ]; then
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

"$VENV_DIR/bin/pip" install --no-deps -e /workspace/splatbot-app
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
command -v rembg >/dev/null
command -v nvcc >/dev/null
"$VENV_DIR/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch CUDA is not available")
from gsplat.cuda import _backend

if _backend._C is None:
    raise SystemExit("gsplat CUDA extension is not available")
PY
if [ -n "${SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION:-}" ]; then
  mkdir -p "$(dirname "$CACHE_MARKER")"
  printf '%s\n' "$SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION" > "$CACHE_MARKER"
fi

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
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR \
  -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
  "cd /opt/splatbot/app && /opt/splatbot/venv/bin/splatbot-jobctl set-status $job_id $status"
SH
chmod +x /workspace/splatbot-set-status
export SPLATBOT_STATUS_COMMAND=/workspace/splatbot-set-status

rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
  "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:/var/lib/splatbot/sessions/$SPLATBOT_SESSION_ID/" \
  /workspace/input-media/

if "$VENV_DIR/bin/splatbot-run-job-dir" "$SPLATBOT_JOB_ID" "$SPLATBOT_SCAN_MODE" /workspace/input-media /workspace/results; then
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "mkdir -p /var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/export /var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/renders"
  rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
    /workspace/results/cleaned_splat.ply \
    "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:/var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/export/cleaned_splat.ply"
  if [ -f /workspace/results/turntable.mp4 ]; then
    rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
      /workspace/results/turntable.mp4 \
      "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:/var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/renders/turntable.mp4"
  fi
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "$VPS_JOBCTL complete $SPLATBOT_JOB_ID --notify"
else
  rc="$?"
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "$VPS_JOBCTL fail $SPLATBOT_JOB_ID --error 'RunPod worker failed with exit code $rc' --notify"
  exit "$rc"
fi

if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${SPLATBOT_RUNPOD_API_KEY:-}" ]; then
  curl -fsS --request DELETE \
    --header "Authorization: Bearer $SPLATBOT_RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" || true
fi

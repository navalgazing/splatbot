#!/usr/bin/env bash
set -euo pipefail

: "${SPLATBOT_JOB_ID:?}"
: "${SPLATBOT_SESSION_ID:?}"
: "${SPLATBOT_SCAN_MODE:?}"
: "${SPLATBOT_VPS_HOST:?}"
: "${SPLATBOT_VPS_USER:=root}"

export DEBIAN_FRONTEND=noninteractive
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
SSH_OPTS="-i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
VENV_DIR="${SPLATBOT_RUNPOD_VENV:-/workspace/venv}"

fail_job() {
  rc="$?"
  failed_command="${BASH_COMMAND:-unknown}"
  error="RunPod worker failed before completion with exit code $rc while running: $failed_command"
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "/opt/splatbot/venv/bin/splatbot-jobctl fail $SPLATBOT_JOB_ID --error $(printf %q "$error") --notify" || true
  exit "$rc"
}

trap fail_job ERR

ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
  "/opt/splatbot/venv/bin/splatbot-jobctl set-status $SPLATBOT_JOB_ID preparing"

if [ -n "${SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND:-}" ]; then
  bash -lc "$SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
fi

"$VENV_DIR/bin/pip" install -e /workspace/splatbot-app
"$VENV_DIR/bin/pip" install boto3
if [ -n "${SPLATBOT_RUNPOD_SETUP_COMMAND:-}" ]; then
  bash -lc "$SPLATBOT_RUNPOD_SETUP_COMMAND"
fi
export PATH="$VENV_DIR/bin:$PATH"

command -v ns-process-data >/dev/null
command -v ns-train >/dev/null
command -v ns-export >/dev/null
command -v ns-render >/dev/null

rm -rf /workspace/input-media /workspace/results
mkdir -p /workspace/input-media /workspace/results
rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
  "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:/var/lib/splatbot/sessions/$SPLATBOT_SESSION_ID/" \
  /workspace/input-media/

ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
  "/opt/splatbot/venv/bin/splatbot-jobctl set-status $SPLATBOT_JOB_ID colmap"

set +e
"$VENV_DIR/bin/splatbot-run-job-dir" "$SPLATBOT_JOB_ID" "$SPLATBOT_SCAN_MODE" /workspace/input-media /workspace/results
rc="$?"
set -e
trap - ERR

if [ "$rc" -eq 0 ]; then
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "mkdir -p /var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/export /var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/renders"
  rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
    /workspace/results/cleaned_splat.ply \
    "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:/var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/export/cleaned_splat.ply"
  rsync -r --no-perms --no-owner --no-group --omit-dir-times -e "ssh $SSH_OPTS" \
    /workspace/results/turntable.mp4 \
    "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST:/var/lib/splatbot/jobs/$SPLATBOT_JOB_ID/renders/turntable.mp4"
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "/opt/splatbot/venv/bin/splatbot-jobctl complete $SPLATBOT_JOB_ID --notify"
else
  ssh $SSH_OPTS "$SPLATBOT_VPS_USER@$SPLATBOT_VPS_HOST" \
    "/opt/splatbot/venv/bin/splatbot-jobctl fail $SPLATBOT_JOB_ID --error 'RunPod worker failed with exit code $rc' --notify"
  exit "$rc"
fi

if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${SPLATBOT_RUNPOD_API_KEY:-}" ]; then
  curl -fsS --request DELETE \
    --header "Authorization: Bearer $SPLATBOT_RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" || true
fi

#!/usr/bin/env bash
set -euo pipefail

: "${SPLATBOT_JOB_ID:?}"
: "${SPLATBOT_SESSION_ID:?}"
: "${SPLATBOT_SCAN_MODE:?}"
: "${SPLATBOT_VPS_HOST:?}"
: "${SPLATBOT_VPS_USER:=root}"

export DEBIAN_FRONTEND=noninteractive
SSH_OPTS="-i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

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

apt-get update
apt-get install -y openssh-client rsync curl ffmpeg colmap python3 python3-venv python3-pip build-essential

python3 -m venv /workspace/venv
/workspace/venv/bin/pip install --upgrade pip
/workspace/venv/bin/pip install -e /workspace/splatbot-app
/workspace/venv/bin/pip install boto3
${SPLATBOT_RUNPOD_SETUP_COMMAND:-/workspace/venv/bin/pip install nerfstudio rembg}
export PATH="/workspace/venv/bin:$PATH"

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
/workspace/venv/bin/splatbot-run-job-dir "$SPLATBOT_JOB_ID" "$SPLATBOT_SCAN_MODE" /workspace/input-media /workspace/results
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

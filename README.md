# Splatbot

Private Telegram bot and worker for turning a photo set or short video into a Gaussian splat.

## Current MVP

- Telegram intake for one private user list.
- `/new`, `/mode scene|object`, `/preset fast|balanced|best`, media upload, `/submit`, `/status`, `/cancel`.
- SQLite queue shared by the bot and worker.
- Local worker/dispatcher that runs one GPU job at a time.
- RunPod backend for launching ephemeral GPU pods from the VPS.
- Preset-aware reconstruction pipeline with adaptive video frame quality scoring, pluggable segmentation/pose/train backends, object cleanup gates, and per-job metrics.
- Artifact persistence for the cleaned `.ply`, optional mesh, quality reports, and preview `.mp4`.
- Optional S3-compatible artifact upload with signed result URLs.
- Telegram completion/failure notifications when the dispatcher has a bot token.
- Browser result pages with full Gaussian splat viewing, optional mesh viewing, and orbit/pan/zoom point-cloud fallback.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Edit `.env` with at least:

```bash
SPLATBOT_TELEGRAM_TOKEN=...
SPLATBOT_ALLOWED_TELEGRAM_IDS=...
SPLATBOT_DATA_DIR=/var/lib/splatbot
SPLATBOT_DATABASE_PATH=/var/lib/splatbot/splatbot.sqlite3
SPLATBOT_WORKER_BACKEND=local
```

`SPLATBOT_WORKER_BACKEND` supports `local` and `runpod`.

On the GPU machine, make sure these commands are available or set their paths in `.env`:

- `ffmpeg`
- `ns-process-data`
- `ns-train`
- `ns-export`
- `ns-render`
- `rembg` for object mode

## Run Locally

Start the Telegram bot:

```bash
splatbot-api
```

Start the dispatcher on the machine that can run the GPU pipeline:

```bash
splatbot-dispatcher
```

The dispatcher watches SQLite for queued jobs, runs the pipeline, records artifacts, and sends Telegram notifications if `SPLATBOT_TELEGRAM_TOKEN` is set.

## Bot Flow

1. Send `/new`.
2. Choose Fast, Balanced, or Best, or send `/preset fast|balanced|best`.
3. Optional: send `/mode scene` or `/mode object`.
4. Upload either `SPLATBOT_MIN_IMAGES` to `SPLATBOT_MAX_IMAGES` photos, or one supported video.
5. Send `/submit`.
6. Use `/status` for collection progress, queued/running status, errors, and artifact locations.

Presets control the target video frame count and training budget:

- `fast`: fewer frames and iterations for cheaper previews.
- `balanced`: production default, adaptive frame selection, current quality baseline.
- `best`: more frames, higher iteration budget, and `splatfacto-big` for A/B quality trials.

For video uploads, adaptive selection extracts candidates up to
`SPLATBOT_MAX_VIDEO_CANDIDATE_FPS`, scores frames for blur, contrast,
over/underexposure, and duplicate content, drops frames below
`SPLATBOT_FRAME_QUALITY_REJECT_THRESHOLD`, then samples the best surviving frames
while preserving coverage through the clip. Object mode can fall back to solving
COLMAP poses on the original frames while training on `rembg` object frames when
background-removed frames are too unstable for registration. If COLMAP still
registers too few frames, the pipeline retries smaller evenly sampled subsets
from `SPLATBOT_COLMAP_RETRY_FRAME_COUNTS` with methods from
`SPLATBOT_COLMAP_RETRY_MATCHING_METHODS` before failing the job.

Object exports also run conservative silhouette cleanup. The exported Gaussian
centers are projected back into up to `SPLATBOT_SILHOUETTE_CLEANUP_MAX_VIEWS`
training masks, and points that repeatedly land outside the alpha silhouette are
culled before publishing. The pass is bounded by
`SPLATBOT_SILHOUETTE_CLEANUP_MAX_REMOVE_FRACTION` so bad masks or unusual camera
poses cannot delete too much of a result.

Maximum-quality runs can opt into newer reconstruction stages without changing
the Telegram flow:

```bash
SPLATBOT_SEGMENTATION_BACKEND=sam3,sam2,rembg
SPLATBOT_OBJECT_MASK_PROMPT='main object'
SPLATBOT_SAM3_MASK_COMMAND='splatbot-segment --backend sam3 --input {images_dir} --output {object_dir} --prompt {prompt}'
SPLATBOT_SAM2_MASK_COMMAND='splatbot-segment --backend sam2 --input {images_dir} --output {object_dir}'
SPLATBOT_SAM2_CHECKPOINT=/opt/splatbot/models/sam2.1_hiera_large.pt
SPLATBOT_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
SPLATBOT_POSE_BACKENDS=glomap,colmap
SPLATBOT_POSE_BACKEND_COMMAND='splatbot-pose --backend {backend} --input {images_dir} --output {processed_dir} --matching-method {matching_method}'
SPLATBOT_TRAIN_BACKENDS=dn-splatter,splatfacto-big
SPLATBOT_TRAIN_BACKEND_COMMAND='splatbot-train --backend {backend} --data {processed_dir} --output {ns_dir} --max-iterations {max_iterations} --steps-per-save {steps_per_save} {extra_args}'
SPLATBOT_MESH_EXPORT_ENABLED=true
SPLATBOT_MESH_BACKEND=o3dtsdf
SPLATBOT_MESH_EXPORT_COMMAND='splatbot-mesh --backend {backend} --ns-dir {ns_dir} --output {mesh_path}'
```

Backends are attempted in order. In `best` mode, the default object chain is
SAM3, SAM2, then rembg; pose is COLMAP global mapper when available, then normal
COLMAP; training is DN-Splatter-big when the runtime has a compatible
DN-Splatter/Nerfstudio stack, then `splatfacto-big`. SAM3 needs access to
Meta/Hugging Face checkpoints in the RunPod runtime, so deployments without that
token fall back to SAM2/rembg. The CUDA RunPod image intentionally excludes
DN-Splatter for now because current upstream DN-Splatter imports an older
`gsplat` API; the adapter remains available for a future pinned or patched
runtime.

Object-mode postprocessing now includes a mask-support cleanup pass and a
publish-time validation gate. If the cleaned splat still has too many points
with weak mask support, the job fails instead of publishing a misleading viewer.

## Artifact Storage

Without S3 settings, artifacts stay on local disk under:

```text
$SPLATBOT_DATA_DIR/jobs/<job_id>/
```

With S3-compatible settings present, the dispatcher uploads the cleaned `.ply` and preview `.mp4`, persists object keys, and sends signed URLs.

## Result Viewer

Set:

```bash
SPLATBOT_PUBLIC_BASE_URL=http://your-vps-or-domain
SPLATBOT_PUBLIC_RESULTS_DIR=/var/www/splatbot/results
```

Completed jobs publish:

- `index.html`
- `cleaned_splat.ply`
- `metrics.json`
- `quality_report.json`
- `candidate_report.json`
- `mesh.glb`, `mesh.gltf`, or `mesh.obj` when mesh export is enabled
- `turntable.mp4` when preview rendering is enabled

The viewer first loads the PLY with a browser Gaussian splat renderer. If that fails on the client, it falls back to a Three.js colored point preview with orbit, pan, and zoom controls. Mesh outputs can be opened from the same page when present. Telegram sends the viewer URL when available.

Compare saved jobs with:

```bash
splatbot-benchmark /var/lib/splatbot/jobs/<job_id> /var/lib/splatbot/jobs/<other_job_id>
```

Required S3 settings:

```bash
SPLATBOT_S3_ENDPOINT_URL=...
SPLATBOT_S3_REGION=auto
SPLATBOT_S3_BUCKET=...
SPLATBOT_S3_ACCESS_KEY_ID=...
SPLATBOT_S3_SECRET_ACCESS_KEY=...
```

## Tests

```bash
pytest -q
```

## Deployment Notes

For a simple local deployment, the bot and dispatcher can access the same SQLite database and uploaded media paths.

For RunPod, set `SPLATBOT_WORKER_BACKEND=runpod`. The dispatcher launches a GPU pod, the pod SSHes back to the VPS to pull session media, runs `splatbot-run-job-dir`, rsyncs artifacts back under `/var/lib/splatbot/jobs/<job_id>`, and calls `splatbot-jobctl` on the VPS to mark the job complete or failed.

The RunPod SSH keys configured with `SPLATBOT_RUNPOD_POD_SSH_KEY` and
`SPLATBOT_RUNPOD_VPS_SSH_KEY` must be readable by the `splatbot` service user
and should not be group/world-readable. On the VPS, use ownership like
`splatbot:splatbot` with mode `0600` for both private keys.

Remote RunPod workers call `splatbot-jobctl` over SSH from `/opt/splatbot/app`.
Keep `/opt/splatbot/app/.env` linked to `/etc/splatbot/splatbot.env` so those
commands load the same public URL, Telegram token, database path, and retention
settings as the systemd services.

### RunPod Runtime

The normal production path is the baked CUDA image:

```bash
SPLATBOT_RUNPOD_IMAGE_NAME=ghcr.io/navalgazing/splatbot-runpod:cuda-colmap
SPLATBOT_RUNPOD_VENV=/opt/splatbot/venv
SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND=
SPLATBOT_RUNPOD_SETUP_COMMAND=
SPLATBOT_COLMAP_BIN=splatbot-colmap-wrapper
SPLATBOT_COLMAP_USE_GPU=true
SPLATBOT_REMBG_REQUIRE_GPU=true
```

The image keeps `apt-get` and bulk Python installs out of each job and includes
a CUDA-enabled headless COLMAP build plus GPU ONNX Runtime for `rembg`. A RunPod
network volume is still useful for workspace caches and future large model
caches, but the core runtime should not depend on warming a new venv for every
pod.

The older generic-image path works only if you explicitly configure bootstrap and
setup commands. Prefer rebuilding the image instead of adding per-job installs.

Example:

```bash
docker build -f Dockerfile.runpod -t ghcr.io/navalgazing/splatbot-runpod:latest .
docker push ghcr.io/navalgazing/splatbot-runpod:latest
```

Then set:

```bash
SPLATBOT_RUNPOD_IMAGE_NAME=ghcr.io/navalgazing/splatbot-runpod:cuda-colmap
SPLATBOT_RUNPOD_NETWORK_VOLUME_ID=...
SPLATBOT_RUNPOD_DATA_CENTER_IDS=EU-RO-1
SPLATBOT_RUNPOD_VENV=/opt/splatbot/venv
SPLATBOT_RUNPOD_BOOTSTRAP_COMMAND=
SPLATBOT_RUNPOD_SETUP_COMMAND=
SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION=splatbot-runtime-2026-04-26-v1
SPLATBOT_COLMAP_USE_GPU=true
SPLATBOT_REMBG_REQUIRE_GPU=true
```

Bump `SPLATBOT_RUNPOD_RUNTIME_CACHE_VERSION` when changing cache-dependent runtime
behavior. Only enable `SPLATBOT_COLMAP_USE_GPU=true` after the image smoke check
or `scripts/warm_runpod_volume.py` confirms `colmap -h` reports a CUDA build.
Keep `SPLATBOT_REMBG_REQUIRE_GPU=true` with the production CUDA image so a broken
image fails loudly instead of silently falling back to slow CPU background
removal.

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
- `best`: more frames, quality-diverse frame selection, SOTA-capable fallback chains, and `splatfacto-big` as the stable export fallback.

For video uploads, adaptive selection extracts candidates up to
`SPLATBOT_MAX_VIDEO_CANDIDATE_FPS`, scores frames for blur, contrast,
over/underexposure, and duplicate content, drops frames below
`SPLATBOT_FRAME_QUALITY_REJECT_THRESHOLD`, then samples the best surviving frames
while preserving coverage through the clip; `best` mode also spreads selections
across distinct frame signatures so it does not waste budget on near-duplicate
views. When COLMAP is in the configured pose chain, object mode can fall back to
solving poses on the original frames while training on object frames when
background-removed frames are too unstable for registration. If COLMAP still
registers too few frames, the pipeline retries smaller evenly sampled subsets
from `SPLATBOT_COLMAP_RETRY_FRAME_COUNTS` with methods from
`SPLATBOT_COLMAP_RETRY_MATCHING_METHODS` before failing the job. The default
`best` pose chain starts with the dense COLMAP-global baseline, then tries
learned pose adapters when the baseline cannot pass quality gates.

Object exports also run conservative mask and silhouette cleanup. Best-mode
object scans default to `conservative` masking, which runs both the original
`rembg` cutout and SAM2 video propagation, then trains only on foreground pixels
where both masks agree. The combined masks keep the largest connected component,
close tiny holes, erode the edge halo, and reject frames with poor mask
agreement. The exported Gaussian centers are then projected back into up to
`SPLATBOT_SILHOUETTE_CLEANUP_MAX_VIEWS` training masks, and points that
repeatedly land outside the alpha silhouette are culled before publishing. The
pass is bounded by `SPLATBOT_SILHOUETTE_CLEANUP_MAX_REMOVE_FRACTION` so bad
masks or unusual camera poses cannot delete too much of a result.

Maximum-quality runs require the strongest non-gated stages to be installed and
configured. `best` does not silently degrade to legacy backends; if a required
stage is unavailable, the job fails with a Telegram summary naming the missing
stage. Approval-gated models are not part of the default chain.

```bash
SPLATBOT_BEST_SEGMENTATION_BACKENDS=conservative,rembg
SPLATBOT_BEST_SEGMENTATION_REQUIRED_BACKENDS=conservative
SPLATBOT_OBJECT_MASK_STRATEGY=
SPLATBOT_OBJECT_MASK_TRAINING_ALPHA_THRESHOLD=64
SPLATBOT_OBJECT_MASK_AGREEMENT_MIN_IOU=0.45
SPLATBOT_OBJECT_MASK_CLOSE_PX=1
SPLATBOT_OBJECT_MASK_ERODE_PX=1
SPLATBOT_OBJECT_MASK_PROMPT='main object'
SPLATBOT_SAM2_MASK_COMMAND='splatbot-segment --backend sam2 --input {images_dir} --output {object_dir}'
SPLATBOT_SAM2_CHECKPOINT=/opt/splatbot/models/sam2.1_hiera_large.pt
SPLATBOT_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
SPLATBOT_BEST_POSE_BACKENDS=colmap-global,vggt-colmap,mast3r-sfm
SPLATBOT_BEST_POSE_REQUIRED_BACKENDS=
SPLATBOT_POSE_BACKEND_COMMAND='splatbot-pose --backend {backend} --input {images_dir} --output {processed_dir} --matching-method {matching_method}'
SPLATBOT_DA3_MODEL=depth-anything/DA3-LARGE-1.1
SPLATBOT_DA3_USE_RAY_POSE=true
SPLATBOT_DA3_REF_VIEW_STRATEGY=middle
SPLATBOT_DA3_POSE_COMMAND='splatbot-da3 --images {images_dir} --processed {processed_dir}'
SPLATBOT_VGGT_POSE_COMMAND='splatbot-vggt --images {images_dir} --processed {processed_dir} --matching-method {matching_method}'
SPLATBOT_VGGT_ARGS='--use_ba --max_query_pts 2048 --query_frame_num 5'
SPLATBOT_MAST3R_POSE_COMMAND='splatbot-mast3r --images {images_dir} --processed {processed_dir} --matching-method {matching_method}'
SPLATBOT_MAST3R_USE_GLOMAP=true
SPLATBOT_MIN_EXPORT_GAUSSIAN_RETENTION=0.02
SPLATBOT_BEST_DEPTH_BACKENDS=da3,depth-anything-v2-large
SPLATBOT_BEST_DEPTH_REQUIRED_BACKENDS=da3
SPLATBOT_DEPTH_BACKEND_COMMAND='splatbot-depth --backend {backend} --processed {processed_dir} --images {images_dir}'
SPLATBOT_DA3_DEPTH_COMMAND='splatbot-da3 --images {images_dir} --processed {processed_dir} --depth-only'
SPLATBOT_BEST_TRAIN_BACKENDS=splatfacto-big,3dgs-mcmc
SPLATBOT_BEST_TRAIN_REQUIRED_BACKENDS=
SPLATBOT_TRAIN_BACKEND_COMMAND='splatbot-train --backend {backend} --data {processed_dir} --output {ns_dir} --max-iterations {max_iterations} --steps-per-save {steps_per_save} {extra_args}'
SPLATBOT_MCMC_TRAIN_COMMAND='ns-train splatfacto-mcmc --data {processed_dir} --output-dir {ns_dir} --max-num-iterations {max_iterations} --steps-per-save {steps_per_save} --viewer.quit-on-train-completion True {extra_args}'
SPLATBOT_MESH_EXPORT_ENABLED=true
SPLATBOT_MESH_BACKEND=o3dtsdf
SPLATBOT_MESH_EXPORT_COMMAND='splatbot-mesh --backend {backend} --ns-dir {ns_dir} --output {mesh_path}'
```

Backends are attempted in order for non-required stages. In `best` mode, SAM2
segmentation and DA3 depth are required by default; train backends are retried
when the exported splat fails hard quality gates. VGGT and MASt3R remain
available learned pose adapters, but production success is gated on output
quality rather than backend labels. SAM3 is disabled by default because it
requires access approval. DN-Splatter remains explicitly opt-in because its
dependency stack can conflict with Nerfstudio/gsplat.

Every skipped, failed, or recovered backend attempt is written into
`quality_report.json` and summarized in the Telegram completion/failure message.

### Optional VGGT and MASt3R Pose

`splatbot-vggt` and `splatbot-mast3r` are thin production adapters around
externally installed upstream repos. They stage Splatbot's selected images, run
the configured upstream command, require a parseable COLMAP sparse model, copy it
to `processed/colmap/sparse/0`, and call Nerfstudio's `colmap_to_json` to create
`transforms.json`. Transform-only output is not accepted.

VGGT upstream supports COLMAP export through `demo_colmap.py`; the official demo
expects images under `{scene_dir}/images` and writes sparse COLMAP files under
`{scene_dir}/sparse`. Production/commercial use should not use the original
`facebook/VGGT-1B` checkpoint, which is non-commercial. The commercial checkpoint
`facebook/VGGT-1B-Commercial` requires Hugging Face access approval and an
authenticated token. If you use the official demo directly, audit or patch it so
it loads the approved commercial checkpoint instead of the hard-coded original
checkpoint.

Example VGGT install shape for a private RunPod image or setup command:

```bash
git clone https://github.com/facebookresearch/vggt.git /opt/vggt
/opt/splatbot/venv/bin/pip install --no-deps -e /opt/vggt
/opt/splatbot/venv/bin/pip install \
  'numpy<2' Pillow huggingface_hub einops safetensors opencv-python scipy \
  trimesh matplotlib pycolmap==3.10.0 pyceres==2.3 \
  'git+https://github.com/jytime/LightGlue.git#egg=lightglue'
export HF_TOKEN=...
export SPLATBOT_VGGT_POSE_COMMAND='splatbot-vggt --images {images_dir} --processed {processed_dir} --matching-method {matching_method}'
export SPLATBOT_VGGT_RUN_COMMAND='/opt/splatbot/venv/bin/python /opt/vggt/demo_colmap.py --scene_dir {scene_dir} --use_ba --max_query_pts 2048 --query_frame_num 5'
export SPLATBOT_VGGT_MAX_IMAGES=64
```

MASt3R's repository and published checkpoint are CC BY-NC-SA / non-commercial
with additional training-dataset notices. Do not bake or enable MASt3R in a
commercial production image unless that use is covered by your license review or
you have separate permission. If allowed, install the repo recursively plus its
Kapture dependencies, then let the adapter generate the pairs file and normalize
the `kapture_mast3r_mapping.py` reconstruction:

```bash
git clone --recursive https://github.com/naver/mast3r /opt/mast3r
/opt/splatbot/venv/bin/pip install -r /opt/mast3r/requirements.txt -r /opt/mast3r/dust3r/requirements.txt
/opt/splatbot/venv/bin/pip install kapture kapture-localization cython
/opt/splatbot/venv/bin/python - <<'PY'
from pathlib import Path
import site

pth = Path(site.getsitepackages()[0]) / "splatbot-mast3r.pth"
pth.write_text("/opt/mast3r\n/opt/mast3r/dust3r\n", encoding="utf-8")
PY
export SPLATBOT_MAST3R_POSE_COMMAND='splatbot-mast3r --images {images_dir} --processed {processed_dir} --matching-method {matching_method}'
export SPLATBOT_MAST3R_MAX_IMAGES=120
export SPLATBOT_MAST3R_PAIR_WINDOW=5
export SPLATBOT_MAST3R_PAIR_CYCLIC=true
export SPLATBOT_MAST3R_USE_GLOMAP=true
```

For either adapter, run a small non-artifact smoke dataset before making the
backend required. A valid smoke run must leave `transforms.json` and
`colmap/sparse/0/{cameras.bin,images.bin,points3D.bin}` under the processed
directory, and the quality gate must report enough registered images. The
default VGGT and MASt3R image caps are chosen for a 24 GB 4090-class worker.
VGGT retries with fewer staged images if the first cap fails, down to
`SPLATBOT_VGGT_MIN_IMAGES` (default 24). Set caps to `0` only after confirming
GPU memory on a smoke run.

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

`SPLATBOT_RUNPOD_GPU_TYPE_ID` accepts a comma-separated priority list. For best
mode, keep H200/H100/A100 first and include 48 GB fallbacks such as L40S, L40,
RTX 6000 Ada, RTX A6000, and A40. A 24 GB 4090 is cheaper when available but can
OOM the official VGGT COLMAP demo, so it should not be in the default best-mode
pool.

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
SPLATBOT_RUNPOD_GPU_TYPE_ID='NVIDIA H200,NVIDIA H200 NVL,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe,NVIDIA H100 NVL,NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40'
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

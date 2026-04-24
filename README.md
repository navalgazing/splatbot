# Splatbot

Private Telegram bot and worker for turning a photo set or short video into a Gaussian splat.

## Current MVP

- Telegram intake for one private user list.
- `/new`, `/mode scene|object`, media upload, `/submit`, `/status`, `/cancel`.
- SQLite queue shared by the bot and worker.
- Local worker/dispatcher that runs one GPU job at a time.
- RunPod backend for launching ephemeral GPU pods from the VPS.
- Nerfstudio `splatfacto` pipeline with optional object-background removal through `rembg`.
- Artifact persistence for the cleaned `.ply` and preview `.mp4`.
- Optional S3-compatible artifact upload with signed result URLs.
- Telegram completion/failure notifications when the dispatcher has a bot token.
- Browser result pages with orbit/pan/zoom point-cloud viewing.

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
2. Optional: send `/mode scene` or `/mode object`.
3. Upload either `SPLATBOT_MIN_IMAGES` to `SPLATBOT_MAX_IMAGES` photos, or one supported video.
4. Send `/submit`.
5. Use `/status` for collection progress, queued/running status, errors, and artifact locations.

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
- `turntable.mp4`

The viewer loads the PLY in a Three.js point-cloud scene with orbit, pan, and zoom controls. Telegram sends the viewer URL when available.

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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from splatbot.config import ScanMode, ScanPreset, Settings
from splatbot.job_overrides import job_overrides_path
from splatbot.media import classify_path
from splatbot.models import JobStatus, MediaKind, ScanJob, utcnow
from splatbot.runpod_backend import RunPodClient, RunPodLauncher
from splatbot.storage import Store


BASELINE_MATRIX_SETTINGS: dict[str, Any] = {
    "best_pose_backends": "colmap-global",
    "best_pose_required_backends": "colmap-global",
    "best_depth_backends": "da3",
    "best_depth_required_backends": "da3",
    "best_train_backends": "splatfacto-big",
    "best_train_required_backends": "splatfacto-big",
}

POSE_ROW_DESCRIPTIONS = {
    "colmap-global": "COLMAP/GLOMAP global pose with baseline depth, train, and postprocess.",
    "vggt-colmap": "VGGT pose adapter with baseline depth, train, and postprocess.",
    "mast3r-sfm": "MASt3R SfM pose adapter with baseline depth, train, and postprocess.",
    "da3-colmap": "Depth Anything 3 pose adapter with baseline depth, train, and postprocess.",
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _should_match_reference_owner() -> bool:
    return os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0


def match_reference_owner(path: Path, reference: Path) -> None:
    if not _should_match_reference_owner() or not path.exists():
        return
    reference_stat = reference.stat()
    targets = [path]
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            root_path = Path(root)
            targets.extend(root_path / name for name in dirs)
            targets.extend(root_path / name for name in files)
    for target in targets:
        os.chown(target, reference_stat.st_uid, reference_stat.st_gid)


def pose_matrix_settings(pose_backend: str) -> tuple[str, str, dict[str, Any]]:
    row = "baseline" if pose_backend == "colmap-global" else f"pose-{pose_backend}"
    changed = {}
    if pose_backend != "colmap-global":
        changed = {
            "best_pose_backends": pose_backend,
            "best_pose_required_backends": pose_backend,
        }
    return row, POSE_ROW_DESCRIPTIONS[pose_backend], {**BASELINE_MATRIX_SETTINGS, **changed}


def apply_settings_overrides(settings: Settings, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise ValueError(f"unknown settings override: {key}")
        setattr(settings, key, value)


async def create_preparing_job(
    settings: Settings,
    store: Store,
    source: Path,
    telegram_user_id: int,
    mode: ScanMode,
    preset: ScanPreset,
    *,
    matrix_run_id: str | None,
    matrix_row: str | None,
    matrix_description: str | None,
    matrix_changed: dict[str, Any],
    settings_overrides: dict[str, Any],
) -> ScanJob:
    session = await store.create_session(telegram_user_id, mode, preset)
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / source.name
    shutil.copy2(source, dest)
    match_reference_owner(session_dir, settings.data_dir)
    kind = classify_path(dest)
    if kind not in {MediaKind.PHOTO, MediaKind.VIDEO}:
        raise ValueError(f"unsupported source media: {source}")
    await store.add_media(session.id, kind, dest)

    job_id = uuid.uuid4().hex
    job_dir = settings.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    if settings_overrides:
        write_json(
            job_overrides_path(settings, job_id),
            {
                "matrix_run": {
                    "id": matrix_run_id,
                    "row": matrix_row,
                    "description": matrix_description,
                    "source": str(source),
                    "changed_settings": matrix_changed,
                    "baseline_settings": BASELINE_MATRIX_SETTINGS,
                    "submitted_at": utcnow().isoformat(),
                },
                "settings": settings_overrides,
            },
        )
    match_reference_owner(job_dir, settings.data_dir)

    now = utcnow().isoformat()
    async with store._connect() as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (JobStatus.QUEUED.value, now, session.id),
        )
        await db.execute(
            """
            INSERT INTO jobs (id, session_id, telegram_user_id, mode, preset, status, error, created_at, updated_at,
                              claimed_at, heartbeat_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                job_id,
                session.id,
                telegram_user_id,
                mode.value,
                preset.value,
                JobStatus.PREPARING.value,
                now,
                now,
                now,
                now,
            ),
        )
        await db.commit()
    job = await store.get_job(job_id)
    assert job is not None
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one controlled RunPod job from a local media file.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--image")
    parser.add_argument("--telegram-user-id", required=True, type=int)
    parser.add_argument("--mode", choices=[mode.value for mode in ScanMode], default=ScanMode.OBJECT.value)
    parser.add_argument("--preset", choices=[preset.value for preset in ScanPreset], default=ScanPreset.BALANCED.value)
    parser.add_argument("--pose-backend", choices=sorted(POSE_ROW_DESCRIPTIONS))
    parser.add_argument("--matrix-run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path("/etc/splatbot/splatbot.env"))
    parser.add_argument("--colmap-use-gpu", action="store_true")
    parser.add_argument("--colmap-bin", default="colmap")
    args = parser.parse_args()

    load_env_file(args.env_file)
    settings = Settings()
    if args.image:
        settings.runpod_image_name = args.image
    settings.runpod_venv = "/opt/splatbot/venv"
    settings.runpod_bootstrap_command = ""
    settings.runpod_setup_command = ""
    settings.colmap_use_gpu = args.colmap_use_gpu
    settings.colmap_bin = args.colmap_bin

    matrix_run_id = args.matrix_run_id
    matrix_row = None
    matrix_description = None
    matrix_changed: dict[str, Any] = {}
    settings_overrides: dict[str, Any] = {}
    if args.pose_backend:
        if matrix_run_id is None:
            matrix_run_id = datetime.now(UTC).strftime("controlled-pose-%Y%m%dT%H%M%SZ")
        matrix_row, matrix_description, settings_overrides = pose_matrix_settings(args.pose_backend)
        matrix_changed = {
            key: value
            for key, value in settings_overrides.items()
            if BASELINE_MATRIX_SETTINGS.get(key) != value
        }
        apply_settings_overrides(settings, settings_overrides)

    store = Store(settings.database_path)
    asyncio.run(store.init())
    job = asyncio.run(
        create_preparing_job(
            settings,
            store,
            args.source,
            args.telegram_user_id,
            ScanMode(args.mode),
            ScanPreset(args.preset),
            matrix_run_id=matrix_run_id,
            matrix_row=matrix_row,
            matrix_description=matrix_description,
            matrix_changed=matrix_changed,
            settings_overrides=settings_overrides,
        )
    )
    viewer = settings.public_job_url(job.id)
    print(f"created controlled job {job.id} session={job.session_id} viewer={viewer}", flush=True)

    def record_pod_id(pod_id: str | None) -> None:
        asyncio.run(store.set_job_runpod_pod_id(job.id, pod_id))

    report: dict[str, Any] = {
        "job_id": job.id,
        "session_id": job.session_id,
        "viewer_url": viewer,
        "matrix_run_id": matrix_run_id,
        "matrix_row": matrix_row,
        "pose_backend": args.pose_backend,
        "started_at": utcnow().isoformat(),
    }
    launcher = RunPodLauncher(settings, client=RunPodClient(settings.runpod_api_key_value))
    try:
        pod = launcher.launch(job, record_pod_id)
        report["pod_id"] = pod.id
        completed = asyncio.run(store.get_job(job.id))
        report["status"] = completed.status.value if completed else "missing"
        report["error"] = completed.error if completed else "job missing"
        print(f"completed controlled job {job.id} on pod {pod.id}", flush=True)
    except Exception as exc:
        asyncio.run(store.set_job_failed_unless_terminal(job.id, str(exc)))
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["finished_at"] = utcnow().isoformat()
        if args.output:
            write_json(args.output, report)
            match_reference_owner(args.output.parent, settings.data_dir)


if __name__ == "__main__":
    main()

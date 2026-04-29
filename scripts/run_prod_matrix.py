#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from splatbot.config import ScanMode, ScanPreset, Settings
from splatbot.job_overrides import job_overrides_path
from splatbot.media import classify_path
from splatbot.models import JobStatus, MediaKind, ScanJob, utcnow
from splatbot.storage import Store


BASELINE_SETTINGS: dict[str, Any] = {
    "best_pose_backends": "colmap-global",
    "best_pose_required_backends": "colmap-global",
    "best_depth_backends": "da3",
    "best_depth_required_backends": "da3",
    "best_train_backends": "splatfacto-big",
    "best_train_required_backends": "splatfacto-big",
}


@dataclass(frozen=True)
class MatrixRow:
    name: str
    description: str
    changed: dict[str, Any]
    required_command_field: str | None = None

    def settings_overrides(self) -> dict[str, Any]:
        return {**BASELINE_SETTINGS, **self.changed}


MATRIX_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow("baseline", "Locked current production baseline.", {}),
    MatrixRow(
        "pose-vggt-colmap",
        "VGGT pose adapter with baseline depth, train, and postprocess.",
        {
            "best_pose_backends": "vggt-colmap",
            "best_pose_required_backends": "vggt-colmap",
        },
    ),
    MatrixRow(
        "pose-mast3r-sfm",
        "MASt3R SfM pose adapter with baseline depth, train, and postprocess.",
        {
            "best_pose_backends": "mast3r-sfm",
            "best_pose_required_backends": "mast3r-sfm",
        },
    ),
    MatrixRow(
        "pose-da3-colmap",
        "Depth Anything 3 pose adapter with baseline depth, train, and postprocess.",
        {
            "best_pose_backends": "da3-colmap",
            "best_pose_required_backends": "da3-colmap",
        },
    ),
    MatrixRow(
        "train-3dgs-mcmc",
        "3DGS MCMC training with baseline pose, depth, and postprocess.",
        {
            "best_train_backends": "3dgs-mcmc",
            "best_train_required_backends": "3dgs-mcmc",
        },
        required_command_field="mcmc_train_command",
    ),
    MatrixRow(
        "postprocess-mask-support-strict",
        "Stricter mask-support pruning with baseline pose, depth, and train.",
        {
            "mask_support_cleanup_min_inside_views": 2,
            "mask_support_cleanup_min_inside_ratio": 0.05,
            "mask_support_cleanup_max_remove_fraction": 0.6,
            "postprocess_validation_max_low_support_fraction": 0.15,
        },
    ),
    MatrixRow(
        "postprocess-opacity-scale-strict",
        "Stricter opacity and anisotropy pruning with baseline pose, depth, and train.",
        {
            "gaussian_cleanup_min_opacity": -6.0,
            "gaussian_cleanup_max_scale_ratio": 8.0,
            "gaussian_cleanup_max_anisotropy": 15.0,
            "gaussian_cleanup_max_remove_fraction": 0.5,
        },
    ),
    MatrixRow(
        "postprocess-spatial-strict",
        "Stricter radius and connected-component pruning with baseline pose, depth, and train.",
        {
            "spatial_outlier_radius_fraction": 0.03,
            "spatial_outlier_min_neighbors": 4,
            "spatial_component_min_fraction": 0.01,
            "spatial_cleanup_max_remove_fraction": 0.5,
        },
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def available_rows(settings: Settings, requested: list[str] | None, include_unconfigured: bool) -> tuple[list[MatrixRow], list[dict]]:
    selected = list(MATRIX_ROWS)
    if requested:
        by_name = {row.name: row for row in selected}
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise SystemExit(f"unknown matrix row(s): {', '.join(missing)}")
        selected = [by_name[name] for name in requested]
    skipped: list[dict] = []
    rows: list[MatrixRow] = []
    for row in selected:
        if row.required_command_field and not str(getattr(settings, row.required_command_field)).strip():
            if include_unconfigured:
                rows.append(row)
            else:
                skipped.append(
                    {
                        "name": row.name,
                        "reason": f"{row.required_command_field} is not configured",
                    }
                )
            continue
        rows.append(row)
    return rows, skipped


async def submit_job(
    settings: Settings,
    store: Store,
    source: Path,
    telegram_user_id: int,
    mode: ScanMode,
    preset: ScanPreset,
    matrix_run_id: str,
    row: MatrixRow,
) -> ScanJob:
    session = await store.create_session(telegram_user_id, mode, preset)
    session_dir = settings.data_dir / "sessions" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / source.name
    shutil.copy2(source, dest)
    kind = classify_path(dest)
    if kind not in {MediaKind.PHOTO, MediaKind.VIDEO}:
        raise ValueError(f"unsupported source media: {source}")
    await store.add_media(session.id, kind, dest)

    job_id = uuid.uuid4().hex
    job_dir = settings.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    overrides = row.settings_overrides()
    write_json(
        job_overrides_path(settings, job_id),
        {
            "matrix_run": {
                "id": matrix_run_id,
                "row": row.name,
                "description": row.description,
                "source": str(source),
                "changed_settings": row.changed,
                "baseline_settings": BASELINE_SETTINGS,
                "submitted_at": utcnow().isoformat(),
            },
            "settings": overrides,
        },
    )

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
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL)
            """,
            (
                job_id,
                session.id,
                telegram_user_id,
                mode.value,
                preset.value,
                JobStatus.QUEUED.value,
                now,
                now,
            ),
        )
        await db.commit()
    job = await store.get_job(job_id)
    assert job is not None
    return job


async def wait_for_terminal(store: Store, job_id: str, poll_seconds: int) -> ScanJob:
    last_status = None
    while True:
        job = await store.get_job(job_id)
        if job is None:
            raise RuntimeError(f"job disappeared: {job_id}")
        if job.status != last_status:
            print(f"{datetime.now(UTC).isoformat()} job={job.id} status={job.status.value}", flush=True)
            last_status = job.status
        if job.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}:
            return job
        await asyncio.sleep(poll_seconds)


def summarize_job(settings: Settings, job: ScanJob, row: MatrixRow) -> dict[str, Any]:
    job_dir = settings.job_dir(job.id)
    metrics = load_json(job_dir / "metrics.json")
    quality = load_json(job_dir / "quality_report.json")
    cleaned = metrics.get("ply", {}).get("cleaned", {})
    export_metrics = metrics.get("ply", {}).get("export", {})
    return {
        "row": row.name,
        "job_id": job.id,
        "session_id": job.session_id,
        "status": job.status.value,
        "error": job.error,
        "viewer_url": settings.public_job_url(job.id),
        "pose_backend": metrics.get("pose_backend"),
        "depth_backend": metrics.get("depth_backend"),
        "train_backend": metrics.get("train_backend"),
        "vertices": cleaned.get("vertices"),
        "export_retention": export_metrics.get("retention_ratio"),
        "issues": quality.get("issues"),
        "warnings": quality.get("warnings"),
        "stages": metrics.get("stages"),
        "pipeline_events": metrics.get("pipeline_events"),
    }


async def run_matrix(args: argparse.Namespace) -> None:
    settings = Settings()
    store = Store(settings.database_path)
    await store.init()
    source = args.source
    if not source.exists():
        raise SystemExit(f"source does not exist: {source}")
    rows, skipped = available_rows(settings, args.rows, args.include_unconfigured)
    run_id = args.run_id or datetime.now(UTC).strftime("matrix-%Y%m%dT%H%M%SZ")
    report_path = args.output or (settings.data_dir / "matrix_runs" / f"{run_id}.json")
    report: dict[str, Any] = {
        "id": run_id,
        "source": str(source),
        "mode": args.mode,
        "preset": args.preset,
        "started_at": utcnow().isoformat(),
        "skipped": skipped,
        "jobs": [],
    }
    write_json(report_path, report)
    print(f"matrix report: {report_path}", flush=True)
    for row in rows:
        print(f"submitting row={row.name}", flush=True)
        job = await submit_job(
            settings,
            store,
            source,
            args.telegram_user_id,
            ScanMode(args.mode),
            ScanPreset(args.preset),
            run_id,
            row,
        )
        print(f"submitted row={row.name} job={job.id} viewer={settings.public_job_url(job.id)}", flush=True)
        terminal = await wait_for_terminal(store, job.id, args.poll_seconds)
        summary = summarize_job(settings, terminal, row)
        report["jobs"].append(summary)
        write_json(report_path, report)
        print(
            f"completed row={row.name} status={terminal.status.value} job={terminal.id} "
            f"viewer={summary['viewer_url']}",
            flush=True,
        )
        if terminal.status != JobStatus.DONE and not args.continue_on_failure:
            break
    report["finished_at"] = utcnow().isoformat()
    write_json(report_path, report)
    print(f"matrix finished: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a production-style Splatbot method matrix via queued jobs.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--telegram-user-id", required=True, type=int)
    parser.add_argument("--mode", choices=[mode.value for mode in ScanMode], default=ScanMode.OBJECT.value)
    parser.add_argument("--preset", choices=[preset.value for preset in ScanPreset], default=ScanPreset.BEST.value)
    parser.add_argument("--run-id")
    parser.add_argument("--rows", nargs="+")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-unconfigured", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    main()

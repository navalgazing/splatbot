from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .models import ScanJob


JOB_OVERRIDES_FILENAME = "job_overrides.json"


def job_overrides_path(settings: Settings, job_id: str) -> Path:
    return settings.job_dir(job_id) / JOB_OVERRIDES_FILENAME


def load_job_override_payload(settings: Settings, job: ScanJob) -> dict[str, Any]:
    path = job_overrides_path(settings, job.id)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{JOB_OVERRIDES_FILENAME} must contain a JSON object")
    return payload


def settings_for_job(base: Settings, job: ScanJob) -> Settings:
    payload = load_job_override_payload(base, job)
    raw_overrides = payload.get("settings") or {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("job settings overrides must be a JSON object")
    overrides = dict(raw_overrides)
    matrix_run = payload.get("matrix_run")
    if matrix_run is not None:
        overrides.setdefault("matrix_run_metadata", json.dumps(matrix_run, sort_keys=True))
    if not overrides:
        return base
    unknown = sorted(set(overrides) - set(Settings.model_fields))
    if unknown:
        raise ValueError(f"unknown job settings override(s): {', '.join(unknown)}")
    values = base.model_dump()
    values.update(overrides)
    return Settings.model_validate(values)

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .config import ScanMode


class JobStatus(StrEnum):
    COLLECTING = "collecting"
    QUEUED = "queued"
    PREPARING = "preparing"
    COLMAP = "colmap"
    TRAINING = "training"
    EXPORTING = "exporting"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MediaKind(StrEnum):
    PHOTO = "photo"
    VIDEO = "video"


class ArtifactKind(StrEnum):
    PLY = "ply"
    PREVIEW = "preview"


@dataclass(frozen=True)
class UploadSession:
    id: str
    telegram_user_id: int
    mode: ScanMode
    status: JobStatus
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MediaItem:
    id: str
    session_id: str
    kind: MediaKind
    local_path: str
    remote_key: str | None
    created_at: datetime


@dataclass(frozen=True)
class ScanJob:
    id: str
    session_id: str
    telegram_user_id: int
    mode: ScanMode
    status: JobStatus
    error: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class JobArtifact:
    id: str
    job_id: str
    kind: ArtifactKind
    local_path: str
    remote_key: str | None
    url: str | None
    created_at: datetime


def utcnow() -> datetime:
    return datetime.now(UTC)

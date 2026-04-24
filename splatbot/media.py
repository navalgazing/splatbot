from __future__ import annotations

from pathlib import Path

from .models import MediaItem, MediaKind


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


class MediaValidationError(ValueError):
    pass


def classify_path(path: Path) -> MediaKind:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return MediaKind.PHOTO
    if suffix in SUPPORTED_VIDEO_SUFFIXES:
        return MediaKind.VIDEO
    raise MediaValidationError(f"unsupported media type: {path.name}")


def validate_submission(items: list[MediaItem], min_images: int, max_images: int) -> None:
    photos = [item for item in items if item.kind == MediaKind.PHOTO]
    videos = [item for item in items if item.kind == MediaKind.VIDEO]
    if photos and videos:
        raise MediaValidationError("submit either photos or one video, not both")
    if videos and len(videos) != 1:
        raise MediaValidationError("submit exactly one video")
    if photos and len(photos) < min_images:
        raise MediaValidationError(f"need at least {min_images} photos")
    if photos and len(photos) > max_images:
        raise MediaValidationError(f"max {max_images} photos per job")
    if not photos and not videos:
        raise MediaValidationError("no media uploaded")


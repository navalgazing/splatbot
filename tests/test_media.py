from datetime import UTC, datetime

import pytest

from splatbot.media import MediaValidationError, validate_submission
from splatbot.models import MediaItem, MediaKind


def item(kind: MediaKind, name: str) -> MediaItem:
    return MediaItem(
        id=name,
        session_id="s",
        kind=kind,
        local_path=name,
        remote_key=None,
        created_at=datetime.now(UTC),
    )


def test_validate_submission_accepts_one_video() -> None:
    validate_submission([item(MediaKind.VIDEO, "scan.mp4")], min_images=100, max_images=300)


def test_validate_submission_rejects_too_few_photos() -> None:
    with pytest.raises(MediaValidationError, match="need at least 3 photos"):
        validate_submission([item(MediaKind.PHOTO, "a.jpg")], min_images=3, max_images=300)


def test_validate_submission_rejects_mixed_media() -> None:
    with pytest.raises(MediaValidationError, match="not both"):
        validate_submission(
            [item(MediaKind.PHOTO, "a.jpg"), item(MediaKind.VIDEO, "b.mp4")],
            min_images=1,
            max_images=300,
        )


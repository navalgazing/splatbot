from splatbot.config import ScanMode, Settings
from splatbot.telegram_app import _help_text, _upload_hint


def test_help_text_lists_flow_and_commands() -> None:
    text = _help_text(Settings(min_images=20, max_images=80, max_video_seconds=45, max_video_frames=90))

    assert "Choose object or scene" in text
    assert "Press Submit scan" in text
    assert "Photos: 20-80 images" in text
    assert "Video: up to 45s sampled to 90 frames" in text
    assert "/submit" in text


def test_upload_hint_makes_submit_action_explicit() -> None:
    text = _upload_hint(1, ScanMode.OBJECT, Settings(min_images=20, max_images=80))

    assert "Received 1 file(s)" in text
    assert "object scan" in text
    assert "press Submit scan" in text

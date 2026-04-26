from splatbot.config import ScanMode, Settings
from splatbot.telegram_app import _help_text, _main_keyboard, _upload_hint


def test_help_text_lists_flow_and_commands() -> None:
    text = _help_text(Settings(min_images=20, max_images=80, max_video_seconds=45, max_video_frames=90))

    assert "Choose object or scene" in text
    assert "Press Submit scan" in text
    assert "Photos: 20-80 images" in text
    assert "Video: up to 45s sampled to 90 frames" in text
    assert "/help" in text
    assert "/submit" in text


def test_upload_hint_makes_submit_action_explicit() -> None:
    text = _upload_hint(1, ScanMode.OBJECT, Settings(min_images=20, max_images=80))

    assert "Received 1 file(s)" in text
    assert "object scan" in text
    assert "press Submit scan" in text


def test_main_keyboard_starts_with_scan_mode_choices() -> None:
    keyboard = _main_keyboard()

    assert keyboard.inline_keyboard[0][0].text == "New object scan"
    assert keyboard.inline_keyboard[0][0].callback_data == "new:object"
    assert keyboard.inline_keyboard[1][0].text == "New scene scan"
    assert keyboard.inline_keyboard[1][0].callback_data == "new:scene"

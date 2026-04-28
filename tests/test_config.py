import pytest
from pydantic import ValidationError

from splatbot.config import ScanMode, Settings


def test_allowed_telegram_ids_parse_from_csv() -> None:
    settings = Settings(
        telegram_token="x",
        allowed_telegram_ids="123, 456",
        default_scan_mode=ScanMode.SCENE,
    )

    assert settings.allowed_telegram_ids == {123, 456}


def test_allowed_telegram_ids_parse_from_single_int() -> None:
    settings = Settings(
        telegram_token="x",
        allowed_telegram_ids=123,
        default_scan_mode=ScanMode.SCENE,
    )

    assert settings.allowed_telegram_ids == {123}


def test_telegram_access_is_private_by_default() -> None:
    settings = Settings(telegram_token="tok_abc123")

    assert settings.allowed_telegram_ids == set()
    assert settings.allow_all_telegram_users is False
    assert settings.telegram_token_value == "tok_abc123"
    assert "tok_abc123" not in repr(settings)


def test_job_dir_is_under_data_dir(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.job_dir("abc") == tmp_path / "jobs" / "abc"


def test_unsupported_worker_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(worker_backend="ssh")


def test_allowed_telegram_ids_rejects_non_integer_values() -> None:
    with pytest.raises(ValidationError, match="integer chat IDs"):
        Settings(allowed_telegram_ids="123, nope")

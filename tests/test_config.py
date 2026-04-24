from splatbot.config import ScanMode, Settings


def test_allowed_telegram_ids_parse_from_csv() -> None:
    settings = Settings(
        telegram_token="x",
        allowed_telegram_ids="123, 456",
        default_scan_mode=ScanMode.SCENE,
    )

    assert settings.allowed_telegram_ids == {123, 456}


def test_job_dir_is_under_data_dir(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.job_dir("abc") == tmp_path / "jobs" / "abc"


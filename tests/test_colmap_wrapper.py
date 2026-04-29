from pathlib import Path

from splatbot.colmap_wrapper import _mapped_argv, _mapped_command, _maybe_calibrate_global_mapper, promote_largest_sparse_model


def _write_model(path: Path, image_count: int, point_bytes: int) -> None:
    path.mkdir(parents=True)
    (path / "images.bin").write_bytes(f"images={image_count}".encode())
    (path / "points3D.bin").write_bytes(b"x" * point_bytes)


def test_promote_largest_sparse_model_replaces_sparse_zero(tmp_path) -> None:
    sparse = tmp_path / "sparse"
    _write_model(sparse / "0", image_count=2, point_bytes=10)
    _write_model(sparse / "1", image_count=140, point_bytes=100)

    promoted = promote_largest_sparse_model(
        sparse,
        image_count_reader=lambda path: int(path.read_text(encoding="utf-8").split("=")[1]),
    )

    assert promoted == sparse / "0"
    assert (sparse / "0" / "images.bin").read_text(encoding="utf-8") == "images=140"
    assert list(sparse.glob("0-splatbot-replaced-*"))


def test_promote_largest_sparse_model_keeps_sparse_zero_when_best(tmp_path) -> None:
    sparse = tmp_path / "sparse"
    _write_model(sparse / "0", image_count=140, point_bytes=100)
    _write_model(sparse / "1", image_count=2, point_bytes=10)

    promoted = promote_largest_sparse_model(
        sparse,
        image_count_reader=lambda path: int(path.read_text(encoding="utf-8").split("=")[1]),
    )

    assert promoted == sparse / "0"
    assert (sparse / "0" / "images.bin").read_text(encoding="utf-8") == "images=140"
    assert not list(sparse.glob("0-splatbot-replaced-*"))


def test_promote_largest_sparse_model_logs_reader_failure(tmp_path, capsys) -> None:
    sparse = tmp_path / "sparse"
    _write_model(sparse / "0", image_count=2, point_bytes=10)
    _write_model(sparse / "1", image_count=140, point_bytes=100)

    promoted = promote_largest_sparse_model(
        sparse,
        image_count_reader=lambda path: (_ for _ in ()).throw(RuntimeError("broken reader")),
    )

    assert promoted == sparse / "0"
    assert "failed to read registered image count" in capsys.readouterr().err


def test_mapped_argv_uses_global_mapper_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_COLMAP_MAPPER", "global")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: command == "global_mapper")

    assert _mapped_argv(["mapper", "--database_path", "db"]) == ["global_mapper", "--database_path", "db"]


def test_mapped_argv_falls_back_when_global_mapper_missing(monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_COLMAP_MAPPER", "global")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: False)
    monkeypatch.setattr("splatbot.colmap_wrapper._glomap_bin", lambda: None)

    assert _mapped_argv(["mapper", "--database_path", "db"]) == ["mapper", "--database_path", "db"]


def test_mapped_command_uses_glomap_when_colmap_global_mapper_missing(monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_COLMAP_MAPPER", "global")
    monkeypatch.setattr("splatbot.colmap_wrapper._real_colmap", lambda: "colmap-test")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: False)
    monkeypatch.setattr("splatbot.colmap_wrapper._glomap_bin", lambda: "glomap-test")

    command, argv = _mapped_command(
        [
            "mapper",
            "--database_path",
            "db",
            "--Mapper.ba_global_function_tolerance=1e-6",
            "--Mapper.ba_local_max_num_iterations",
            "25",
        ]
    )

    assert command == "glomap-test"
    assert argv == ["mapper", "--database_path", "db"]


def test_glomap_mapper_strips_colmap_mapper_only_options(monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_COLMAP_MAPPER", "global")
    monkeypatch.setattr("splatbot.colmap_wrapper._real_colmap", lambda: "colmap-test")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: False)
    monkeypatch.setattr("splatbot.colmap_wrapper._glomap_bin", lambda: "glomap-test")

    command, argv = _mapped_command(
        [
            "mapper",
            "--database_path",
            "db",
            "--Mapper.ba_global_function_tolerance",
            "0.000001",
            "--Mapper.multiple_models=0",
            "--image_path",
            "images",
        ]
    )

    assert command == "glomap-test"
    assert argv == ["mapper", "--database_path", "db", "--image_path", "images"]


def test_global_mapper_can_run_view_graph_calibrator(monkeypatch) -> None:
    calls = []

    class Result:
        returncode = 0

    def fake_run(argv, check=False, stdout=None, stderr=None):
        calls.append(argv)
        return Result()

    monkeypatch.setenv("SPLATBOT_COLMAP_GLOBAL_CALIBRATE", "true")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: command == "view_graph_calibrator")
    monkeypatch.setattr("splatbot.colmap_wrapper._real_colmap", lambda: "colmap-test")
    monkeypatch.setattr("splatbot.colmap_wrapper.subprocess.run", fake_run)

    _maybe_calibrate_global_mapper(["global_mapper", "--database_path", "db"])

    assert calls == [["colmap-test", "view_graph_calibrator", "--database_path", "db"]]

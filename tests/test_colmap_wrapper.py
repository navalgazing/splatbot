from pathlib import Path

from splatbot.colmap_wrapper import _mapped_argv, promote_largest_sparse_model


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


def test_mapped_argv_uses_global_mapper_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_COLMAP_MAPPER", "global")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: command == "global_mapper")

    assert _mapped_argv(["mapper", "--database_path", "db"]) == ["global_mapper", "--database_path", "db"]


def test_mapped_argv_falls_back_when_global_mapper_missing(monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_COLMAP_MAPPER", "global")
    monkeypatch.setattr("splatbot.colmap_wrapper._has_colmap_command", lambda command: False)

    assert _mapped_argv(["mapper", "--database_path", "db"]) == ["mapper", "--database_path", "db"]

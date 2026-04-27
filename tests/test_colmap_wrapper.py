from pathlib import Path

from splatbot.colmap_wrapper import promote_largest_sparse_model


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

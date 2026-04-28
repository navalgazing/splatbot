import struct
import sys
from pathlib import Path

import pytest

from splatbot import pose_adapters


def write_fake_sparse(model_dir: Path, registered: int = 3, points: int = 1) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.bin").write_bytes(b"camera")
    (model_dir / "images.bin").write_text(f"images={registered}", encoding="utf-8")
    (model_dir / "points3D.bin").write_bytes(struct.pack("<Q", points) + b"point-data")


def write_images(images_dir: Path, count: int = 3) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (images_dir / f"frame_{idx + 1:05d}.jpg").write_bytes(b"image")


def fake_generate_transforms(processed_dir: Path, sparse_model_dir: Path) -> int:
    registered = pose_adapters.read_registered_image_count(sparse_model_dir / "images.bin") or 0
    processed_dir.joinpath("transforms.json").write_text(
        '{"frames": [' + ",".join("{}" for _ in range(registered)) + "]}\n",
        encoding="utf-8",
    )
    return registered


def test_finalize_colmap_pose_dataset_requires_sparse_and_writes_nerfstudio_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    processed = tmp_path / "processed"
    sparse = tmp_path / "source-sparse"
    write_images(images)
    write_fake_sparse(sparse)
    monkeypatch.setattr(pose_adapters, "generate_transforms", fake_generate_transforms)

    pose_adapters.finalize_colmap_pose_dataset(images, processed, sparse)

    assert (processed / "images" / "frame_00001.jpg").exists()
    assert (processed / "colmap" / "sparse" / "0" / "images.bin").read_text(encoding="utf-8") == "images=3"
    assert (processed / "transforms.json").exists()


def test_finalize_colmap_pose_dataset_rejects_transform_only_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    sparse = tmp_path / "source-sparse"
    write_images(images)
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_text("images=3", encoding="utf-8")
    monkeypatch.setattr(pose_adapters, "generate_transforms", fake_generate_transforms)

    with pytest.raises(SystemExit, match="missing cameras/images/points3D"):
        pose_adapters.finalize_colmap_pose_dataset(images, tmp_path / "processed", sparse)


def test_stage_images_even_samples_to_configured_cap(tmp_path: Path) -> None:
    images = tmp_path / "images"
    staged = tmp_path / "staged"
    write_images(images, count=5)

    selected = pose_adapters.stage_images(images, staged, max_images=3)

    assert [path.name for path in selected] == ["frame_00001.jpg", "frame_00003.jpg", "frame_00005.jpg"]
    assert sorted(path.name for path in staged.iterdir()) == ["frame_00001.jpg", "frame_00003.jpg", "frame_00005.jpg"]


def test_vggt_adapter_runs_configured_command_and_normalizes_sparse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    images = tmp_path / "images"
    processed = tmp_path / "processed"
    write_images(images)

    def fake_run(argv: list[str]) -> None:
        calls.append(argv)
        scene_dir = Path(argv[argv.index("--scene") + 1])
        write_fake_sparse(scene_dir / "sparse")

    monkeypatch.setattr(pose_adapters, "run", fake_run)
    monkeypatch.setattr(pose_adapters, "generate_transforms", fake_generate_transforms)
    monkeypatch.setenv("SPLATBOT_VGGT_RUN_COMMAND", "fake-vggt --scene {scene_dir}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [
                "splatbot-vggt",
                "--images",
                str(images),
                "--processed",
                str(processed),
                "--work-dir",
                str(tmp_path / "work"),
            ],
        )
        pose_adapters.vggt_main()

    assert calls[0][0] == "fake-vggt"
    assert (processed / "colmap" / "sparse" / "0" / "points3D.bin").exists()
    assert (processed / "transforms.json").exists()


def test_retry_image_caps_halves_to_minimum() -> None:
    assert pose_adapters.retry_image_caps(64, 24) == [64, 32, 24]
    assert pose_adapters.retry_image_caps(20, 24) == [20]
    assert pose_adapters.retry_image_caps(0, 24) == [0]


def test_mast3r_adapter_writes_pairs_and_normalizes_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    pairs_text: list[str] = []
    images = tmp_path / "images"
    processed = tmp_path / "processed"
    write_images(images, count=4)

    def fake_run(argv: list[str]) -> None:
        calls.append(argv)
        pairs_path = Path(argv[argv.index("--pairs") + 1])
        pairs_text.append(pairs_path.read_text(encoding="utf-8"))
        output_dir = Path(argv[argv.index("--out") + 1])
        write_fake_sparse(output_dir / "reconstruction" / "0", registered=4)

    monkeypatch.setattr(pose_adapters, "run", fake_run)
    monkeypatch.setattr(pose_adapters, "generate_transforms", fake_generate_transforms)
    monkeypatch.setenv("SPLATBOT_MAST3R_RUN_COMMAND", "fake-mast3r --pairs {pairs_file} --out {mast3r_output_dir}")
    monkeypatch.setenv("SPLATBOT_MAST3R_PAIR_WINDOW", "1")
    monkeypatch.setenv("SPLATBOT_MAST3R_PAIR_CYCLIC", "false")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [
                "splatbot-mast3r",
                "--images",
                str(images),
                "--processed",
                str(processed),
                "--matching-method",
                "sequential",
            ],
        )
        pose_adapters.mast3r_main()

    assert calls[0][0] == "fake-mast3r"
    assert pairs_text == [
        "# kapture format: 1.1\n"
        "# query_image, map_image, score\n"
        "frame_00001.jpg, frame_00002.jpg, 1.0\n"
        "frame_00002.jpg, frame_00003.jpg, 1.0\n"
        "frame_00003.jpg, frame_00004.jpg, 1.0\n"
    ]
    assert (processed / "colmap" / "sparse" / "0" / "images.bin").read_text(encoding="utf-8") == "images=4"


def test_mast3r_default_command_uses_glomap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPLATBOT_MAST3R_USE_GLOMAP", raising=False)

    argv = pose_adapters.render_argv_template(
        pose_adapters.mast3r_default_command(
            tmp_path / "mast3r",
            tmp_path / "out",
            tmp_path / "pairs.txt",
            tmp_path / "images",
        ),
        {},
    )

    assert "--use_glomap_mapper" in argv

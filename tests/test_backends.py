from types import ModuleType

import pytest

from splatbot import backends


def test_pose_backend_sets_global_mapper_env(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(argv, env=None):
        calls.append((argv, env))

    monkeypatch.setattr(backends, "run", fake_run)
    monkeypatch.setenv("SPLATBOT_COLMAP_USE_GPU", "true")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "sys.argv",
            [
                "splatbot-pose",
                "--backend",
                "colmap-global",
                "--input",
                str(tmp_path / "images"),
                "--output",
                str(tmp_path / "processed"),
                "--matching-method",
                "sequential",
            ],
        )
        backends.pose_main()

    argv, env = calls[0]
    assert env["SPLATBOT_COLMAP_MAPPER"] == "global"
    assert "--no-gpu" not in argv
    assert argv[:2] == ["ns-process-data", "images"]
    assert "splatbot-colmap-wrapper" in argv


def test_train_backend_requires_dn_splatter_for_dn_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(__import__("sys").modules, "dn_splatter", None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "sys.argv",
            [
                "splatbot-train",
                "--backend",
                "dn-splatter-big",
                "--data",
                str(tmp_path / "processed"),
                "--output",
                str(tmp_path / "ns"),
                "--max-iterations",
                "10",
            ],
        )
        with pytest.raises(SystemExit, match="dn-splatter"):
            backends.train_main()


def test_train_backend_runs_dn_splatter_depth_only(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(argv, env=None):
        calls.append(argv)

    monkeypatch.setattr(backends, "run", fake_run)
    monkeypatch.setattr(backends, "prepare_dn_splatter_depths", lambda data_dir: None)
    monkeypatch.setitem(__import__("sys").modules, "dn_splatter", ModuleType("dn_splatter"))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "sys.argv",
            [
                "splatbot-train",
                "--backend",
                "dn-splatter-big",
                "--data",
                str(tmp_path / "processed"),
                "--output",
                str(tmp_path / "ns"),
                "--max-iterations",
                "10",
            ],
        )
        backends.train_main()

    argv = calls[0]
    assert "--pipeline.datamanager.dataparser.load-normals" in argv
    assert argv[argv.index("--pipeline.datamanager.dataparser.load-normals") + 1] == "False"
    assert "--pipeline.model.use-depth-loss" in argv
    assert argv[argv.index("--pipeline.model.use-depth-loss") + 1] == "True"
    assert "--pipeline.model.use-normal-loss" in argv
    assert argv[argv.index("--pipeline.model.use-normal-loss") + 1] == "False"


def test_mesh_backend_requires_gs_mesh(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(backends.shutil, "which", lambda _: None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "sys.argv",
            [
                "splatbot-mesh",
                "--backend",
                "o3dtsdf",
                "--ns-dir",
                str(tmp_path / "ns"),
                "--output",
                str(tmp_path / "mesh.glb"),
            ],
        )
        with pytest.raises(SystemExit, match="gs-mesh"):
            backends.mesh_main()


def test_segment_rembg_delegates_to_rembg_command(monkeypatch, tmp_path) -> None:
    calls = []
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "object"
    input_dir.mkdir()

    def fake_run(argv, env=None):
        calls.append(argv)

    monkeypatch.setattr(backends, "run", fake_run)
    monkeypatch.setenv("SPLATBOT_REMBG_BIN", "rembg-test")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "sys.argv",
            [
                "splatbot-segment",
                "--backend",
                "rembg",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
            ],
        )
        backends.segment_main()

    assert calls == [["rembg-test", "p", str(input_dir), str(output_dir)]]


def test_sam2_bootstrap_sampling_covers_video_span(tmp_path) -> None:
    items = []
    for idx in range(10):
        path = tmp_path / f"frame_{idx:05d}.jpg"
        path.write_bytes(b"")
        items.append(path)

    sampled = backends.indexed_even_sample(items, 4)

    assert [idx for idx, _ in sampled] == [0, 3, 6, 9]


def test_bbox_score_prefers_centered_valid_mask() -> None:
    centered = backends.bbox_score((30, 20, 70, 80), 100, 100)
    edge_touching = backends.bbox_score((0, 0, 90, 90), 100, 100)

    assert centered > edge_touching

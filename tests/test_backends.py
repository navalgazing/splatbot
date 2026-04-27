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

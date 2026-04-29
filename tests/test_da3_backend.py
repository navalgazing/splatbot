import json
import struct
from types import SimpleNamespace

import pytest

from splatbot import da3_backend
from splatbot.da3_backend import (
    Da3BackendError,
    opencv_world_to_camera_to_nerfstudio_c2w,
    required_prediction_attr,
    rotation_matrix_to_qvec,
    scale_intrinsics_to_image,
)


def test_required_prediction_attr_raises_runtime_error() -> None:
    with pytest.raises(Da3BackendError, match="required field"):
        required_prediction_attr(object(), "extrinsics", "exts")


def test_opencv_world_to_camera_rejects_invalid_shape() -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(Da3BackendError, match="unsupported DA3 extrinsic shape"):
        opencv_world_to_camera_to_nerfstudio_c2w(np.zeros((2, 2)))


def test_ensure_da3_model_uses_existing_cache(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    model_dir = cache / "depth-anything__DA3-LARGE-1.1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(da3_backend, "MIN_MODEL_FILE_BYTES", 1)
    monkeypatch.setenv("SPLATBOT_DA3_MODEL_CACHE_DIR", str(cache))

    assert da3_backend.ensure_da3_model("depth-anything/DA3-LARGE-1.1") == str(model_dir)


def test_ensure_da3_model_downloads_to_cache(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr(da3_backend, "MIN_MODEL_FILE_BYTES", 1)
    monkeypatch.setenv("SPLATBOT_DA3_MODEL_CACHE_DIR", str(cache))
    monkeypatch.setenv("SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD", "true")

    def fake_download(model_name, cache_dir):
        assert model_name == "depth-anything/DA3-LARGE-1.1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text("{}", encoding="utf-8")
        (cache_dir / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(da3_backend, "download_da3_model", fake_download)

    model_path = da3_backend.ensure_da3_model("depth-anything/DA3-LARGE-1.1")

    assert model_path == str(cache / "depth-anything__DA3-LARGE-1.1")


def test_ensure_da3_model_fails_when_download_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPLATBOT_DA3_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD", "false")

    with pytest.raises(Da3BackendError, match="pre-warm"):
        da3_backend.ensure_da3_model("depth-anything/DA3-LARGE-1.1")


def test_scale_intrinsics_to_image_uses_depth_resolution() -> None:
    np = pytest.importorskip("numpy")

    k = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics_to_image(k, width=200, height=100, depth_shape=(10, 20))

    assert scaled[0, 0] == 100.0
    assert scaled[1, 1] == 100.0
    assert scaled[0, 2] == 100.0
    assert scaled[1, 2] == 50.0


def test_rotation_matrix_to_qvec_identity() -> None:
    np = pytest.importorskip("numpy")

    assert rotation_matrix_to_qvec(np.eye(3)) == [1.0, 0.0, 0.0, 0.0]


def test_write_nerfstudio_dataset_writes_da3_sparse_seed(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")

    images = []
    for idx, color in enumerate([(255, 0, 0, 255), (0, 255, 0, 255)], start=1):
        path = tmp_path / f"frame_{idx:05d}.png"
        Image.new("RGBA", (20, 10), color).save(path)
        images.append(path)
    prediction = SimpleNamespace(
        depth=np.ones((2, 5, 10), dtype=np.float32),
        conf=np.ones((2, 5, 10), dtype=np.float32),
        extrinsics=np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 2, axis=0),
        intrinsics=np.repeat(
            np.array([[[10.0, 0.0, 5.0], [0.0, 10.0, 2.5], [0.0, 0.0, 1.0]]], dtype=np.float64),
            2,
            axis=0,
        ),
    )
    monkeypatch.setenv("SPLATBOT_DA3_SPARSE_POINTS_PER_IMAGE", "8")
    monkeypatch.setenv("SPLATBOT_DA3_MIN_SPARSE_POINTS", "1")

    da3_backend.write_nerfstudio_dataset(images, prediction, tmp_path / "processed")

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    assert transforms["frames"][0]["fl_x"] == 20.0
    assert transforms["frames"][0]["fl_y"] == 20.0
    assert transforms["ply_file_path"] == "sparse_pc.ply"
    sparse_ply = (tmp_path / "processed" / "sparse_pc.ply").read_text(encoding="utf-8")
    assert "element vertex" in sparse_ply
    assert "property uchar red" in sparse_ply
    sparse = tmp_path / "processed" / "colmap" / "sparse" / "0"
    assert struct.unpack("<Q", (sparse / "cameras.bin").read_bytes()[:8])[0] == 2
    assert struct.unpack("<Q", (sparse / "images.bin").read_bytes()[:8])[0] == 2
    assert struct.unpack("<Q", (sparse / "points3D.bin").read_bytes()[:8])[0] >= 2

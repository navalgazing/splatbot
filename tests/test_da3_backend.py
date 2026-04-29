import pytest

from splatbot import da3_backend
from splatbot.da3_backend import Da3BackendError, opencv_world_to_camera_to_nerfstudio_c2w, required_prediction_attr


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

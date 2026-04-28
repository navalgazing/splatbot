import pytest

from splatbot.da3_backend import Da3BackendError, opencv_world_to_camera_to_nerfstudio_c2w, required_prediction_attr


def test_required_prediction_attr_raises_runtime_error() -> None:
    with pytest.raises(Da3BackendError, match="required field"):
        required_prediction_attr(object(), "extrinsics", "exts")


def test_opencv_world_to_camera_rejects_invalid_shape() -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(Da3BackendError, match="unsupported DA3 extrinsic shape"):
        opencv_world_to_camera_to_nerfstudio_c2w(np.zeros((2, 2)))

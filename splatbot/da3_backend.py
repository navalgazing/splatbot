from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import time
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"
DEFAULT_MODEL_CACHE_DIR = Path("/opt/splatbot/models/da3")
MIN_MODEL_FILE_BYTES = 1_000_000


class Da3BackendError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 for Splatbot pose/depth priors.")
    parser.add_argument("--images", "--input", dest="images_dir", required=True, type=Path)
    parser.add_argument("--processed", "--output", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("SPLATBOT_DA3_MODEL", DEFAULT_MODEL))
    parser.add_argument("--device", default=os.environ.get("SPLATBOT_DA3_DEVICE", "cuda"))
    parser.add_argument(
        "--use-ray-pose",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.environ.get("SPLATBOT_DA3_USE_RAY_POSE", "true")),
    )
    parser.add_argument(
        "--ref-view-strategy",
        default=os.environ.get("SPLATBOT_DA3_REF_VIEW_STRATEGY", "middle"),
    )
    parser.add_argument("--depth-only", action="store_true")
    args = parser.parse_args()

    images = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found under {args.images_dir}")
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    if args.depth_only and (args.processed_dir / "transforms.json").exists() and depth_priors_exist(args.processed_dir):
        return

    try:
        prediction = run_da3(
            images,
            args.model,
            args.device,
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy=args.ref_view_strategy,
        )
        if not args.depth_only:
            write_nerfstudio_dataset(images, prediction, args.processed_dir)
        write_depth_priors(prediction, args.processed_dir)
    except Da3BackendError as exc:
        raise SystemExit(str(exc)) from exc


def run_da3(
    images: list[Path],
    model_name: str,
    device: str,
    *,
    use_ray_pose: bool,
    ref_view_strategy: str,
):
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except Exception as exc:  # noqa: BLE001
        raise Da3BackendError(f"Depth Anything 3 is not installed or importable: {exc}") from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise Da3BackendError("Depth Anything 3 requires CUDA for Splatbot best preset, but CUDA is unavailable")
    model_path = ensure_da3_model(model_name)
    model = DepthAnything3.from_pretrained(model_path)
    model = model.to(device=torch.device(device))
    return model.inference(
        [str(path) for path in images],
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    )


def ensure_da3_model(model_name: str) -> str:
    local_path = local_da3_model_path(model_name)
    if local_path is not None and has_da3_model_files(local_path):
        return str(local_path)
    if looks_like_local_path(model_name):
        raise Da3BackendError(f"DA3 model path is missing or incomplete: {model_name}")

    cache_dir = configured_cache_dir() / safe_model_dir_name(model_name)
    if has_da3_model_files(cache_dir):
        return str(cache_dir)
    if not parse_bool(os.environ.get("SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD", "true")):
        raise Da3BackendError(
            f"DA3 model {model_name!r} is missing from {cache_dir}; "
            "pre-warm the RunPod volume/image or set SPLATBOT_DA3_ALLOW_MODEL_DOWNLOAD=true."
        )
    download_da3_model(model_name, cache_dir)
    if not has_da3_model_files(cache_dir):
        raise Da3BackendError(f"downloaded DA3 model is missing required files under {cache_dir}")
    return str(cache_dir)


def local_da3_model_path(model_name: str) -> Path | None:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return candidate if looks_like_local_path(model_name) else None


def looks_like_local_path(model_name: str) -> bool:
    return model_name.startswith(("/", "./", "../", "~"))


def configured_cache_dir() -> Path:
    raw = os.environ.get("SPLATBOT_DA3_MODEL_CACHE_DIR", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MODEL_CACHE_DIR


def safe_model_dir_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", ".", "_"} else "__" for ch in model_name)


def has_da3_model_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size >= MIN_MODEL_FILE_BYTES
    if not (path / "config.json").exists():
        return False
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        if any(file.is_file() and file.stat().st_size >= MIN_MODEL_FILE_BYTES for file in path.glob(pattern)):
            return True
    return False


def download_da3_model(model_name: str, cache_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise Da3BackendError(f"huggingface_hub is required to download DA3 model {model_name!r}: {exc}") from exc

    attempts = max(1, int(os.environ.get("SPLATBOT_DA3_MODEL_DOWNLOAD_ATTEMPTS", "5") or "5"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"downloading DA3 model attempt {attempt}/{attempts}: {model_name} -> {cache_dir}", flush=True)
            snapshot_download(
                repo_id=model_name,
                local_dir=str(cache_dir),
                resume_download=True,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                time.sleep(min(30, 2**attempt))
    raise Da3BackendError(f"failed to download DA3 model {model_name!r} after {attempts} attempt(s): {last_error}")


def write_nerfstudio_dataset(images: list[Path], prediction, processed_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    image_out = processed_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    extrinsics = np.asarray(required_prediction_attr(prediction, "extrinsics", "exts"))
    intrinsics = np.asarray(required_prediction_attr(prediction, "intrinsics", "ixts"))
    depths = np.asarray(getattr(prediction, "depth", []))
    if len(extrinsics) != len(images) or len(intrinsics) != len(images):
        raise Da3BackendError(
            f"DA3 returned {len(extrinsics)} pose(s) and {len(intrinsics)} intrinsic(s) for {len(images)} image(s)"
        )

    frames = []
    for idx, src in enumerate(images):
        dest = image_out / src.name
        shutil.copy2(src, dest)
        with Image.open(src) as image:
            width, height = image.size
        transform = opencv_world_to_camera_to_nerfstudio_c2w(extrinsics[idx])
        k = scale_intrinsics_to_image(intrinsics[idx], width, height, depth_shape_at(depths, idx))
        frames.append(
            {
                "file_path": f"images/{dest.name}",
                "fl_x": float(k[0, 0]),
                "fl_y": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "w": int(width),
                "h": int(height),
                "transform_matrix": transform,
            }
        )
    write_json(
        processed_dir / "transforms.json",
        {
            "camera_model": "OPENCV",
            "frames": frames,
        },
    )
    write_da3_sparse_seed(images, prediction, processed_dir)


def depth_shape_at(depths, idx: int) -> tuple[int, int] | None:
    try:
        depth = depths[idx]
    except (IndexError, TypeError):
        return None
    if getattr(depth, "ndim", 0) < 2:
        return None
    return int(depth.shape[-2]), int(depth.shape[-1])


def scale_intrinsics_to_image(k, width: int, height: int, depth_shape: tuple[int, int] | None):
    import numpy as np

    scaled = np.asarray(k, dtype=np.float64).copy()
    if scaled.shape[0] < 3 or scaled.shape[1] < 3:
        raise Da3BackendError(f"unsupported DA3 intrinsic shape: {scaled.shape}")
    if depth_shape is None:
        return scaled
    depth_h, depth_w = depth_shape
    if depth_w <= 0 or depth_h <= 0:
        return scaled
    if depth_w != width:
        scaled[0, 0] *= width / depth_w
        scaled[0, 2] *= width / depth_w
    if depth_h != height:
        scaled[1, 1] *= height / depth_h
        scaled[1, 2] *= height / depth_h
    return scaled


def write_da3_sparse_seed(images: list[Path], prediction, processed_dir: Path) -> None:
    """Write a COLMAP-compatible sparse seed model from DA3 poses and object depths."""

    import numpy as np
    from PIL import Image

    depths = np.asarray(required_prediction_attr(prediction, "depth"))
    extrinsics = np.asarray(required_prediction_attr(prediction, "extrinsics", "exts"))
    intrinsics = np.asarray(required_prediction_attr(prediction, "intrinsics", "ixts"))
    conf = np.asarray(getattr(prediction, "conf", []))
    if len(depths) != len(images):
        raise Da3BackendError(f"DA3 returned {len(depths)} depth map(s) for {len(images)} image(s)")

    sparse_dir = processed_dir / "colmap" / "sparse" / "0"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cameras = []
    image_entries = []
    points = []
    points_per_image = max(1, int(os.environ.get("SPLATBOT_DA3_SPARSE_POINTS_PER_IMAGE", "96") or "96"))
    alpha_threshold = max(0, int(os.environ.get("SPLATBOT_DA3_SPARSE_ALPHA_THRESHOLD", "16") or "16"))
    point_id = 1

    for idx, src in enumerate(images):
        image_id = idx + 1
        camera_id = idx + 1
        with Image.open(src) as image:
            rgba = image.convert("RGBA")
            width, height = rgba.size
            rgba_np = np.asarray(rgba)
        depth = np.asarray(depths[idx], dtype=np.float32)
        if depth.ndim != 2:
            raise Da3BackendError(f"DA3 depth map {idx + 1} has unsupported shape: {depth.shape}")
        depth_h, depth_w = int(depth.shape[0]), int(depth.shape[1])
        k = scale_intrinsics_to_image(intrinsics[idx], width, height, (depth_h, depth_w))
        cameras.append(
            {
                "id": camera_id,
                "model_id": 1,  # PINHOLE
                "width": width,
                "height": height,
                "params": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
            }
        )
        w2c = opencv_world_to_camera_matrix(extrinsics[idx])
        c2w = np.linalg.inv(w2c)
        samples = select_sparse_depth_samples(depth, confidence_at(conf, len(depths), idx), points_per_image)
        points2d = []
        local_point_index = 0
        for y_depth, x_depth in samples:
            z = float(depth[y_depth, x_depth])
            if not np.isfinite(z) or z <= 0:
                continue
            x_img = (float(x_depth) + 0.5) * width / depth_w
            y_img = (float(y_depth) + 0.5) * height / depth_h
            x_rgba = min(width - 1, max(0, int(round(x_img - 0.5))))
            y_rgba = min(height - 1, max(0, int(round(y_img - 0.5))))
            pixel = rgba_np[y_rgba, x_rgba]
            if int(pixel[3]) <= alpha_threshold:
                continue
            xyz_cam = np.array(
                [
                    (x_img - float(k[0, 2])) * z / float(k[0, 0]),
                    (y_img - float(k[1, 2])) * z / float(k[1, 1]),
                    z,
                    1.0,
                ],
                dtype=np.float64,
            )
            xyz_world = c2w @ xyz_cam
            points2d.append((x_img, y_img, point_id))
            points.append(
                {
                    "id": point_id,
                    "xyz": [float(value) for value in xyz_world[:3]],
                    "rgb": [int(pixel[0]), int(pixel[1]), int(pixel[2])],
                    "track": [(image_id, local_point_index)],
                }
            )
            point_id += 1
            local_point_index += 1
        image_entries.append(
            {
                "id": image_id,
                "qvec": rotation_matrix_to_qvec(w2c[:3, :3]),
                "tvec": [float(value) for value in w2c[:3, 3]],
                "camera_id": camera_id,
                "name": src.name,
                "points2d": points2d,
            }
        )

    if len(points) < int(os.environ.get("SPLATBOT_DA3_MIN_SPARSE_POINTS", "1000") or "1000"):
        raise Da3BackendError(
            f"DA3 sparse seed created only {len(points)} point(s); object masks/depths are too sparse"
        )
    ply_path = processed_dir / "sparse_pc.ply"
    write_sparse_points_ply(ply_path, points)
    add_sparse_point_cloud_to_transforms(processed_dir / "transforms.json", ply_path.name)
    write_cameras_binary(sparse_dir / "cameras.bin", cameras)
    write_images_binary(sparse_dir / "images.bin", image_entries)
    write_points3d_binary(sparse_dir / "points3D.bin", points)


def opencv_world_to_camera_matrix(extrinsic):
    import numpy as np

    matrix = np.asarray(extrinsic, dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    if matrix.shape == (3, 4):
        w2c[:3, :4] = matrix
    elif matrix.shape == (4, 4):
        w2c = matrix
    else:
        raise Da3BackendError(f"unsupported DA3 extrinsic shape: {matrix.shape}")
    return w2c


def confidence_at(conf, depth_count: int, idx: int):
    try:
        if getattr(conf, "ndim", 0) >= 3 and len(conf) == depth_count:
            return conf[idx]
    except TypeError:
        return None
    return None


def select_sparse_depth_samples(depth, conf, max_points: int) -> list[tuple[int, int]]:
    import numpy as np

    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        confidence = np.asarray(conf)
        if confidence.shape == depth.shape and np.isfinite(confidence).any():
            threshold = np.nanpercentile(confidence, 60)
            valid &= confidence >= threshold
    coords = np.argwhere(valid)
    if len(coords) <= max_points:
        return [(int(y), int(x)) for y, x in coords]
    step = max(1, len(coords) // max_points)
    selected = coords[::step][:max_points]
    return [(int(y), int(x)) for y, x in selected]


def rotation_matrix_to_qvec(rotation) -> list[float]:
    import numpy as np

    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = (1.0 + r[0, 0] - r[1, 1] - r[2, 2]) ** 0.5 * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = (1.0 + r[1, 1] - r[0, 0] - r[2, 2]) ** 0.5 * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = (1.0 + r[2, 2] - r[0, 0] - r[1, 1]) ** 0.5 * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return [float(value) for value in q]


def write_cameras_binary(path: Path, cameras: list[dict]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            handle.write(
                struct.pack(
                    "<iiQQ",
                    int(camera["id"]),
                    int(camera["model_id"]),
                    int(camera["width"]),
                    int(camera["height"]),
                )
            )
            for param in camera["params"]:
                handle.write(struct.pack("<d", float(param)))


def write_images_binary(path: Path, images: list[dict]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            handle.write(struct.pack("<i", int(image["id"])))
            for value in image["qvec"]:
                handle.write(struct.pack("<d", float(value)))
            for value in image["tvec"]:
                handle.write(struct.pack("<d", float(value)))
            handle.write(struct.pack("<i", int(image["camera_id"])))
            handle.write(str(image["name"]).encode("utf-8") + b"\x00")
            points2d = image["points2d"]
            handle.write(struct.pack("<Q", len(points2d)))
            for x, y, point3d_id in points2d:
                handle.write(struct.pack("<ddq", float(x), float(y), int(point3d_id)))


def write_points3d_binary(path: Path, points: list[dict]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point in points:
            handle.write(struct.pack("<Q", int(point["id"])))
            for value in point["xyz"]:
                handle.write(struct.pack("<d", float(value)))
            handle.write(bytes(int(max(0, min(255, value))) for value in point["rgb"]))
            handle.write(struct.pack("<d", 0.0))
            track = point["track"]
            handle.write(struct.pack("<Q", len(track)))
            for image_id, point2d_idx in track:
                handle.write(struct.pack("<ii", int(image_id), int(point2d_idx)))


def write_sparse_points_ply(path: Path, points: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point in points:
            x, y, z = point["xyz"]
            red, green, blue = point["rgb"]
            handle.write(
                f"{float(x):.9g} {float(y):.9g} {float(z):.9g} "
                f"{int(red)} {int(green)} {int(blue)}\n"
            )


def add_sparse_point_cloud_to_transforms(transforms_path: Path, ply_file_path: str) -> None:
    try:
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Da3BackendError(f"could not update transforms.json with DA3 sparse point cloud: {exc}") from exc
    data["ply_file_path"] = ply_file_path
    write_json(transforms_path, data)


def opencv_world_to_camera_to_nerfstudio_c2w(extrinsic) -> list[list[float]]:
    import numpy as np

    w2c = opencv_world_to_camera_matrix(extrinsic)
    c2w_opencv = np.linalg.inv(w2c)
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = c2w_opencv @ opencv_to_opengl
    return [[float(value) for value in row] for row in c2w.tolist()]


def write_depth_priors(prediction, processed_dir: Path) -> None:
    import numpy as np

    depth_dir = processed_dir / "depth_priors"
    depth_dir.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(prediction.depth)
    conf = np.asarray(getattr(prediction, "conf", []))
    for idx, depth in enumerate(depths):
        np.save(depth_dir / f"depth_{idx + 1:05d}.npy", depth.astype("float32"))
        if len(conf) == len(depths):
            np.save(depth_dir / f"confidence_{idx + 1:05d}.npy", conf[idx].astype("float32"))

    transforms_path = processed_dir / "transforms.json"
    if transforms_path.exists():
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        if isinstance(frames, list):
            for idx, frame in enumerate(frames[: len(depths)]):
                if isinstance(frame, dict):
                    frame["depth_file_path"] = f"depth_priors/depth_{idx + 1:05d}.npy"
            write_json(transforms_path, data)


def required_prediction_attr(prediction, *names: str):
    for name in names:
        if hasattr(prediction, name):
            return getattr(prediction, name)
    raise Da3BackendError(f"DA3 prediction did not include required field: {'/'.join(names)}")


def depth_priors_exist(processed_dir: Path) -> bool:
    depth_dir = processed_dir / "depth_priors"
    return depth_dir.exists() and any(depth_dir.glob("depth_*.npy"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"


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
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=torch.device(device))
    return model.inference(
        [str(path) for path in images],
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    )


def write_nerfstudio_dataset(images: list[Path], prediction, processed_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    image_out = processed_dir / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    extrinsics = np.asarray(required_prediction_attr(prediction, "extrinsics", "exts"))
    intrinsics = np.asarray(required_prediction_attr(prediction, "intrinsics", "ixts"))
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
        k = intrinsics[idx]
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

def opencv_world_to_camera_to_nerfstudio_c2w(extrinsic) -> list[list[float]]:
    import numpy as np

    w2c = np.eye(4, dtype=np.float64)
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape == (3, 4):
        w2c[:3, :4] = matrix
    elif matrix.shape == (4, 4):
        w2c = matrix
    else:
        raise Da3BackendError(f"unsupported DA3 extrinsic shape: {matrix.shape}")
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

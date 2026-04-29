from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from splatbot.commands import render_argv_template


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
DEFAULT_MAST3R_WEIGHTS = Path("/opt/splatbot/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
DEFAULT_MAST3R_WEIGHTS_URL = (
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
MIN_MAST3R_WEIGHTS_BYTES = 1_000_000_000


class ExternalCommandFailed(RuntimeError):
    def __init__(self, argv: list[str], returncode: int) -> None:
        self.argv = argv
        self.returncode = returncode
        super().__init__(f"external command failed ({returncode}): {' '.join(argv)}")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def even_sample(items: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    if max_count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (max_count - 1)
    return [items[round(idx * step)] for idx in range(max_count)]


def max_images_from_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def run(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise ExternalCommandFailed(argv, result.returncode)


@contextlib.contextmanager
def work_dir(path: Path | None, prefix: str) -> Iterator[Path]:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield Path(tmp)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def stage_images(images_dir: Path, staged_dir: Path, max_images: int = 0) -> list[Path]:
    images = image_files(images_dir)
    if not images:
        raise SystemExit(f"no images found under {images_dir}")
    selected = even_sample(images, max_images)
    if len(selected) != len(images):
        print(f"pose adapter selected {len(selected)}/{len(images)} images for {staged_dir}", flush=True)
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    for src in selected:
        link_or_copy(src.resolve(), staged_dir / src.name)
    return selected


def copy_images(images: list[Path], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        shutil.copy2(src, output_dir / src.name)


def model_file(model_dir: Path, stem: str) -> Path:
    binary = model_dir / f"{stem}.bin"
    if binary.exists():
        return binary
    return model_dir / f"{stem}.txt"


def has_colmap_model_files(model_dir: Path) -> bool:
    return all(model_file(model_dir, stem).exists() for stem in ("cameras", "images", "points3D"))


def read_registered_image_count(images_path: Path) -> int | None:
    if images_path.suffix == ".bin":
        try:
            from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

            return len(read_images_binary(images_path))
        except Exception:  # noqa: BLE001
            pass
        try:
            text = images_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        if text.startswith("images="):
            try:
                return int(text.split("=", 1)[1])
            except ValueError:
                return None
        return None

    try:
        text = images_path.read_text(encoding="utf-8")
    except OSError:
        return None
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count // 2 if count else 0


def points3d_count(points_path: Path) -> int | None:
    if points_path.suffix == ".bin":
        try:
            with points_path.open("rb") as handle:
                data = handle.read(8)
        except OSError:
            return None
        if len(data) != 8:
            return None
        return struct.unpack("<Q", data)[0]

    try:
        text = points_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return sum(1 for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def sparse_model_score(model_dir: Path) -> tuple[int, int]:
    registered = read_registered_image_count(model_file(model_dir, "images"))
    points_count = points3d_count(model_file(model_dir, "points3D"))
    return (registered if registered is not None else -1, points_count if points_count is not None else -1)


def find_best_sparse_model(*roots: Path) -> Path:
    candidates: list[Path] = []
    for root in roots:
        if has_colmap_model_files(root):
            candidates.append(root)
        if root.exists():
            for images_path in root.rglob("images.bin"):
                model_dir = images_path.parent
                if has_colmap_model_files(model_dir):
                    candidates.append(model_dir)
            for images_path in root.rglob("images.txt"):
                model_dir = images_path.parent
                if has_colmap_model_files(model_dir):
                    candidates.append(model_dir)
    unique = sorted(set(candidates))
    if not unique:
        root_list = ", ".join(str(root) for root in roots)
        raise SystemExit(f"pose adapter did not find a COLMAP sparse model under: {root_list}")
    return max(unique, key=sparse_model_score)


def validate_sparse_model(model_dir: Path) -> tuple[int, int | None]:
    if not has_colmap_model_files(model_dir):
        raise SystemExit(f"COLMAP sparse model is missing cameras/images/points3D files: {model_dir}")

    registered = read_registered_image_count(model_file(model_dir, "images"))
    if registered is None:
        raise SystemExit(f"could not read registered image count from {model_file(model_dir, 'images')}")
    if registered <= 0:
        raise SystemExit(f"COLMAP sparse model registered no images: {model_dir}")

    point_count = points3d_count(model_file(model_dir, "points3D"))
    if point_count == 0 and not truthy(os.environ.get("SPLATBOT_POSE_ALLOW_EMPTY_POINTS")):
        raise SystemExit(
            f"COLMAP sparse model has zero points: {model_dir}. "
            "Set SPLATBOT_POSE_ALLOW_EMPTY_POINTS=true only for explicit diagnostics."
        )
    return registered, point_count


def generate_transforms(processed_dir: Path, sparse_model_dir: Path) -> int:
    try:
        from nerfstudio.process_data.colmap_utils import colmap_to_json
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Nerfstudio colmap_to_json is required to write transforms.json: {exc}") from exc

    return int(colmap_to_json(sparse_model_dir, processed_dir))


def finalize_colmap_pose_dataset(images_dir: Path, processed_dir: Path, sparse_model_dir: Path) -> None:
    images = image_files(images_dir)
    if not images:
        raise SystemExit(f"no images found under {images_dir}")

    registered, _ = validate_sparse_model(sparse_model_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    copy_images(images, processed_dir / "images")

    target_sparse = processed_dir / "colmap" / "sparse" / "0"
    if target_sparse.exists():
        shutil.rmtree(target_sparse)
    target_sparse.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sparse_model_dir, target_sparse)

    transforms_path = processed_dir / "transforms.json"
    transforms_path.unlink(missing_ok=True)
    frames = generate_transforms(processed_dir, target_sparse)
    if frames <= 0:
        raise SystemExit(f"Nerfstudio wrote no transforms frames from {target_sparse}")
    if frames != registered:
        print(
            f"warning: transforms frame count ({frames}) differs from sparse registered images ({registered})",
            file=sys.stderr,
            flush=True,
        )
    if not transforms_path.exists():
        raise SystemExit(f"Nerfstudio did not create {transforms_path}")


def default_python() -> str:
    return os.environ.get("SPLATBOT_POSE_PYTHON", "").strip() or sys.executable or "python"


def render_and_run(command: str, values: dict[str, str]) -> None:
    run(render_argv_template(command, values))


def retry_image_caps(max_images: int, min_images: int) -> list[int]:
    if max_images <= 0:
        return [0]
    if max_images <= min_images:
        return [max_images]
    caps: list[int] = []
    current = max_images
    minimum = max(1, min_images)
    while current >= minimum:
        caps.append(current)
        next_count = max(minimum, current // 2)
        if next_count == current:
            break
        current = next_count
    return caps


def vggt_default_command(script: Path) -> str:
    extra_args = os.environ.get("SPLATBOT_VGGT_ARGS", "").strip()
    command = f"{default_python()} {script} --scene_dir {{scene_dir}}"
    if extra_args:
        command += f" {extra_args}"
    return command


def remove_flag(command: str, flag: str) -> str:
    tokens = shlex.split(command)
    filtered = [token for token in tokens if token != flag]
    return shlex.join(filtered)


def replace_flag_value(command: str, flag: str, value: str, *, only_if_greater: int | None = None) -> str:
    tokens = shlex.split(command)
    try:
        index = tokens.index(flag)
    except ValueError:
        return command
    value_index = index + 1
    if value_index >= len(tokens):
        return command
    if only_if_greater is not None:
        try:
            current = int(tokens[value_index])
        except ValueError:
            return command
        if current <= only_if_greater:
            return command
    tokens[value_index] = value
    return shlex.join(tokens)


def vggt_command_variants(command: str) -> list[str]:
    lower_query = os.environ.get("SPLATBOT_VGGT_RETRY_MAX_QUERY_PTS", "").strip() or "1024"
    variants = [
        command,
        replace_flag_value(command, "--max_query_pts", lower_query, only_if_greater=int(lower_query)),
    ]
    if "--use_ba" in shlex.split(command):
        no_ba = remove_flag(command, "--use_ba")
        variants.extend(
            [
                no_ba,
                replace_flag_value(no_ba, "--max_query_pts", lower_query, only_if_greater=int(lower_query)),
            ]
        )
    unique: list[str] = []
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return unique


def vggt_main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT pose and normalize it for Splatbot/Nerfstudio.")
    parser.add_argument("--images", "--input", dest="images_dir", required=True, type=Path)
    parser.add_argument("--processed", "--output", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--matching-method", default="")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    vggt_repo = Path(os.environ.get("SPLATBOT_VGGT_REPO", "").strip() or "/opt/vggt")
    script = Path(os.environ.get("SPLATBOT_VGGT_DEMO_COLMAP", "").strip() or str(vggt_repo / "demo_colmap.py"))
    command = os.environ.get("SPLATBOT_VGGT_RUN_COMMAND", "").strip() or vggt_default_command(script)
    if "{scene_dir}" not in command:
        raise SystemExit("SPLATBOT_VGGT_RUN_COMMAND must include the {scene_dir} placeholder")

    with work_dir(args.work_dir, "splatbot-vggt-") as tmp:
        last_error: ExternalCommandFailed | None = None
        max_images = max_images_from_env("SPLATBOT_VGGT_MAX_IMAGES", 64)
        min_images = max_images_from_env("SPLATBOT_VGGT_MIN_IMAGES", 24)
        for image_cap in retry_image_caps(max_images, min_images):
            for attempt_index, attempt_command in enumerate(vggt_command_variants(command), start=1):
                scene_dir = tmp / f"scene_{image_cap or 'all'}_attempt_{attempt_index}"
                staged_images = scene_dir / "images"
                if scene_dir.exists():
                    shutil.rmtree(scene_dir)
                stage_images(args.images_dir, staged_images, max_images=image_cap)
                try:
                    render_and_run(
                        attempt_command,
                        {
                            "scene_dir": str(scene_dir),
                            "images_dir": str(staged_images),
                            "input_dir": str(staged_images),
                            "processed_dir": str(args.processed_dir),
                            "output_dir": str(args.processed_dir),
                            "work_dir": str(tmp),
                            "vggt_repo": str(vggt_repo),
                            "vggt_script": str(script),
                            "python": default_python(),
                            "matching_method": args.matching_method,
                        },
                    )
                except ExternalCommandFailed as exc:
                    last_error = exc
                    print(
                        f"VGGT failed with {image_cap} image(s) attempt {attempt_index}, "
                        f"retrying if another variant is available: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                sparse_model = find_best_sparse_model(scene_dir / "sparse", scene_dir, tmp)
                finalize_colmap_pose_dataset(staged_images, args.processed_dir, sparse_model)
                return
        if last_error is not None:
            raise SystemExit(str(last_error)) from last_error
        raise SystemExit("VGGT did not run")


def write_pairs_file(images: list[Path], pairs_path: Path, matching_method: str) -> None:
    pair_window = int(os.environ.get("SPLATBOT_MAST3R_PAIR_WINDOW", "").strip() or "5")
    cyclic = truthy(os.environ.get("SPLATBOT_MAST3R_PAIR_CYCLIC", "").strip() or "true")
    exhaustive = matching_method.strip().lower() == "exhaustive"
    names = [image.name for image in images]
    pairs: set[tuple[str, str]] = set()
    if exhaustive:
        for idx, left in enumerate(names):
            for right in names[idx + 1 :]:
                pairs.add((left, right))
    else:
        window = max(1, pair_window)
        for idx, left in enumerate(names):
            upper = min(len(names), idx + window + 1)
            for right_idx in range(idx + 1, upper):
                pairs.add((left, names[right_idx]))
            if cyclic and len(names) > 2:
                for offset in range(1, min(window, len(names) - 1) + 1):
                    right_idx = (idx + offset) % len(names)
                    if right_idx <= idx:
                        pair = tuple(sorted((left, names[right_idx])))
                        pairs.add(pair)
    if not pairs:
        raise SystemExit("MASt3R requires at least one image pair")
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(
        "# kapture format: 1.1\n"
        "# query_image, map_image, score\n"
        + "".join(f"{left}, {right}, 1.0\n" for left, right in sorted(pairs)),
        encoding="utf-8",
    )


def mast3r_default_command(
    repo: Path,
    output_dir: Path,
    pairs_path: Path,
    staged_images: Path,
    weights_path: Path | str | None = None,
) -> str:
    device = os.environ.get("SPLATBOT_MAST3R_DEVICE", "").strip() or "cuda"
    glomap_bin = os.environ.get("SPLATBOT_GLOMAP_BIN", "").strip() or "glomap"
    weights = str(weights_path) if weights_path is not None else (
        os.environ.get("SPLATBOT_MAST3R_WEIGHTS", "").strip() or str(DEFAULT_MAST3R_WEIGHTS)
    )
    shared_camera = truthy(os.environ.get("SPLATBOT_MAST3R_SHARED_CAMERA", "").strip() or "true")
    use_glomap = truthy(os.environ.get("SPLATBOT_MAST3R_USE_GLOMAP", "").strip() or "true")
    input_flag = "--dir_same_camera" if shared_camera else "--dir"
    if weights:
        model_args = f"--weights {shlex.quote(weights)}"
    else:
        model_args = "--model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    extra_args = os.environ.get("SPLATBOT_MAST3R_ARGS", "").strip()
    command = (
        f"{default_python()} {repo / 'kapture_mast3r_mapping.py'} "
        f"{model_args} {input_flag} {staged_images} --output {output_dir} "
        f"--pairsfile_path {pairs_path} --device {device} --glomap_bin {glomap_bin}"
    )
    if use_glomap:
        command += " --use_glomap_mapper"
    if extra_args:
        command += f" {extra_args}"
    return command


def mast3r_weight_path() -> Path:
    configured = os.environ.get("SPLATBOT_MAST3R_WEIGHTS", "").strip()
    return Path(configured) if configured else DEFAULT_MAST3R_WEIGHTS


def ensure_mast3r_weights() -> Path:
    weights = mast3r_weight_path()
    if weights.exists() and weights.stat().st_size >= MIN_MAST3R_WEIGHTS_BYTES:
        return weights
    if not truthy(os.environ.get("SPLATBOT_MAST3R_ALLOW_WEIGHT_DOWNLOAD", "true")):
        raise SystemExit(
            f"MASt3R weights are missing at {weights}; set SPLATBOT_MAST3R_ALLOW_WEIGHT_DOWNLOAD=true "
            "or pre-warm the RunPod volume/image."
        )
    url = os.environ.get("SPLATBOT_MAST3R_WEIGHTS_URL", "").strip() or DEFAULT_MAST3R_WEIGHTS_URL
    weights.parent.mkdir(parents=True, exist_ok=True)
    partial = weights.with_suffix(weights.suffix + ".part")
    if partial.exists() and partial.stat().st_size < MIN_MAST3R_WEIGHTS_BYTES:
        partial.unlink()
    print(f"downloading MASt3R weights: {url} -> {weights}", flush=True)
    if shutil.which("curl"):
        result = subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "-o",
                str(partial),
                url,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"failed to download MASt3R weights from {url}") from None
    else:
        urllib.request.urlretrieve(url, partial)
    if not partial.exists() or partial.stat().st_size < MIN_MAST3R_WEIGHTS_BYTES:
        raise SystemExit(f"downloaded MASt3R weights are missing or too small: {partial}")
    partial.replace(weights)
    return weights


def glomap_wrapper_main() -> None:
    real_bin = os.environ.get("SPLATBOT_REAL_GLOMAP_BIN", "").strip() or "glomap"
    argv = sys.argv[1:]
    if argv[:1] == ["mapper"]:
        extra_args = os.environ.get("SPLATBOT_GLOMAP_MAPPER_ARGS", "").strip()
        if extra_args:
            argv = ["mapper", *shlex.split(extra_args), *argv[1:]]
    print("+ " + " ".join([real_bin, *argv]), flush=True)
    result = subprocess.run([real_bin, *argv], check=False)
    raise SystemExit(result.returncode)


def mast3r_main() -> None:
    parser = argparse.ArgumentParser(description="Run MASt3R SfM and normalize it for Splatbot/Nerfstudio.")
    parser.add_argument("--images", "--input", dest="images_dir", required=True, type=Path)
    parser.add_argument("--processed", "--output", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--matching-method", default="")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    repo = Path(os.environ.get("SPLATBOT_MAST3R_REPO", "").strip() or "/opt/mast3r")
    command = os.environ.get("SPLATBOT_MAST3R_RUN_COMMAND", "").strip()

    with work_dir(args.work_dir, "splatbot-mast3r-") as tmp:
        staged_images = tmp / "images"
        images = stage_images(
            args.images_dir,
            staged_images,
            max_images=max_images_from_env("SPLATBOT_MAST3R_MAX_IMAGES", 120),
        )
        output_dir = tmp / "mast3r"
        pairs_path = tmp / "pairs.txt"
        write_pairs_file(images, pairs_path, args.matching_method)
        mast3r_weights = mast3r_weight_path()
        if not command or "{mast3r_weights}" in command:
            mast3r_weights = ensure_mast3r_weights()
        command = command or mast3r_default_command(repo, output_dir, pairs_path, staged_images, mast3r_weights)
        render_and_run(
            command,
            {
                "images_dir": str(staged_images),
                "input_dir": str(staged_images),
                "processed_dir": str(args.processed_dir),
                "output_dir": str(args.processed_dir),
                "work_dir": str(tmp),
                "mast3r_output_dir": str(output_dir),
                "pairs_file": str(pairs_path),
                "mast3r_repo": str(repo),
                "mast3r_weights": str(mast3r_weights),
                "python": default_python(),
                "matching_method": args.matching_method,
            },
        )
        sparse_model = find_best_sparse_model(output_dir / "reconstruction", output_dir, tmp)
        finalize_colmap_pose_dataset(staged_images, args.processed_dir, sparse_model)


def colmap_pose_adapter_main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an existing COLMAP sparse model for Splatbot/Nerfstudio.")
    parser.add_argument("--images", "--input", dest="images_dir", required=True, type=Path)
    parser.add_argument("--processed", "--output", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--sparse", required=True, type=Path)
    args = parser.parse_args()

    sparse_model = find_best_sparse_model(args.sparse)
    finalize_colmap_pose_dataset(args.images_dir, args.processed_dir, sparse_model)

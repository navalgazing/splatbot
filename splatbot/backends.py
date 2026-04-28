from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(argv: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(argv), flush=True)
    result = subprocess.run(argv, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def latest_config(ns_dir: Path) -> Path:
    configs = sorted(ns_dir.glob("**/config.yml"), key=lambda path: path.stat().st_mtime)
    if not configs:
        raise SystemExit(f"no Nerfstudio config.yml found under {ns_dir}")
    return configs[-1]


def ensure_output_files(output_dir: Path) -> None:
    if not any(path.is_file() for path in output_dir.iterdir()):
        raise SystemExit(f"backend produced no files under {output_dir}")


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def indexed_even_sample(items: list[Path], max_count: int) -> list[tuple[int, Path]]:
    if max_count <= 0 or len(items) <= max_count:
        return list(enumerate(items))
    if max_count == 1:
        idx = len(items) // 2
        return [(idx, items[idx])]
    indices = sorted({round(slot * (len(items) - 1) / (max_count - 1)) for slot in range(max_count)})
    return [(idx, items[idx]) for idx in indices]


def write_rgba_from_alpha(image_path: Path, alpha, output_path: Path) -> None:
    from PIL import Image

    image = Image.open(image_path).convert("RGBA")
    alpha_image = Image.fromarray(alpha.astype("uint8") * 255, mode="L")
    image.putalpha(alpha_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".png")
    image.save(output_path)


def best_mask(masks, scores=None):
    import numpy as np

    if masks is None:
        return None
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.size == 0:
        return None
    if masks.ndim == 2:
        return masks > 0
    masks = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
    if scores is not None:
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores).reshape(-1)
    ranked = []
    for idx, mask in enumerate(masks):
        area = float((mask > 0).sum())
        score = float(scores[idx]) if scores is not None and idx < len(scores) else 1.0
        ranked.append((score, area, idx))
    if not ranked:
        return None
    _, _, idx = max(ranked)
    return masks[idx] > 0


def segment_with_rembg(input_dir: Path, output_dir: Path) -> None:
    rembg = os.environ.get("SPLATBOT_REMBG_BIN", "rembg")
    run([rembg, "p", str(input_dir), str(output_dir)])


def render_backend_command(command: str, **values: str) -> list[str]:
    return render_argv_template(command, values)


def segment_with_external_command(input_dir: Path, output_dir: Path, backend: str, prompt: str) -> None:
    env_name = "SPLATBOT_MATTING_COMMAND" if backend in {"matting", "matanyone"} else "SPLATBOT_OBJECT_MASK_COMMAND"
    command = os.environ.get(env_name, "").strip()
    if not command:
        raise SystemExit(f"{env_name} is required for segmentation backend {backend!r}")
    run(
        render_backend_command(
            command,
            backend=backend,
            input_dir=str(input_dir),
            images_dir=str(input_dir),
            output_dir=str(output_dir),
            object_dir=str(output_dir),
            prompt=prompt,
        )
    )
    ensure_output_files(output_dir)


def segment_with_sam3(input_dir: Path, output_dir: Path, prompt: str) -> None:
    try:
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"SAM3 is not installed or importable: {exc}") from exc

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    images = image_files(input_dir)
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=prompt)
        mask = best_mask(result.get("masks"), result.get("scores"))
        if mask is None:
            continue
        write_rgba_from_alpha(image_path, mask, output_dir / image_path.with_suffix(".png").name)
    ensure_output_files(output_dir)


def segment_backend_self_test(backend: str) -> None:
    if backend == "sam3":
        try:
            from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
            from sam3.model_builder import build_sam3_image_model  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"SAM3 self-test failed: {exc}") from exc
        return
    if backend in {"sam2", "sam2-video"}:
        checkpoint = os.environ.get("SPLATBOT_SAM2_CHECKPOINT", "")
        config = os.environ.get("SPLATBOT_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
        if not checkpoint:
            raise SystemExit("SAM2 self-test failed: SPLATBOT_SAM2_CHECKPOINT is required")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size <= 0:
            raise SystemExit(f"SAM2 self-test failed: checkpoint is missing: {checkpoint_path}")
        try:
            build_sam2_predictor(config, checkpoint)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"SAM2 self-test failed: {exc}") from exc
        return
    if backend in {"rembg", "matting", "matanyone", "external"}:
        if backend in {"matting", "matanyone"} and not os.environ.get("SPLATBOT_MATTING_COMMAND", "").strip():
            raise SystemExit("matting self-test failed: SPLATBOT_MATTING_COMMAND is required")
        if backend == "external" and not os.environ.get("SPLATBOT_OBJECT_MASK_COMMAND", "").strip():
            raise SystemExit("external segmentation self-test failed: SPLATBOT_OBJECT_MASK_COMMAND is required")
        try:
            if backend == "rembg":
                import rembg  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"rembg self-test failed: {exc}") from exc
        return
    raise SystemExit(f"unsupported segmentation backend for self-test: {backend}")


def bbox_score(bbox: tuple[int, int, int, int], width: int, height: int) -> float:
    left, top, right, bottom = bbox
    area = max(0, right - left) * max(0, bottom - top)
    pixels = max(1, width * height)
    area_ratio = area / pixels
    center_x = ((left + right) / 2.0) / max(1, width)
    center_y = ((top + bottom) / 2.0) / max(1, height)
    center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    center_score = 1.0 - min(center_distance / 0.7072, 1.0)
    if area_ratio <= 0:
        area_score = 0.0
    elif area_ratio < 0.02:
        area_score = area_ratio / 0.02
    elif area_ratio > 0.75:
        area_score = max(0.0, 1.0 - ((area_ratio - 0.75) / 0.25))
    else:
        area_score = 1.0
    edge_penalty = 0.25 if left <= 1 or top <= 1 or right >= width - 1 or bottom >= height - 1 else 0.0
    return (0.65 * area_score) + (0.35 * center_score) - edge_penalty


def rembg_bootstrap_box(input_dir: Path, max_samples: int = 12) -> tuple[int, tuple[int, int, int, int]] | None:
    import tempfile

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Pillow is required for SAM2 box bootstrapping: {exc}") from exc
    images = image_files(input_dir)
    sampled = indexed_even_sample(images, max_samples)
    if not sampled:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_input = Path(tmp) / "input"
        tmp_output = Path(tmp) / "output"
        tmp_input.mkdir()
        tmp_output.mkdir()
        for _, image_path in sampled:
            shutil.copy2(image_path, tmp_input / image_path.name)
        segment_with_rembg(tmp_input, tmp_output)
        candidates: list[tuple[float, int, tuple[int, int, int, int]]] = []
        for frame_idx, image_path in sampled:
            segmented = tmp_output / f"{image_path.stem}.png"
            if not segmented.exists():
                segmented = next(iter(sorted(tmp_output.glob(f"{image_path.stem}.*"))), None)
            if segmented is None or not segmented.exists():
                continue
            alpha = Image.open(segmented).convert("RGBA").getchannel("A")
            bbox = alpha.getbbox()
            if bbox is None:
                continue
            candidates.append((bbox_score(bbox, alpha.width, alpha.height), frame_idx, bbox))
        if not candidates:
            return None
        _, frame_idx, bbox = max(candidates)
        return frame_idx, bbox


def build_sam2_predictor(config: str, checkpoint: str):
    from sam2.build_sam import build_sam2_video_predictor

    vos_optimized = os.environ.get("SPLATBOT_SAM2_VOS_OPTIMIZED", "").lower() in {"1", "true", "yes", "on"}
    if vos_optimized:
        try:
            return build_sam2_video_predictor(config, checkpoint, vos_optimized=True)
        except TypeError:
            pass
    return build_sam2_video_predictor(config, checkpoint)


def write_sam2_mask(frame_idx: int, mask_logits, images: list[Path], output_dir: Path) -> None:
    if frame_idx < 0 or frame_idx >= len(images):
        return
    try:
        if len(mask_logits) == 0:
            return
        mask_source = mask_logits[0] > 0
    except TypeError:
        mask_source = mask_logits > 0
    mask = best_mask(mask_source)
    if mask is None:
        return
    write_rgba_from_alpha(images[frame_idx], mask, output_dir / images[frame_idx].with_suffix(".png").name)


def has_object_ids(object_ids) -> bool:
    if object_ids is None:
        return False
    try:
        return len(object_ids) > 0
    except TypeError:
        return bool(object_ids)


def stage_sam2_video_frames(images: list[Path], video_dir: Path) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(images):
        staged = video_dir / f"{idx}.jpg"
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                staged.symlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, staged)
            continue
        from PIL import Image

        Image.open(image_path).convert("RGB").save(staged, quality=95)


def segment_with_sam2(input_dir: Path, output_dir: Path) -> None:
    try:
        import numpy as np
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"SAM2 is not installed or importable: {exc}") from exc
    checkpoint = os.environ.get("SPLATBOT_SAM2_CHECKPOINT", "")
    config = os.environ.get("SPLATBOT_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    if not checkpoint:
        raise SystemExit("SPLATBOT_SAM2_CHECKPOINT is required for SAM2 segmentation")
    images = image_files(input_dir)
    if not images:
        raise SystemExit(f"no images found for SAM2 segmentation under {input_dir}")
    bootstrap = rembg_bootstrap_box(input_dir)
    if bootstrap is None:
        raise SystemExit("could not bootstrap a SAM2 tracking box from rembg")
    bootstrap_idx, box = bootstrap
    predictor = build_sam2_predictor(config, checkpoint)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with tempfile.TemporaryDirectory() as tmp:
        video_dir = Path(tmp) / "sam2_frames"
        stage_sam2_video_frames(images, video_dir)
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(video_dir))
            frame_idx, object_ids, mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=bootstrap_idx,
                obj_id=1,
                box=np.array(box, dtype=np.float32),
            )
            if has_object_ids(object_ids):
                write_sam2_mask(frame_idx, mask_logits, images, output_dir)
            for reverse in (False, True):
                for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=bootstrap_idx,
                    reverse=reverse,
                ):
                    if has_object_ids(object_ids):
                        write_sam2_mask(frame_idx, mask_logits, images, output_dir)
    ensure_output_files(output_dir)


def segment_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot segmentation backends.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["sam3", "sam3-video", "sam2", "sam2-video", "matting", "matanyone", "rembg", "external"],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default=os.environ.get("SPLATBOT_OBJECT_MASK_PROMPT", "main object"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        segment_backend_self_test(args.backend)
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend in {"sam3", "sam3-video"}:
        segment_with_sam3(args.input, args.output, args.prompt)
    elif args.backend in {"sam2", "sam2-video"}:
        segment_with_sam2(args.input, args.output)
    elif args.backend in {"matting", "matanyone", "external"}:
        segment_with_external_command(args.input, args.output, args.backend, args.prompt)
    else:
        segment_with_rembg(args.input, args.output)


def pose_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot pose backends.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--input", "--images", dest="input_dir", required=True, type=Path)
    parser.add_argument("--output", "--processed", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--matching-method", default="")
    args = parser.parse_args()
    if args.backend in {"da3-colmap", "vggt-colmap", "mast3r-sfm"}:
        run_external_pose_backend(args.backend, args.input_dir, args.processed_dir, args.matching_method)
        return
    ns_process = os.environ.get("SPLATBOT_NS_PROCESS_DATA_BIN", "ns-process-data")
    env = os.environ.copy()
    if args.backend in {"colmap-global", "global", "glomap"}:
        env["SPLATBOT_COLMAP_MAPPER"] = "global"
    argv = [
        ns_process,
        "images",
        "--data",
        str(args.input_dir),
        "--output-dir",
        str(args.processed_dir),
        "--colmap-cmd",
        "splatbot-colmap-wrapper",
    ]
    if args.matching_method:
        argv.extend(["--matching-method", args.matching_method])
    if env.get("SPLATBOT_COLMAP_USE_GPU", "false").lower() not in {"1", "true", "yes", "on"}:
        argv.append("--no-gpu")
    run(argv, env=env)


def run_external_pose_backend(
    backend: str,
    input_dir: Path,
    processed_dir: Path,
    matching_method: str,
) -> None:
    command = pose_command_for_backend(backend)
    if not command:
        raise SystemExit(
            f"pose backend {backend!r} is not configured; set the matching SPLATBOT_*_POSE_COMMAND"
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    run(
        render_backend_command(
            command,
            backend=backend,
            input_dir=str(input_dir),
            images_dir=str(input_dir),
            output_dir=str(processed_dir),
            processed_dir=str(processed_dir),
            matching_method=matching_method,
            colmap_bin=os.environ.get("SPLATBOT_COLMAP_BIN", "colmap"),
            glomap_bin=os.environ.get("SPLATBOT_GLOMAP_BIN", "glomap"),
        )
    )
    if not (processed_dir / "transforms.json").exists():
        raise SystemExit(
            f"pose backend {backend!r} did not create {processed_dir / 'transforms.json'}"
        )


def pose_command_for_backend(backend: str) -> str:
    env_names = {
        "da3-colmap": "SPLATBOT_DA3_POSE_COMMAND",
        "vggt-colmap": "SPLATBOT_VGGT_POSE_COMMAND",
        "mast3r-sfm": "SPLATBOT_MAST3R_POSE_COMMAND",
    }
    return os.environ.get(env_names.get(backend, ""), "").strip()


def depth_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot depth-prior backends.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--processed", "--data", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--images", "--input", dest="images_dir", required=True, type=Path)
    args = parser.parse_args()
    command = depth_command_for_backend(args.backend)
    if not command:
        raise SystemExit(
            f"depth backend {args.backend!r} is not configured; set SPLATBOT_DA3_DEPTH_COMMAND, "
            "SPLATBOT_DEPTH_ANYTHING_V2_COMMAND, or SPLATBOT_DEPTH_BACKEND_COMMAND"
        )
    run(
        render_backend_command(
            command,
            backend=args.backend,
            processed_dir=str(args.processed_dir),
            data_dir=str(args.processed_dir),
            images_dir=str(args.images_dir),
            input_dir=str(args.images_dir),
        )
    )


def depth_command_for_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized.startswith("da3"):
        return os.environ.get("SPLATBOT_DA3_DEPTH_COMMAND", "").strip()
    if normalized.startswith("depth-anything-v2"):
        return os.environ.get("SPLATBOT_DEPTH_ANYTHING_V2_COMMAND", "").strip()
    return os.environ.get("SPLATBOT_DEPTH_BACKEND_COMMAND", "").strip()


def prepare_dn_splatter_depths(data_dir: Path) -> None:
    script = [
        sys.executable,
        "-m",
        "dn_splatter.scripts.align_depth",
        "--data",
        str(data_dir),
        "--skip-colmap-to-depths",
        "--skip_alignment",
    ]
    result = subprocess.run(script, check=False)
    if result.returncode != 0:
        raise SystemExit("dn-splatter depth preparation failed")
    mono_depth_dir = data_dir / "mono_depth"
    if not mono_depth_dir.exists() or not any(mono_depth_dir.glob("*.npy")):
        raise SystemExit("dn-splatter depth preparation produced no mono_depth npy files")


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot training backends.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--data", "--processed", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--output", "--ns-dir", dest="ns_dir", required=True, type=Path)
    parser.add_argument("--max-iterations", required=True)
    parser.add_argument("--steps-per-save", default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ns_train = os.environ.get("SPLATBOT_NS_TRAIN_BIN", "ns-train")
    backend = args.backend
    if backend in {"3dgs-mcmc", "mip-splatting", "2dgs"}:
        run_external_train_backend(
            backend,
            args.processed_dir,
            args.ns_dir,
            args.max_iterations,
            args.steps_per_save,
            [arg for arg in args.extra if arg != "--"],
        )
        return
    if backend in {"dn-splatter", "dn-splatter-big", "ags-mesh"}:
        try:
            import dn_splatter  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"dn-splatter is not installed: {exc}") from exc
        prepare_dn_splatter_depths(args.processed_dir)
    argv = [
        ns_train,
        backend,
        "--data",
        str(args.processed_dir),
        "--output-dir",
        str(args.ns_dir),
        "--max-num-iterations",
        str(args.max_iterations),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    if args.steps_per_save:
        argv.extend(["--steps-per-save", str(args.steps_per_save)])
    if backend in {"dn-splatter", "dn-splatter-big", "ags-mesh"}:
        argv.extend(
            [
                "--pipeline.datamanager.dataparser.load-normals",
                "False",
                "--pipeline.model.use-depth-loss",
                "True",
                "--pipeline.model.depth-lambda",
                "0.2",
                "--pipeline.model.depth-loss-type",
                "PearsonDepth",
                "--pipeline.model.use-normal-loss",
                "False",
                "--pipeline.model.use-normal-tv-loss",
                "False",
            ]
        )
    argv.extend(arg for arg in args.extra if arg != "--")
    run(argv)


def run_external_train_backend(
    backend: str,
    processed_dir: Path,
    ns_dir: Path,
    max_iterations: str,
    steps_per_save: str | None,
    extra_args: list[str],
) -> None:
    command = train_command_for_backend(backend)
    if not command:
        raise SystemExit(
            f"training backend {backend!r} is not configured; set the matching SPLATBOT_*_TRAIN_COMMAND"
        )
    ns_dir.mkdir(parents=True, exist_ok=True)
    argv = render_argv_template(
        command,
        {
            "backend": backend,
            "processed_dir": str(processed_dir),
            "data_dir": str(processed_dir),
            "ns_dir": str(ns_dir),
            "output_dir": str(ns_dir),
            "max_iterations": max_iterations,
            "steps_per_save": steps_per_save or "",
            "extra_args": extra_args,
        },
    )
    run(argv)
    latest_config(ns_dir)


def train_command_for_backend(backend: str) -> str:
    env_names = {
        "3dgs-mcmc": "SPLATBOT_MCMC_TRAIN_COMMAND",
        "mip-splatting": "SPLATBOT_MIP_SPLATTING_TRAIN_COMMAND",
        "2dgs": "SPLATBOT_2DGS_TRAIN_COMMAND",
    }
    return os.environ.get(env_names.get(backend, ""), "").strip()


def mesh_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot mesh extraction backends.")
    parser.add_argument("--backend", default="o3dtsdf")
    parser.add_argument("--ns-dir", required=True, type=Path)
    parser.add_argument("--output", "--mesh-path", dest="mesh_path", required=True, type=Path)
    args = parser.parse_args()
    gs_mesh = shutil.which("gs-mesh")
    if not gs_mesh:
        raise SystemExit("gs-mesh is not installed; install dn-splatter/AGS-Mesh in the RunPod image")
    config = latest_config(args.ns_dir)
    output_dir = args.mesh_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    run([gs_mesh, args.backend, "--load-config", str(config), "--output-dir", str(output_dir)])
    candidates = sorted(output_dir.glob("*.ply")) + sorted(output_dir.glob("*.obj")) + sorted(output_dir.glob("*.glb"))
    if not candidates:
        raise SystemExit(f"gs-mesh produced no mesh under {output_dir}")
    produced = candidates[-1]
    if produced != args.mesh_path:
        shutil.copy2(produced, args.mesh_path)
from .commands import render_argv_template

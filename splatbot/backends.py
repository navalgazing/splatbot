from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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


def segment_with_sam3(input_dir: Path, output_dir: Path, prompt: str) -> None:
    try:
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"SAM3 is not installed or importable: {exc}") from exc

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=prompt)
        mask = best_mask(result.get("masks"), result.get("scores"))
        if mask is None:
            continue
        write_rgba_from_alpha(image_path, mask, output_dir / image_path.with_suffix(".png").name)
    ensure_output_files(output_dir)


def rembg_box_for_first_frame(input_dir: Path) -> tuple[int, int, int, int] | None:
    import tempfile

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Pillow is required for SAM2 box bootstrapping: {exc}") from exc
    first = next(
        (
            path
            for path in sorted(input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ),
        None,
    )
    if first is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_input = Path(tmp) / "input"
        tmp_output = Path(tmp) / "output"
        tmp_input.mkdir()
        tmp_output.mkdir()
        shutil.copy2(first, tmp_input / first.name)
        segment_with_rembg(tmp_input, tmp_output)
        segmented = next(tmp_output.glob("*.png"), None)
        if segmented is None:
            return None
        alpha = Image.open(segmented).convert("RGBA").getchannel("A")
        bbox = alpha.getbbox()
        return bbox


def segment_with_sam2(input_dir: Path, output_dir: Path) -> None:
    try:
        import numpy as np
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"SAM2 is not installed or importable: {exc}") from exc
    checkpoint = os.environ.get("SPLATBOT_SAM2_CHECKPOINT", "")
    config = os.environ.get("SPLATBOT_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    if not checkpoint:
        raise SystemExit("SPLATBOT_SAM2_CHECKPOINT is required for SAM2 segmentation")
    box = rembg_box_for_first_frame(input_dir)
    if box is None:
        raise SystemExit("could not bootstrap a SAM2 tracking box from rembg")
    predictor = build_sam2_video_predictor(config, checkpoint)
    state = predictor.init_state(video_path=str(input_dir))
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        box=np.array(box, dtype=np.float32),
    )
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
        if frame_idx >= len(images):
            continue
        if not object_ids:
            continue
        mask = best_mask(mask_logits[0] > 0)
        if mask is None:
            continue
        write_rgba_from_alpha(images[frame_idx], mask, output_dir / images[frame_idx].with_suffix(".png").name)
    ensure_output_files(output_dir)


def segment_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot segmentation backends.")
    parser.add_argument("--backend", required=True, choices=["sam3", "sam2", "rembg"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default=os.environ.get("SPLATBOT_OBJECT_MASK_PROMPT", "main object"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "sam3":
        segment_with_sam3(args.input, args.output, args.prompt)
    elif args.backend == "sam2":
        segment_with_sam2(args.input, args.output)
    else:
        segment_with_rembg(args.input, args.output)


def pose_main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatbot pose backends.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--input", "--images", dest="input_dir", required=True, type=Path)
    parser.add_argument("--output", "--processed", dest="processed_dir", required=True, type=Path)
    parser.add_argument("--matching-method", default="")
    args = parser.parse_args()
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
        print("dn-splatter depth preparation failed; continuing without generated mono depth", file=sys.stderr)


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
                "--pipeline.model.use-depth-loss",
                "True",
                "--pipeline.model.depth-lambda",
                "0.2",
                "--pipeline.model.depth-loss-type",
                "PearsonDepth",
            ]
        )
    argv.extend(arg for arg in args.extra if arg != "--")
    run(argv)


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


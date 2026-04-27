from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def summarize_job(job_dir: Path) -> dict[str, Any]:
    metrics = load_json(job_dir / "metrics.json")
    quality = load_json(job_dir / "quality_report.json")
    cleanup = metrics.get("ply", {}).get("cleanup", {})
    validation = cleanup.get("validation", {})
    cleaned = metrics.get("ply", {}).get("cleaned", {})
    stages = metrics.get("stages", {})
    duration = sum(
        float(stage.get("duration_seconds", 0.0))
        for stage in stages.values()
        if isinstance(stage, dict)
    )
    issues = quality.get("issues") or []
    warnings = quality.get("warnings") or []
    outside = float(validation.get("outside_candidate_fraction") or 0.0)
    low_support = float(validation.get("low_support_fraction") or 0.0)
    vertices = int(cleaned.get("vertices") or 0)
    score = (
        100.0
        - (25.0 * len(issues))
        - (5.0 * len(warnings))
        - (40.0 * outside)
        - (40.0 * low_support)
    )
    return {
        "job_dir": str(job_dir),
        "job_id": metrics.get("job_id") or job_dir.name,
        "mode": metrics.get("mode"),
        "preset": metrics.get("preset"),
        "passed": quality.get("passed", not issues),
        "score": round(max(0.0, score), 3),
        "issues": issues,
        "warnings": warnings,
        "duration_seconds": round(duration, 3),
        "vertices": vertices,
        "pose_backend": metrics.get("pose_backend"),
        "segmentation_backend": (metrics.get("masks", {}).get("backend") or {}).get("selected"),
        "validation": validation,
        "viewer_ready": (job_dir / "export" / "cleaned_splat.ply").exists()
        or (job_dir / "cleaned_splat.ply").exists(),
        "mesh_ready": any((job_dir / name).exists() for name in ("mesh.glb", "mesh.gltf", "mesh.obj"))
        or any((job_dir / "export" / name).exists() for name in ("mesh.glb", "mesh.gltf", "mesh.obj")),
    }


def build_benchmark_report(job_dirs: list[Path]) -> dict[str, Any]:
    jobs = [summarize_job(path) for path in job_dirs]
    ranked = sorted(jobs, key=lambda item: (item["passed"], item["score"]), reverse=True)
    return {
        "jobs": ranked,
        "best_job_id": ranked[0]["job_id"] if ranked else None,
        "best_score": ranked[0]["score"] if ranked else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Splatbot job outputs using saved metrics.")
    parser.add_argument("job_dir", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_benchmark_report(args.job_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

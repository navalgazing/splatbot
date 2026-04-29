from __future__ import annotations

import html
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from importlib.resources import files
from pathlib import Path

from .config import Settings
from .pipeline import PipelineOutputs

ALLOWED_MESH_EXTENSIONS = {".glb", ".gltf", ".obj"}
VIEWER_ASSET_DIR_NAME = "_viewer_assets"
GAUSSIAN_SPLATS_MODULE_PATH = "gaussian-splats-3d.module.js"
THREE_MODULE_URL = "https://unpkg.com/three@0.165.0/build/three.module.js"
THREE_ADDONS_URL = "https://unpkg.com/three@0.165.0/examples/jsm/"
GAUSSIAN_SPLATS_MODULE_URL = (
    "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/"
    "build/gaussian-splats-3d.module.js"
)
VIEWER_ASSET_INTEGRITY = {
    "three.module.js": "sha384-Qvl1RLjZOCDFOOH2bKcGnaDHMM8MVv3zVtMvhy3juQdiIOs6RgQ/7zYdM1FbpHzI",
    "controls/OrbitControls.js": "sha384-BZPDnhvqQ9HQ5XsmqEusjwpN9TIaiGbsJ5XWilDEuRu9Psbw4ZyYk67DUO0nbP3z",
    "loaders/PLYLoader.js": "sha384-O7ZFYS9EuaDFTByrnLgtegHfTUwTgluDq8VMh7JWJaAnXi6SQCu6YSuTs1voPqsg",
    "loaders/GLTFLoader.js": "sha384-pK8zo1cGi3nIm4Yh/hrezBkXvN2riHD+7kg4agEXeMxDNqiAKGO2Ds20+xjzcYgZ",
    "loaders/OBJLoader.js": "sha384-qOxu19eVIcHchuUw1oqOwdUJBMFRbuIik14tOMzsoBIa4yuIxVZYj4/YeaWokjND",
    GAUSSIAN_SPLATS_MODULE_PATH: "sha384-wn1UKOYaDMuKGmgBbBxjN4akrwQN6pl+kT1KvGuyOOUeBn8bzo5UnnzJTWeljp2C",
}
CORE_MODULE_ASSETS = (
    "three.module.js",
    "controls/OrbitControls.js",
    "loaders/PLYLoader.js",
)
MESH_MODULE_ASSETS = (
    "loaders/GLTFLoader.js",
    "loaders/OBJLoader.js",
)
ALL_VIEWER_ASSETS = tuple(VIEWER_ASSET_INTEGRITY)
VIEWER_ASSET_VERSION = hashlib.sha256(
    "\n".join(f"{path}:{VIEWER_ASSET_INTEGRITY[path]}" for path in sorted(ALL_VIEWER_ASSETS)).encode("ascii")
).hexdigest()[:16]
VIEWER_HTML_VERSION = "filters-on-demand-20260429"
VIEWER_PAGE_VERSION = f"viewer-{VIEWER_ASSET_VERSION}-{VIEWER_HTML_VERSION}"


def cache_busted_viewer_url(url: str) -> str:
    if not url:
        return ""
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={VIEWER_PAGE_VERSION}"


def publish_viewer(settings: Settings, job_id: str, outputs: PipelineOutputs) -> Path:
    ensure_viewer_assets(settings.public_results_dir)
    target = safe_result_dir(settings.public_results_dir, job_id)
    tmp_target = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp_target, ignore_errors=True)
    tmp_target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(outputs.cleaned_ply, tmp_target / "cleaned_splat.ply")
    write_viewer_point_cloud(outputs.cleaned_ply, tmp_target / "viewer_points.ply")
    mesh_name = None
    if outputs.mesh_path is not None and outputs.mesh_path.exists():
        if outputs.mesh_path.suffix.lower() not in ALLOWED_MESH_EXTENSIONS:
            raise ValueError(f"unsupported viewer mesh extension: {outputs.mesh_path.suffix}")
        mesh_name = outputs.mesh_path.name
        shutil.copy2(outputs.mesh_path, tmp_target / mesh_name)
    has_preview = outputs.preview_mp4 is not None and outputs.preview_mp4.exists()
    if has_preview and outputs.preview_mp4 is not None:
        shutil.copy2(outputs.preview_mp4, tmp_target / "turntable.mp4")
    if outputs.metrics_path is not None and outputs.metrics_path.exists():
        shutil.copy2(outputs.metrics_path, tmp_target / "metrics.json")
    has_quality_report = outputs.quality_report_path is not None and outputs.quality_report_path.exists()
    if has_quality_report and outputs.quality_report_path is not None:
        shutil.copy2(outputs.quality_report_path, tmp_target / "quality_report.json")
    if outputs.candidate_report_path is not None and outputs.candidate_report_path.exists():
        shutil.copy2(outputs.candidate_report_path, tmp_target / "candidate_report.json")
    (tmp_target / "index.html").write_text(
        render_viewer_html(
            job_id,
            has_preview=has_preview,
            mesh_name=mesh_name,
            has_quality_report=has_quality_report,
        ),
        encoding="utf-8",
    )
    replace_result_dir(tmp_target, target)
    return target / "index.html"


def ensure_viewer_assets(public_results_dir: Path) -> None:
    asset_root = public_results_dir / VIEWER_ASSET_DIR_NAME
    asset_root.mkdir(parents=True, exist_ok=True)
    versioned_asset_root = asset_root / VIEWER_ASSET_VERSION
    versioned_asset_root.mkdir(parents=True, exist_ok=True)
    package_assets = files("splatbot").joinpath("viewer_assets")
    for relative_path in ALL_VIEWER_ASSETS:
        source = package_assets.joinpath(*relative_path.split("/"))
        for target_root in (asset_root, versioned_asset_root):
            copy_viewer_asset(source, target_root / relative_path)


def copy_viewer_asset(source, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.tmp-",
                dir=target.parent,
                delete=False,
            ) as dst:
                tmp_target = Path(dst.name)
                with source.open("rb") as src:
                    shutil.copyfileobj(src, dst)
            tmp_target.chmod(0o644)
            tmp_target.replace(target)
            target.chmod(0o644)
        finally:
            if tmp_target is not None:
                tmp_target.unlink(missing_ok=True)


def replace_result_dir(tmp_target: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.old-{os.getpid()}")
    shutil.rmtree(backup, ignore_errors=True)
    try:
        if target.exists():
            target.replace(backup)
        tmp_target.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def safe_result_dir(root: Path, job_id: str) -> Path:
    root = root.resolve()
    target = (root / job_id).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"invalid job id for result path: {job_id}")
    return target


def render_viewer_html(
    job_id: str,
    has_preview: bool = True,
    mesh_name: str | None = None,
    has_quality_report: bool = False,
) -> str:
    title = f"Splatbot Job {job_id}"
    preview_html = (
        '<video controls playsinline src="turntable.mp4"></video>\n'
        '      <a href="turntable.mp4" download>Download preview video</a>'
        if has_preview
        else ""
    )
    mesh_button = '<button id="mesh-button" type="button">Use mesh view</button>' if mesh_name else ""
    mesh_download = f'<a href="{html.escape(mesh_name)}" download>Download mesh</a>' if mesh_name else ""
    report_download = '<a href="quality_report.json" download>Download quality report</a>' if has_quality_report else ""
    module_preloads = ""
    metadata = {
        "job_id": job_id,
        "mesh": mesh_name,
        "has_quality_report": has_quality_report,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex, nofollow">
  <meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="pragma" content="no-cache">
  <meta http-equiv="expires" content="0">
  <title>{html.escape(title)}</title>
{module_preloads}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111827;
      color: #f9fafb;
    }}
    main {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
    }}
    #viewport {{
      min-height: 100vh;
      background: #0b1020;
      position: relative;
    }}
    #fallback-viewport {{
      display: none;
      position: absolute;
      inset: 0;
    }}
    #splat-viewport {{
      display: none;
      position: absolute;
      inset: 0;
    }}
    #mesh-viewport {{
      display: none;
      position: absolute;
      inset: 0;
    }}
    canvas {{ display: block; width: 100%; height: 100%; }}
    aside {{
      border-left: 1px solid #2f3748;
      background: #151b2b;
      padding: 18px;
      overflow: auto;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.25;
      margin: 0 0 16px;
      overflow-wrap: anywhere;
    }}
    a {{
      display: block;
      color: #93c5fd;
      text-decoration: none;
      padding: 10px 0;
      border-top: 1px solid #2f3748;
    }}
    button {{
      width: 100%;
      border: 1px solid #374151;
      background: #1f2937;
      color: #f9fafb;
      padding: 10px 12px;
      margin: 0 0 12px;
      cursor: pointer;
      text-align: left;
    }}
    button:hover {{ background: #263244; }}
    button:disabled {{
      opacity: 0.52;
      cursor: not-allowed;
    }}
    video {{
      width: 100%;
      margin: 12px 0 16px;
      background: #000;
    }}
    .filter-panel {{
      border-top: 1px solid #2f3748;
      margin: 4px 0 14px;
      padding: 14px 0 2px;
    }}
    .filter-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 8px;
      font-size: 13px;
      line-height: 1.3;
    }}
    .filter-header strong {{
      font-size: 13px;
      font-weight: 650;
    }}
    #filter-counts {{
      color: #cbd5e1;
      font-size: 12px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .filter-control {{
      display: block;
      padding: 8px 0;
      color: #e5e7eb;
      font-size: 12px;
    }}
    .filter-label-row {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
    }}
    .filter-control output {{
      color: #bfdbfe;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .filter-control input {{
      width: 100%;
      margin: 0;
      accent-color: #60a5fa;
    }}
    .filter-control input:disabled {{
      opacity: 0.52;
      cursor: not-allowed;
    }}
    #filter-reset {{
      margin: 10px 0 0;
      text-align: center;
    }}
    .status {{
      position: absolute;
      left: 16px;
      bottom: 16px;
      max-width: min(520px, calc(100% - 32px));
      padding: 10px 12px;
      background: rgba(17, 24, 39, 0.86);
      border: 1px solid rgba(255, 255, 255, 0.16);
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 780px) {{
      main {{ grid-template-columns: 1fr; }}
      #viewport {{ min-height: 68vh; }}
      aside {{ border-left: 0; border-top: 1px solid #2f3748; }}
    }}
  </style>
</head>
<body>
  <main>
    <section id="viewport">
      <div id="splat-viewport"></div>
      <div id="fallback-viewport"></div>
      <div id="mesh-viewport"></div>
      <div class="status" id="status">Loading Gaussian splat scene...</div>
    </section>
    <aside>
      <h1>{html.escape(title)}</h1>
      <button id="splat-button" type="button">Use full splat view</button>
      {mesh_button}
      <button id="fallback-button" type="button">Use point preview</button>
      <div class="filter-panel" id="filter-panel" aria-label="Splat filters">
        <div class="filter-header">
          <strong>Filters</strong>
          <span id="filter-counts">0 kept / 0 hidden</span>
        </div>
        <button id="filter-load" type="button">Load filter sliders</button>
        <label class="filter-control" for="filter-opacity">
          <span class="filter-label-row">
            <span>Min opacity</span>
            <output id="filter-opacity-value" for="filter-opacity">2%</output>
          </span>
          <input id="filter-opacity" type="range" min="0" max="1" step="0.01" value="0.02" disabled>
        </label>
        <label class="filter-control" for="filter-scale">
          <span class="filter-label-row">
            <span>Max scale</span>
            <output id="filter-scale-value" for="filter-scale">all</output>
          </span>
          <input id="filter-scale" type="range" min="0" max="1" step="0.01" value="1" disabled>
        </label>
        <label class="filter-control" for="filter-anisotropy">
          <span class="filter-label-row">
            <span>Max anisotropy</span>
            <output id="filter-anisotropy-value" for="filter-anisotropy">all</output>
          </span>
          <input id="filter-anisotropy" type="range" min="1" max="1" step="0.01" value="1" disabled>
        </label>
        <button id="filter-reset" type="button" disabled>Reset filters</button>
      </div>
      {preview_html}
      <a href="cleaned_splat.ply" download>Download PLY</a>
      {mesh_download}
      {report_download}
    </aside>
  </main>
  <script type="importmap">
    {{
      "imports": {{
        "three": "{THREE_MODULE_URL}",
        "three/addons/": "{THREE_ADDONS_URL}"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ PLYLoader }} from "three/addons/loaders/PLYLoader.js";
    import * as GaussianSplats3D from "{GAUSSIAN_SPLATS_MODULE_URL}";

    const container = document.getElementById("viewport");
    const splatContainer = document.getElementById("splat-viewport");
    const fallbackContainer = document.getElementById("fallback-viewport");
    const meshContainer = document.getElementById("mesh-viewport");
    const status = document.getElementById("status");
    const splatButton = document.getElementById("splat-button");
    const fallbackButton = document.getElementById("fallback-button");
    const meshButton = document.getElementById("mesh-button");
    const filterCounts = document.getElementById("filter-counts");
    const opacityInput = document.getElementById("filter-opacity");
    const scaleInput = document.getElementById("filter-scale");
    const anisotropyInput = document.getElementById("filter-anisotropy");
    const opacityValue = document.getElementById("filter-opacity-value");
    const scaleValue = document.getElementById("filter-scale-value");
    const anisotropyValue = document.getElementById("filter-anisotropy-value");
    const filterLoad = document.getElementById("filter-load");
    const filterReset = document.getElementById("filter-reset");
    const meshName = {json.dumps(mesh_name)};
    const DEFAULT_MIN_OPACITY_ALPHA = 5;
    const DEFAULT_MIN_OPACITY = DEFAULT_MIN_OPACITY_ALPHA / 255;
    const SPLAT_OFFSET = {{
      SCALE0: 3,
      SCALE1: 4,
      SCALE2: 5,
      OPACITY: 13,
    }};
    let splatViewer = null;
    let splatViewerGeneration = 0;
    let splatArray = null;
    let splatMetrics = null;
    let filterDataLoading = false;
    let filterApplyRunning = false;
    let filterApplyQueued = false;
    let filterDebounce = null;
    let fallbackStarted = false;
    let meshStarted = false;
    let activeView = null;

    function disposeSplatViewer() {{
      if (!splatViewer) return;
      splatViewerGeneration++;
      try {{ splatViewer.dispose(); }} catch (error) {{ console.warn(error); }}
      splatViewer = null;
    }}

    function setFilterControlsEnabled(enabled) {{
      for (const element of [opacityInput, scaleInput, anisotropyInput, filterReset]) {{
        element.disabled = !enabled;
      }}
    }}

    function setFilterLoadState(text, disabled) {{
      filterLoad.textContent = text;
      filterLoad.disabled = disabled;
    }}

    function formatInteger(value) {{
      return Math.round(value).toLocaleString();
    }}

    function formatCompactNumber(value) {{
      if (!Number.isFinite(value)) return "all";
      let text;
      const absolute = Math.abs(value);
      if (absolute >= 100) {{
        text = value.toFixed(0);
      }} else if (absolute >= 10) {{
        text = value.toFixed(1);
      }} else if (absolute >= 1) {{
        text = value.toFixed(2);
      }} else if (absolute > 0) {{
        text = value.toPrecision(2);
      }} else {{
        text = "0";
      }}
      return text.replace(/(\\.\\d*?[1-9])0+$/, "$1").replace(/\\.0+$/, "");
    }}

    function niceRangeMax(value, fallback = 1) {{
      if (!Number.isFinite(value) || value <= 0) return fallback;
      const exponent = Math.floor(Math.log10(value));
      const magnitude = 10 ** exponent;
      const fraction = value / magnitude;
      const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
      return niceFraction * magnitude;
    }}

    function setRange(input, min, max, value) {{
      input.min = String(min);
      input.max = String(max);
      input.step = String(Math.max((max - min) / 500, Number.EPSILON));
      input.value = String(value);
    }}

    function isSliderAtMax(input) {{
      const max = Number(input.max);
      const step = Number(input.step) || 0;
      return Number(input.value) >= max - step * 0.5;
    }}

    function currentFilters() {{
      return {{
        minOpacityAlpha: Math.round(Number(opacityInput.value) * 255),
        maxScale: isSliderAtMax(scaleInput) ? Number.POSITIVE_INFINITY : Number(scaleInput.value),
        maxAnisotropy: isSliderAtMax(anisotropyInput) ? Number.POSITIVE_INFINITY : Number(anisotropyInput.value),
      }};
    }}

    function updateFilterOutputs() {{
      const opacity = Number(opacityInput.value);
      opacityValue.textContent = `${{Math.round(opacity * 100)}}%`;
      scaleValue.textContent = isSliderAtMax(scaleInput) ? "all" : formatCompactNumber(Number(scaleInput.value));
      anisotropyValue.textContent = isSliderAtMax(anisotropyInput) ? "all" : formatCompactNumber(Number(anisotropyInput.value));
    }}

    function updateFilterCounts(kept, hidden) {{
      filterCounts.textContent = `${{formatInteger(kept)}} kept / ${{formatInteger(hidden)}} hidden`;
    }}

    function resetFilterInputs() {{
      opacityInput.value = String(DEFAULT_MIN_OPACITY);
      scaleInput.value = scaleInput.max;
      anisotropyInput.value = anisotropyInput.max;
      updateFilterOutputs();
    }}

    function countFilterMatches(filters) {{
      let kept = 0;
      const count = splatMetrics.count;
      for (let index = 0; index < count; index++) {{
        if (
          splatMetrics.opacity[index] >= filters.minOpacityAlpha &&
          splatMetrics.maxScale[index] <= filters.maxScale &&
          splatMetrics.anisotropy[index] <= filters.maxAnisotropy
        ) {{
          kept++;
        }}
      }}
      return kept;
    }}

    function computeSplatMetrics(parsedSplatArray) {{
      const count = parsedSplatArray.splatCount || parsedSplatArray.splats.length;
      const opacity = new Uint8Array(count);
      const maxScale = new Float32Array(count);
      const anisotropy = new Float32Array(count);
      let maxScaleValue = 0;
      let maxFiniteAnisotropyValue = 1;

      for (let index = 0; index < count; index++) {{
        const splat = parsedSplatArray.splats[index];
        const scale0 = Number.isFinite(splat[SPLAT_OFFSET.SCALE0]) && splat[SPLAT_OFFSET.SCALE0] > 0 ? splat[SPLAT_OFFSET.SCALE0] : 0;
        const scale1 = Number.isFinite(splat[SPLAT_OFFSET.SCALE1]) && splat[SPLAT_OFFSET.SCALE1] > 0 ? splat[SPLAT_OFFSET.SCALE1] : 0;
        const scale2 = Number.isFinite(splat[SPLAT_OFFSET.SCALE2]) && splat[SPLAT_OFFSET.SCALE2] > 0 ? splat[SPLAT_OFFSET.SCALE2] : 0;
        const largestScale = Math.max(scale0, scale1, scale2);
        const positiveScales = [scale0, scale1, scale2].filter(value => value > 0);
        const smallestScale = positiveScales.length ? Math.min(...positiveScales) : 0;
        const scaleRatio = smallestScale > 0 ? largestScale / smallestScale : (largestScale > 0 ? Number.POSITIVE_INFINITY : 1);

        opacity[index] = Math.max(0, Math.min(255, Math.round(splat[SPLAT_OFFSET.OPACITY] || 0)));
        maxScale[index] = largestScale;
        anisotropy[index] = scaleRatio;
        if (Number.isFinite(largestScale) && largestScale > maxScaleValue) maxScaleValue = largestScale;
        if (Number.isFinite(scaleRatio) && scaleRatio > maxFiniteAnisotropyValue) maxFiniteAnisotropyValue = scaleRatio;
      }}

      return {{
        count,
        opacity,
        maxScale,
        anisotropy,
        maxScaleValue,
        maxFiniteAnisotropyValue,
      }};
    }}

    function configureFilterControls() {{
      const scaleMax = niceRangeMax(splatMetrics.maxScaleValue, 1);
      const anisotropyMax = Math.max(1, niceRangeMax(splatMetrics.maxFiniteAnisotropyValue, 1));
      setRange(scaleInput, 0, scaleMax, scaleMax);
      setRange(anisotropyInput, 1, anisotropyMax, anisotropyMax);
      resetFilterInputs();
      setFilterControlsEnabled(true);
      setFilterLoadState("Filters ready", true);
      const kept = countFilterMatches(currentFilters());
      updateFilterCounts(kept, splatMetrics.count - kept);
    }}

    function buildFilteredSplatArray(filters) {{
      const keptSplats = [];
      const count = splatMetrics.count;
      for (let index = 0; index < count; index++) {{
        if (
          splatMetrics.opacity[index] >= filters.minOpacityAlpha &&
          splatMetrics.maxScale[index] <= filters.maxScale &&
          splatMetrics.anisotropy[index] <= filters.maxAnisotropy
        ) {{
          keptSplats.push(splatArray.splats[index]);
        }}
      }}
      return {{
        sphericalHarmonicsDegree: splatArray.sphericalHarmonicsDegree || 0,
        sphericalHarmonicsCount: splatArray.sphericalHarmonicsCount || 0,
        componentCount: splatArray.componentCount || 14,
        defaultSphericalHarmonics: splatArray.defaultSphericalHarmonics || [],
        splats: keptSplats,
        splatCount: keptSplats.length,
      }};
    }}

    async function rebuildFilteredSplatScene(initial = false) {{
      if (!splatViewer || !splatArray || !splatMetrics) return;
      const viewer = splatViewer;
      const viewerGeneration = splatViewerGeneration;
      const filters = currentFilters();
      status.textContent = initial ? "Building Gaussian splat scene..." : "Applying splat filters...";
      await new Promise(resolve => setTimeout(resolve, 0));
      const filteredSplatArray = buildFilteredSplatArray(filters);
      const kept = filteredSplatArray.splatCount;
      const hidden = splatMetrics.count - kept;
      updateFilterCounts(kept, hidden);
      const splatBuffer = GaussianSplats3D.SplatBuffer.generateFromUncompressedSplatArrays(
        [filteredSplatArray],
        0,
        0,
        new THREE.Vector3(),
      );
      if (viewer !== splatViewer || viewerGeneration !== splatViewerGeneration) return;
      await viewer.addSplatBuffers(
        [splatBuffer],
        [{{ splatAlphaRemovalThreshold: 0 }}],
        true,
        false,
        false,
        true,
        false,
        true,
      );
      if (viewer !== splatViewer || viewerGeneration !== splatViewerGeneration) return;
      status.textContent = kept > 0 ? "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan." : "No splats match filters.";
    }}

    async function applyCurrentSplatFilters(initial = false) {{
      if (filterApplyRunning) {{
        filterApplyQueued = true;
        return;
      }}
      filterApplyRunning = true;
      try {{
        do {{
          filterApplyQueued = false;
          await rebuildFilteredSplatScene(initial);
          initial = false;
        }} while (filterApplyQueued);
      }} catch (error) {{
        console.error(error);
        status.textContent = "Could not apply splat filters.";
      }} finally {{
        filterApplyRunning = false;
      }}
    }}

    function scheduleFilterApply() {{
      if (!splatArray || !splatMetrics) return;
      updateFilterOutputs();
      window.clearTimeout(filterDebounce);
      filterDebounce = window.setTimeout(() => {{
        void applyCurrentSplatFilters(false);
      }}, 180);
    }}

    async function loadFilterableSplatData() {{
      const response = await fetch("cleaned_splat.ply");
      if (!response.ok) throw new Error(`Could not fetch cleaned_splat.ply: ${{response.status}}`);
      const plyFileData = await response.arrayBuffer();
      const parsed = GaussianSplats3D.PlyParser.parseToUncompressedSplatArray(plyFileData, 0);
      if (!parsed || !parsed.splats || !parsed.splats.length) {{
        throw new Error("PLY parser did not return Gaussian splat data.");
      }}
      splatArray = parsed;
      splatMetrics = computeSplatMetrics(parsed);
      configureFilterControls();
    }}

    async function prepareFilterableSplatData() {{
      if (splatArray && splatMetrics) return true;
      if (filterDataLoading) return false;
      filterDataLoading = true;
      setFilterLoadState("Loading filters...", true);
      try {{
        if (activeView === "splat") status.textContent = "Preparing splat filters...";
        await loadFilterableSplatData();
        if (activeView === "splat") status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
        return true;
      }} catch (error) {{
        console.error(error);
        filterCounts.textContent = "filters unavailable";
        setFilterLoadState("Retry filter sliders", false);
        if (activeView === "splat") status.textContent = "Could not prepare splat filters.";
        return false;
      }} finally {{
        filterDataLoading = false;
      }}
    }}

    async function loadSplatSceneWithoutFilters() {{
      setFilterControlsEnabled(false);
      await splatViewer.addSplatScene("cleaned_splat.ply", {{
        format: GaussianSplats3D.SceneFormat.Ply,
        splatAlphaRemovalThreshold: DEFAULT_MIN_OPACITY_ALPHA,
        showLoadingUI: true,
        progressiveLoad: true,
      }});
    }}

    async function startSplatViewer() {{
      activeView = "splat";
      splatContainer.style.display = "block";
      fallbackContainer.style.display = "none";
      meshContainer.style.display = "none";
      if (splatViewer) {{
        status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
        return;
      }}
      status.textContent = "Loading full Gaussian splat scene...";
      try {{
        splatViewer = new GaussianSplats3D.Viewer({{
          rootElement: splatContainer,
          cameraUp: [0, 0, 1],
          initialCameraPosition: [1.4, -2.0, 1.2],
          initialCameraLookAt: [0, 0, 0],
          sharedMemoryForWorkers: false,
          gpuAcceleratedSort: false,
          halfPrecisionCovariancesOnGPU: true,
          integerBasedSort: false,
          sphericalHarmonicsDegree: 0,
          showLoadingUI: true,
        }});
        splatViewerGeneration++;
        await loadSplatSceneWithoutFilters();
        if (splatArray && splatMetrics) setFilterControlsEnabled(true);
        splatViewer.start();
        status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
      }} catch (error) {{
        console.error(error);
        disposeSplatViewer();
        status.textContent = "Full splat renderer failed. Showing point preview.";
        startPointPreview();
      }}
    }}

    function startPointPreview() {{
      activeView = "point";
      disposeSplatViewer();
      splatContainer.style.display = "none";
      fallbackContainer.style.display = "block";
      meshContainer.style.display = "none";
      if (fallbackStarted) {{
        status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
        return;
      }}
      fallbackStarted = true;
      status.textContent = "Loading point preview...";

      const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    camera.position.set(0, 0.8, 2.8);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      fallbackContainer.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 1.8));
    const grid = new THREE.GridHelper(2, 10, 0x334155, 0x1f2937);
    grid.position.y = -0.7;
    scene.add(grid);

    function resize() {{
        const width = fallbackContainer.clientWidth || container.clientWidth;
        const height = fallbackContainer.clientHeight || container.clientHeight;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    }}
    window.addEventListener("resize", resize);
    resize();

      new PLYLoader().load("viewer_points.ply", geometry => {{
      geometry.computeBoundingSphere();
      const material = new THREE.PointsMaterial({{ size: 0.006, vertexColors: geometry.hasAttribute("color") }});
      if (!geometry.hasAttribute("color")) material.color.set(0xf8fafc);
      const points = new THREE.Points(geometry, material);
      scene.add(points);
      const sphere = geometry.boundingSphere;
      if (sphere) {{
        controls.target.copy(sphere.center);
        const radius = Math.max(sphere.radius, 0.4);
        camera.position.copy(sphere.center).add(new THREE.Vector3(radius * 1.4, radius * 0.7, radius * 1.8));
        camera.near = radius / 100;
        camera.far = radius * 100;
        camera.updateProjectionMatrix();
      }}
      status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
    }}, undefined, error => {{
      console.error(error);
      status.textContent = "Could not load the PLY in-browser. Use the download link.";
    }});

    function animate() {{
      requestAnimationFrame(animate);
      if (activeView === "point") {{
        controls.update();
        renderer.render(scene, camera);
      }}
    }}
    animate();
    }}

    async function startMeshPreview() {{
      if (!meshName) return;
      activeView = "mesh";
      disposeSplatViewer();
      splatContainer.style.display = "none";
      fallbackContainer.style.display = "none";
      meshContainer.style.display = "block";
      if (meshStarted) {{
        status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
        return;
      }}
      meshStarted = true;
      status.textContent = "Loading mesh...";

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b1020);
      const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
      camera.position.set(0, 0.8, 2.8);
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      meshContainer.appendChild(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 1.8));
      const key = new THREE.DirectionalLight(0xffffff, 1.5);
      key.position.set(2, -3, 4);
      scene.add(key);

      function resize() {{
        const width = meshContainer.clientWidth || container.clientWidth;
        const height = meshContainer.clientHeight || container.clientHeight;
        camera.aspect = width / Math.max(height, 1);
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      }}
      window.addEventListener("resize", resize);
      resize();

      function frameObject(object) {{
        const box = new THREE.Box3().setFromObject(object);
        const sphere = box.getBoundingSphere(new THREE.Sphere());
        if (sphere.radius > 0) {{
          controls.target.copy(sphere.center);
          camera.position.copy(sphere.center).add(new THREE.Vector3(sphere.radius * 1.6, sphere.radius * 0.8, sphere.radius * 1.8));
          camera.near = sphere.radius / 100;
          camera.far = sphere.radius * 100;
          camera.updateProjectionMatrix();
        }}
      }}

      const extension = meshName.split(".").pop().toLowerCase();
      const onLoad = object => {{
        const root = object.scene || object;
        scene.add(root);
        frameObject(root);
        status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
      }};
      const onError = error => {{
        console.error(error);
        status.textContent = "Could not load the mesh in-browser. Use the download link.";
      }};
      if (extension === "glb" || extension === "gltf") {{
        const {{ GLTFLoader }} = await import("three/addons/loaders/GLTFLoader.js");
        new GLTFLoader().load(meshName, onLoad, undefined, onError);
      }} else if (extension === "obj") {{
        const {{ OBJLoader }} = await import("three/addons/loaders/OBJLoader.js");
        new OBJLoader().load(meshName, onLoad, undefined, onError);
      }} else {{
        status.textContent = "Mesh format is available for download but not previewed here.";
      }}

      function animate() {{
        requestAnimationFrame(animate);
        if (activeView === "mesh") {{
          controls.update();
          renderer.render(scene, camera);
        }}
      }}
      animate();
    }}

    splatButton.addEventListener("click", startSplatViewer);
    if (meshButton) meshButton.addEventListener("click", startMeshPreview);
    fallbackButton.addEventListener("click", startPointPreview);
    filterLoad.addEventListener("click", async () => {{
      if (activeView !== "splat") await startSplatViewer();
      await prepareFilterableSplatData();
    }});
    for (const input of [opacityInput, scaleInput, anisotropyInput]) {{
      input.addEventListener("input", scheduleFilterApply);
    }}
    filterReset.addEventListener("click", () => {{
      resetFilterInputs();
      window.clearTimeout(filterDebounce);
      void applyCurrentSplatFilters(false);
    }});
    startSplatViewer();
  </script>
  <script type="application/json" id="metadata">{json.dumps(metadata)}</script>
</body>
</html>
"""


PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def write_viewer_point_cloud(src: Path, dest: Path) -> None:
    with src.open("rb") as file:
        header_bytes = read_ply_header(file)
    if header_bytes is None:
        shutil.copy2(src, dest)
        return
    header = header_bytes.decode("ascii", errors="replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        shutil.copy2(src, dest)
        return

    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3:
            try:
                vertex_count = int(parts[2])
            except ValueError:
                shutil.copy2(src, dest)
                return
            in_vertex = True
            continue
        if parts[:1] == ["element"] and parts[1:2] != ["vertex"]:
            in_vertex = False
        if in_vertex and parts[:1] == ["property"] and len(parts) == 3:
            properties.append((parts[1], parts[2]))

    names = [name for _, name in properties]
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    if not vertex_count or any(name not in names for name in required):
        shutil.copy2(src, dest)
        return

    try:
        vertex_struct = struct.Struct("<" + "".join(PLY_SCALAR_FORMATS[kind] for kind, _ in properties))
    except KeyError:
        shutil.copy2(src, dest)
        return

    indexes = {name: names.index(name) for name in required}
    out_header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    out_vertex = struct.Struct("<fffBBB")

    c0 = 0.28209479177387814

    def channel(value: float) -> int:
        if not math.isfinite(value):
            return 0
        return max(0, min(255, round((0.5 + c0 * value) * 255)))

    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with src.open("rb") as source, tmp.open("wb") as target:
            source.seek(len(header_bytes))
            target.write(out_header)
            for _ in range(vertex_count):
                chunk = source.read(vertex_struct.size)
                if len(chunk) != vertex_struct.size:
                    raise EOFError("truncated vertex data")
                values = vertex_struct.unpack(chunk)
                target.write(
                    out_vertex.pack(
                        float(values[indexes["x"]]),
                        float(values[indexes["y"]]),
                        float(values[indexes["z"]]),
                        channel(float(values[indexes["f_dc_0"]])),
                        channel(float(values[indexes["f_dc_1"]])),
                        channel(float(values[indexes["f_dc_2"]])),
                    )
                )
    except (EOFError, struct.error, ValueError):
        tmp.unlink(missing_ok=True)
        shutil.copy2(src, dest)
        return
    tmp.replace(dest)


def read_ply_header(file) -> bytes | None:
    header = bytearray()
    while line := file.readline():
        header.extend(line)
        if line.rstrip(b"\r\n") == b"end_header":
            return bytes(header)
        if len(header) > 1024 * 1024:
            return None
    return None

from __future__ import annotations

import html
import json
import shutil
import struct
from pathlib import Path

from .config import Settings
from .pipeline import PipelineOutputs


def publish_viewer(settings: Settings, job_id: str, outputs: PipelineOutputs) -> Path:
    target = settings.public_results_dir / job_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outputs.cleaned_ply, target / "cleaned_splat.ply")
    write_viewer_point_cloud(outputs.cleaned_ply, target / "viewer_points.ply")
    has_preview = outputs.preview_mp4 is not None and outputs.preview_mp4.exists()
    if has_preview and outputs.preview_mp4 is not None:
        shutil.copy2(outputs.preview_mp4, target / "turntable.mp4")
    (target / "index.html").write_text(render_viewer_html(job_id, has_preview=has_preview), encoding="utf-8")
    return target / "index.html"


def render_viewer_html(job_id: str, has_preview: bool = True) -> str:
    title = f"Splatbot Job {job_id}"
    preview_html = (
        '<video controls playsinline src="turntable.mp4"></video>\n'
        '      <a href="turntable.mp4" download>Download preview video</a>'
        if has_preview
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
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
    video {{
      width: 100%;
      margin: 12px 0 16px;
      background: #000;
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
      <div id="fallback-viewport"></div>
      <div class="status" id="status">Loading Gaussian splat scene...</div>
    </section>
    <aside>
      <h1>{html.escape(title)}</h1>
      <button id="fallback-button" type="button">Use point preview</button>
      {preview_html}
      <a href="cleaned_splat.ply" download>Download PLY</a>
    </aside>
  </main>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.165.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.165.0/examples/jsm/"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ PLYLoader }} from "three/addons/loaders/PLYLoader.js";
    import * as GaussianSplats3D from "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js";

    const container = document.getElementById("viewport");
    const fallbackContainer = document.getElementById("fallback-viewport");
    const status = document.getElementById("status");
    const fallbackButton = document.getElementById("fallback-button");
    let splatViewer = null;
    let fallbackStarted = false;

    async function startSplatViewer() {{
      try {{
        splatViewer = new GaussianSplats3D.Viewer({{
          rootElement: container,
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
        await splatViewer.addSplatScene("cleaned_splat.ply", {{
          format: GaussianSplats3D.SceneFormat.Ply,
          splatAlphaRemovalThreshold: 5,
          showLoadingUI: true,
          progressiveLoad: true,
        }});
        splatViewer.start();
        status.textContent = "Drag to orbit. Scroll or pinch to zoom. Right-drag to pan.";
      }} catch (error) {{
        console.error(error);
        status.textContent = "Full splat renderer failed. Showing point preview.";
        startPointPreview();
      }}
    }}

    function startPointPreview() {{
      if (fallbackStarted) return;
      fallbackStarted = true;
      if (splatViewer) {{
        try {{ splatViewer.dispose(); }} catch (error) {{ console.warn(error); }}
      }}
      fallbackContainer.style.display = "block";
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
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
    }}

    fallbackButton.addEventListener("click", startPointPreview);
    startSplatViewer();
  </script>
  <script type="application/json" id="metadata">{json.dumps({"job_id": job_id})}</script>
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
    data = src.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end == -1:
        shutil.copy2(src, dest)
        return
    header_end += len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        shutil.copy2(src, dest)
        return

    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3:
            vertex_count = int(parts[2])
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
        return max(0, min(255, round((0.5 + c0 * value) * 255)))

    with dest.open("wb") as file:
        file.write(out_header)
        offset = header_end
        for _ in range(vertex_count):
            values = vertex_struct.unpack_from(data, offset)
            offset += vertex_struct.size
            file.write(
                out_vertex.pack(
                    float(values[indexes["x"]]),
                    float(values[indexes["y"]]),
                    float(values[indexes["z"]]),
                    channel(float(values[indexes["f_dc_0"]])),
                    channel(float(values[indexes["f_dc_1"]])),
                    channel(float(values[indexes["f_dc_2"]])),
                )
            )

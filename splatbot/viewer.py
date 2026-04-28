from __future__ import annotations

import html
import json
import math
import os
import shutil
import struct
from pathlib import Path

from .config import Settings
from .pipeline import PipelineOutputs

ALLOWED_MESH_EXTENSIONS = {".glb", ".gltf", ".obj"}
MODULE_INTEGRITY = {
    "https://unpkg.com/three@0.165.0/build/three.module.js": "sha384-Qvl1RLjZOCDFOOH2bKcGnaDHMM8MVv3zVtMvhy3juQdiIOs6RgQ/7zYdM1FbpHzI",
    "https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js": "sha384-BZPDnhvqQ9HQ5XsmqEusjwpN9TIaiGbsJ5XWilDEuRu9Psbw4ZyYk67DUO0nbP3z",
    "https://unpkg.com/three@0.165.0/examples/jsm/loaders/GLTFLoader.js": "sha384-pK8zo1cGi3nIm4Yh/hrezBkXvN2riHD+7kg4agEXeMxDNqiAKGO2Ds20+xjzcYgZ",
    "https://unpkg.com/three@0.165.0/examples/jsm/loaders/OBJLoader.js": "sha384-qOxu19eVIcHchuUw1oqOwdUJBMFRbuIik14tOMzsoBIa4yuIxVZYj4/YeaWokjND",
    "https://unpkg.com/three@0.165.0/examples/jsm/loaders/PLYLoader.js": "sha384-O7ZFYS9EuaDFTByrnLgtegHfTUwTgluDq8VMh7JWJaAnXi6SQCu6YSuTs1voPqsg",
    "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js": "sha384-wn1UKOYaDMuKGmgBbBxjN4akrwQN6pl+kT1KvGuyOOUeBn8bzo5UnnzJTWeljp2C",
}


def publish_viewer(settings: Settings, job_id: str, outputs: PipelineOutputs) -> Path:
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
    module_preloads = "\n".join(
        f'  <link rel="modulepreload" href="{html.escape(url)}" '
        f'integrity="{html.escape(integrity)}" crossorigin="anonymous">'
        for url, integrity in MODULE_INTEGRITY.items()
    )
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
      <div id="mesh-viewport"></div>
      <div class="status" id="status">Loading Gaussian splat scene...</div>
    </section>
    <aside>
      <h1>{html.escape(title)}</h1>
      <button id="splat-button" type="button">Use splat view</button>
      {mesh_button}
      <button id="fallback-button" type="button">Use point preview</button>
      {preview_html}
      <a href="cleaned_splat.ply" download>Download PLY</a>
      {mesh_download}
      {report_download}
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
    import {{ GLTFLoader }} from "three/addons/loaders/GLTFLoader.js";
    import {{ OBJLoader }} from "three/addons/loaders/OBJLoader.js";
    import {{ PLYLoader }} from "three/addons/loaders/PLYLoader.js";
    import * as GaussianSplats3D from "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js";

    const container = document.getElementById("viewport");
    const fallbackContainer = document.getElementById("fallback-viewport");
    const meshContainer = document.getElementById("mesh-viewport");
    const status = document.getElementById("status");
    const splatButton = document.getElementById("splat-button");
    const fallbackButton = document.getElementById("fallback-button");
    const meshButton = document.getElementById("mesh-button");
    const meshName = {json.dumps(mesh_name)};
    let splatViewer = null;
    let fallbackStarted = false;
    let meshStarted = false;

    async function startSplatViewer() {{
      fallbackContainer.style.display = "none";
      meshContainer.style.display = "none";
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
      meshContainer.style.display = "none";
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

    function startMeshPreview() {{
      if (!meshName) return;
      if (meshStarted) {{
        fallbackContainer.style.display = "none";
        meshContainer.style.display = "block";
        return;
      }}
      meshStarted = true;
      if (splatViewer) {{
        try {{ splatViewer.dispose(); }} catch (error) {{ console.warn(error); }}
      }}
      fallbackContainer.style.display = "none";
      meshContainer.style.display = "block";
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
        new GLTFLoader().load(meshName, onLoad, undefined, onError);
      }} else if (extension === "obj") {{
        new OBJLoader().load(meshName, onLoad, undefined, onError);
      }} else {{
        status.textContent = "Mesh format is available for download but not previewed here.";
      }}

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    }}

    splatButton.addEventListener("click", startSplatViewer);
    if (meshButton) meshButton.addEventListener("click", startMeshPreview);
    fallbackButton.addEventListener("click", startPointPreview);
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

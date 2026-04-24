from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from .config import Settings
from .pipeline import PipelineOutputs


def publish_viewer(settings: Settings, job_id: str, outputs: PipelineOutputs) -> Path:
    target = settings.public_results_dir / job_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outputs.cleaned_ply, target / "cleaned_splat.ply")
    shutil.copy2(outputs.preview_mp4, target / "turntable.mp4")
    (target / "index.html").write_text(render_viewer_html(job_id), encoding="utf-8")
    return target / "index.html"


def render_viewer_html(job_id: str) -> str:
    title = f"Splatbot Job {job_id}"
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
      <div class="status" id="status">Loading point cloud...</div>
    </section>
    <aside>
      <h1>{html.escape(title)}</h1>
      <video controls playsinline src="turntable.mp4"></video>
      <a href="cleaned_splat.ply" download>Download PLY</a>
      <a href="turntable.mp4" download>Download preview video</a>
    </aside>
  </main>
  <script type="module">
    import * as THREE from "https://unpkg.com/three@0.165.0/build/three.module.js";
    import {{ OrbitControls }} from "https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js";
    import {{ PLYLoader }} from "https://unpkg.com/three@0.165.0/examples/jsm/loaders/PLYLoader.js";

    const container = document.getElementById("viewport");
    const status = document.getElementById("status");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    camera.position.set(0, 0.8, 2.8);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x223044, 1.8));
    const grid = new THREE.GridHelper(2, 10, 0x334155, 0x1f2937);
    grid.position.y = -0.7;
    scene.add(grid);

    function resize() {{
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    }}
    window.addEventListener("resize", resize);
    resize();

    new PLYLoader().load("cleaned_splat.ply", geometry => {{
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
  </script>
  <script type="application/json" id="metadata">{json.dumps({"job_id": job_id})}</script>
</body>
</html>
"""

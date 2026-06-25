"""Deferred G-buffer AOVs from the Vulkan renderer, as numpy — labels for free.

The deferred (RasterFirst) renderer writes a full G-buffer every frame. This
reads several attachments back as (H, W, 3) uint8 images: the shaded RGB, world
normals, per-instance segmentation, and albedo — the raw material for synthetic
training data (no path tracer, no manual labelling).

    python vulkan_aovs.py

Needs a build with -DTHREEPP_WITH_VULKAN=ON. Saves aov_<name>.png for each AOV
(and a combined aov_montage.png if Pillow is installed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import threepp as tp

if not tp.HAS_VULKAN:
    print("This threepp build has no Vulkan backend "
          "(configure with -DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

here = os.path.dirname(os.path.abspath(__file__))
W, H = 512, 384

# A headless Vulkan surface. vsync=False suits the off-screen MAILBOX readback.
canvas = tp.Canvas("aovs", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)

scene = tp.Scene()
scene.background = 0x202830

# Three distinct objects so the segmentation AOV shows three distinct ids.
specs = [(0xff5555, 0.45, tp.BoxGeometry(1.1, 1.1, 1.1)),
         (0x55ff66, 0.30, tp.SphereGeometry(0.7, 32, 16)),
         (0x5599ff, 0.60, tp.TorusKnotGeometry(0.45, 0.16))]
for i, (color, rough, geo) in enumerate(specs):
    mat = tp.MeshStandardMaterial()
    mat.color = color
    mat.roughness = rough
    mat.metalness = 0.1
    mesh = tp.Mesh(geo, mat)
    mesh.position.x = (i - 1) * 2.0
    scene.add(mesh)

ground_mat = tp.MeshStandardMaterial()
ground_mat.color = 0x555560
ground = tp.Mesh(tp.PlaneGeometry(40, 40), ground_mat)
ground.position.y = -1.0
ground.rotate_x(-3.14159 / 2)
scene.add(ground)

scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.0))
sun = tp.DirectionalLight(0xffffff, 3.0)
sun.position.set(4, 7, 5)
scene.add(sun)

camera = tp.PerspectiveCamera(50, W / H, 0.1, 100)
camera.position.set(0, 2.5, 6.5)
camera.look_at(0, 0, 0)

names = ["rgb", "normals", "segmentation", "albedo"]
aovs = renderer.render_aovs(scene, camera, names)
for name in names:
    img = aovs[name]
    print(f"{name:14s} {img.shape} {img.dtype}")

# Depth is metric float32 (H, W), distance from the camera in scene units.
depth = renderer.read_depth(scene, camera)
fg = depth[depth < 99.0]  # exclude the far-plane background
print(f"{'depth':14s} {depth.shape} {depth.dtype}  "
      f"foreground {fg.min():.2f}..{fg.max():.2f} units")

# Normalize the metric depth to a grayscale tile for the montage (near = bright).
lo, hi = (float(fg.min()), float(fg.max())) if fg.size else (0.0, 1.0)
dn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
depth_gray = np.repeat((255 * (1.0 - dn)).astype(np.uint8)[:, :, None], 3, axis=2)

try:
    from PIL import Image

    for name in names:
        Image.fromarray(aovs[name]).save(os.path.join(here, f"aov_{name}.png"))
    Image.fromarray(depth_gray).save(os.path.join(here, "aov_depth.png"))
    np.save(os.path.join(here, "aov_depth.npy"), depth)  # the raw metric float32

    tiles = [aovs["rgb"], aovs["normals"], aovs["segmentation"], aovs["albedo"], depth_gray]
    montage = Image.new("RGB", (W * 3, H * 2))
    for idx, tile in enumerate(tiles):
        montage.paste(Image.fromarray(tile), ((idx % 3) * W, (idx // 3) * H))
    montage.save(os.path.join(here, "aov_montage.png"))
    print("wrote aov_*.png, aov_depth.npy and aov_montage.png")
except ImportError:
    print("(install Pillow to save the AOVs as images)")

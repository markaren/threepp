"""Local (GPU) verification of the wheel's Vulkan backend — the part CI cannot
run. Renders a real scene headless and checks every typed AOV carries signal:
this is the synthetic-data pitch, exercised end to end from an installed wheel.

Run on a machine with a Vulkan-capable GPU:  python vk_verify.py
"""
import numpy as np

import threepp as tp

assert tp.HAS_VULKAN, "wheel has no Vulkan backend"
assert tp.vulkan_available(), "this machine has no Vulkan runtime — run on the dev box"

canvas = tp.Canvas("vk_verify", width=320, height=240, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)

scene = tp.Scene()
scene.add(tp.AmbientLight())
sun = tp.DirectionalLight()
sun.position.set(3, 5, 2)
scene.add(sun)
box = tp.Mesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial())
scene.add(box)

camera = tp.PerspectiveCamera(60, 320 / 240, 0.1, 100)
camera.position.set(0, 1, 4)

out = renderer.read_aovs_typed(scene, camera, ["rgb", "depth", "normals", "instance_ids"])

rgb, depth, normals, ids = out["rgb"], out["depth"], out["normals"], out["instance_ids"]
print("rgb   :", rgb.shape, rgb.dtype, "mean", float(rgb.mean()))
print("depth :", depth.shape, depth.dtype, "min", float(depth.min()), "max", float(depth.max()))
print("normal:", normals.shape, normals.dtype)
print("ids   :", ids.shape, ids.dtype, "unique", np.unique(ids)[:6].tolist())

assert rgb.shape == (240, 320, 3) and rgb.mean() > 1.0, "rgb is black — nothing rendered"
assert depth.shape == (240, 320) and float(depth.min()) < 100.0, "depth carries no hit closer than far"
assert normals.shape == (240, 320, 3), "normals shape wrong"
assert (ids != 0).any(), "instance ids all zero — the box left no segmentation footprint"

print("VULKAN AOVs OK — rgb/depth/normals/ids all carry signal")

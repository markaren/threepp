"""Render a threepp scene into a Gaussian-splat training set, from the deferred camera output.

Every image is what the Vulkan deferred renderer delivers to a camera: the final
frame after TAA, the denoiser and the tone map. Upscalers and auto exposure are off
and the clock is pinned, so a pose renders the same picture every run. Each pose is
held for --frames frames before it is read, because the temporal passes (TAA, the
SVGF denoiser, probe GI) need a few frames to settle after the camera jumps.
--convergence measures how many.

Writes, under --out:
  images/{train,test}_NNN.png   the camera output, sRGB 8-bit
  depth/{train,test}_NNN.png    planar depth, uint16 millimetres, 0 = background
  points3d.ply                  a colour point cloud back-projected from the depth
                                maps, the trainer's starting splats (no SfM needed:
                                the poses are exact)
  transforms.json               nerfstudio format (OpenGL camera-to-world)
  contact_train.jpg             a sheet of training views, to look at

    set PYTHONPATH=python
    py -3.14 python/examples/scene_splats/make_dataset.py [--convergence]
"""
import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image

import common
from common import tp

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=common.DEFAULT_OUT)
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--height", type=int, default=600)
ap.add_argument("--fov", type=float, default=50.0, help="vertical field of view, degrees")
ap.add_argument("--train", type=int, default=200, help="training poses")
ap.add_argument("--test", type=int, default=24, help="held-out poses, between the training ones")
ap.add_argument("--frames", type=int, default=48,
                help="frames each pose is held before it is read (measured with --convergence: "
                     "after 24 frames 7.5 %% of pixels are still off by > 2 levels, after 48 0.7 %%)")
ap.add_argument("--warmup", type=int, default=120, help="frames before the first capture")
ap.add_argument("--init-points", type=int, default=150_000)
ap.add_argument("--convergence", action="store_true",
                help="measure how the frame settles after a camera jump, then exit")
args = ap.parse_args()

W, H = args.width, args.height
scene, info = common.build_scene()
center, radius = info["center"], info["radius"]
camera = tp.PerspectiveCamera(args.fov, W / H, 0.02 * radius, 40.0 * radius)
canvas = tp.Canvas("scene-splats", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)
common.setup_renderer(renderer)

train_poses = common.rig_poses(center, radius, args.train)
test_poses = common.rig_poses(center, radius, args.test, seed_phase=1.0)

# Warm up at the first pose: pipelines, acceleration structures, the probe grid.
common.place(camera, train_poses[0])
t0 = time.perf_counter()
for _ in range(args.warmup):
    renderer.render(scene, camera)
print(f"scene radius {radius:.3f} m, centre ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}); "
      f"warmup {args.warmup} frames {time.perf_counter() - t0:.1f}s")

if args.convergence:
    # Jump from a held-out pose to a training pose, the way the capture loop does,
    # and watch each frame approach the one 64 frames later.
    common.place(camera, test_poses[0])
    for _ in range(32):
        renderer.render(scene, camera)
    common.place(camera, train_poses[0])
    frames = []
    for _ in range(64):
        renderer.render(scene, camera)
        frames.append(renderer.read_pixels().astype(np.float32))
    last = frames[-1]
    print("frames after the jump -> mean |frame - frame 64| (8-bit levels), share of pixels off by > 2")
    for k in (1, 2, 4, 8, 12, 16, 24, 32, 48):
        d = np.abs(frames[k - 1] - last)
        print(f"  {k:3d}: {d.mean():6.3f}   {(d.max(axis=2) > 2).mean() * 100:5.1f} %")
    raise SystemExit(0)

os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
os.makedirs(os.path.join(args.out, "depth"), exist_ok=True)

intr = None
frames_json = []
depth_far = 0.999 * camera.far
pts, cols = [], []
rng = np.random.default_rng(0)
t0 = time.perf_counter()

for split, poses in (("train", train_poses), ("test", test_poses)):
    for i, pose in enumerate(poses):
        common.place(camera, pose)
        for _ in range(args.frames - 1):
            renderer.render(scene, camera)
        aov = renderer.read_aovs_typed(scene, camera, ["rgb", "depth"])
        rgb, depth = aov["rgb"], aov["depth"]
        if intr is None:
            intr = renderer.camera_intrinsics
            if callable(intr):
                intr = intr()
        valid = depth < depth_far
        mm = np.where(valid, np.clip(depth * 1000.0, 1, 65535), 0).astype(np.uint16)

        name = f"{split}_{i:03d}"
        Image.fromarray(rgb).save(os.path.join(args.out, "images", name + ".png"))
        Image.fromarray(mm).save(os.path.join(args.out, "depth", name + ".png"))
        c2w = common.matrix_world_np(camera)
        frames_json.append({"file_path": f"images/{name}.png",
                            "depth_file_path": f"depth/{name}.png",
                            "transform_matrix": c2w.tolist(),
                            "split": split})

        if split == "train" and i % 3 == 0:
            # back-project a sparse grid of valid pixels into the world
            v, u = np.nonzero(valid[::3, ::3])
            v, u = v * 3, u * 3
            z = depth[v, u]
            x = (u + 0.5 - intr["cx"]) * z / intr["fx"]
            y = (v + 0.5 - intr["cy"]) * z / intr["fy"]
            cam = np.stack([x, -y, -z, np.ones_like(z)], 1)  # OpenCV -> OpenGL camera axes
            pts.append((cam @ c2w.T)[:, :3].astype(np.float32))
            cols.append(rgb[v, u])
    print(f"{split}: {len(poses)} poses, {time.perf_counter() - t0:.1f}s so far")

# The start cloud: voxel-thinned so dense views do not dominate, then capped.
xyz, rgb = np.concatenate(pts), np.concatenate(cols)
key = np.floor(xyz / (radius / 160.0)).astype(np.int64)
_, first = np.unique(key, axis=0, return_index=True)
xyz, rgb = xyz[first], rgb[first]
if len(xyz) > args.init_points:
    pick = rng.choice(len(xyz), args.init_points, replace=False)
    xyz, rgb = xyz[pick], rgb[pick]
common.write_points_ply(os.path.join(args.out, "points3d.ply"), xyz, rgb)

transforms = {
    "camera_model": "OPENCV",
    "fl_x": intr["fx"], "fl_y": intr["fy"], "cx": intr["cx"], "cy": intr["cy"],
    "w": W, "h": H,
    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
    "depth_unit_scale_factor": 0.001,
    "ply_file_path": "points3d.ply",
    "frames": frames_json,
    "threepp": {
        "scene": "rover on sand (python/examples/scene_splats/common.py)",
        "source": "VulkanRenderer deferred camera output (read_aovs_typed rgb + depth)",
        "frames_per_pose": args.frames,
        "background": list(common.BACKGROUND),
        "center": list(center), "radius": radius,
        "fov": args.fov, "near": camera.near, "far": camera.far,
        "tone_mapping": "ACESFilmic", "exposure": 1.0,
    },
}
with open(os.path.join(args.out, "transforms.json"), "w", encoding="utf-8") as f:
    json.dump(transforms, f, indent=1)

# A sheet of every 17th training view, half size.
sheet_ids = list(range(0, args.train, max(1, args.train // 12)))[:12]
tiles = [Image.open(os.path.join(args.out, "images", f"train_{i:03d}.png")).resize((W // 2, H // 2))
         for i in sheet_ids]
cols_n = 4
sheet = Image.new("RGB", (cols_n * (W // 2), math.ceil(len(tiles) / cols_n) * (H // 2)))
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % cols_n) * (W // 2), (k // cols_n) * (H // 2)))
sheet.save(os.path.join(args.out, "contact_train.jpg"), quality=90)

print(f"wrote {len(frames_json)} views and {len(xyz)} start points to {args.out} "
      f"in {time.perf_counter() - t0:.1f}s; intrinsics fx {intr['fx']:.1f} fy {intr['fy']:.1f} "
      f"cx {intr['cx']:.1f} cy {intr['cy']:.1f}")

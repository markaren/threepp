"""Load trained splats back into threepp and hold them against the scene they came from.

Three checks, each against something exact:
  1. picture: threepp's Vulkan splat pass renders the trained .ply from the
     held-out cameras; PSNR against the deferred camera output there, and
     against gsplat's own render of the same splats (does threepp read the file
     the way gsplat wrote it?)
  2. geometry: splats.bake_surface fuses the splats' median depth into a mesh,
     from the dataset's own cameras; every baked vertex is measured against the
     real surfaces (the sand plane exactly, the rover's triangles densely sampled)
  3. coverage: the share of the rover's VISIBLE surface (seen in a training depth
     map) that the bake reached

Writes compare_sheet.jpg (camera output | gsplat | threepp splats | baked surface
| |threepp - camera| x4) and compare.json next to the splats.

    py -3.14 python/examples/scene_splats/compare.py [--splats <dir>/splats.ply]
"""
import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

import common
from common import tp

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=common.DEFAULT_OUT)
ap.add_argument("--splats", default=None, help="default <data>/splats_7000/splats.ply")
ap.add_argument("--frames", type=int, default=48, help="frames each pose is held, as in the dataset")
ap.add_argument("--voxel", type=float, default=0.02, help="bake voxel size, metres")
ap.add_argument("--bake-every", type=int, default=4, help="bake from every Nth training camera")
args = ap.parse_args()

splats_path = args.splats or os.path.join(args.data, "splats_7000", "splats.ply")
out_dir = os.path.dirname(os.path.abspath(splats_path))
tf = common.load_transforms(args.data)
meta = tf["threepp"]
W, H = tf["w"], tf["h"]
frames = {"train": [f for f in tf["frames"] if f["split"] == "train"],
          "test": [f for f in tf["frames"] if f["split"] == "test"]}

canvas = tp.Canvas("scene-splats-compare", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)
common.setup_renderer(renderer)
camera = tp.PerspectiveCamera(meta["fov"], W / H, meta["near"], meta["far"])


def pose_of(frame):
    """(position, target, up) of a dataset camera, from its OpenGL camera-to-world."""
    m = np.array(frame["transform_matrix"])
    pos = m[:3, 3]
    return pos, pos - m[:3, 2], m[:3, 1]


def place(frame):
    pos, tgt, up = pose_of(frame)
    camera.up.set(*up)
    camera.position.set(*pos)
    camera.look_at(*tgt)
    camera.update_matrix_world(True)


def shoot(scene, frame):
    place(frame)
    for _ in range(args.frames):
        renderer.render(scene, camera)
    return renderer.read_pixels()


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean() / 255.0 ** 2
    return -10 * math.log10(mse + 1e-12)


# --- 1. the picture -----------------------------------------------------------
data = tp.SplatLoader.load_ply(splats_path)
n_splats = len(data)
cloud = tp.SplatCloud(data)
# The file is in the COLMAP frame (+Y down) like every 3DGS .ply; the editor's
# import turns it upright with this same half turn on the node.
cloud.rotation.x = math.pi
splat_scene = tp.Scene()
splat_scene.background = tp.Color(*common.BACKGROUND)
splat_scene.add(cloud)
# The splats were fitted to the tone-mapped output, so they ARE display values:
# a tone curve on top would apply it twice.
renderer.tone_mapping = tp.ToneMapping.NoToneMapping

t0 = time.perf_counter()
rows, psnr_cam, psnr_gsplat = [], [], []
renders = {}
for j, fr in enumerate(frames["test"]):
    img = shoot(splat_scene, fr)
    gt = np.asarray(Image.open(os.path.join(args.data, fr["file_path"])).convert("RGB"))
    psnr_cam.append(psnr(img, gt))
    gs_path = os.path.join(out_dir, f"test_{j:03d}_gsplat.png")
    if os.path.isfile(gs_path):
        psnr_gsplat.append(psnr(img, np.asarray(Image.open(gs_path).convert("RGB"))))
    renders[j] = img
line = (f"threepp splat render ({n_splats} splats) vs camera output, {len(psnr_cam)} held-out views: "
        f"PSNR {np.mean(psnr_cam):.2f} dB (worst {np.min(psnr_cam):.2f})")
if psnr_gsplat:
    line += f"; vs gsplat's render of the same splats: {np.mean(psnr_gsplat):.2f} dB"
print(line)

# --- 2. the bake ----------------------------------------------------------------
bake_frames = frames["train"][::args.bake_every]
poses = []
for fr in bake_frames:
    pos, tgt, up = pose_of(fr)
    poses.append(tp.BakePose(tp.Vector3(*pos), tp.Vector3(*tgt), tp.Vector3(*up), meta["fov"]))
t0 = time.perf_counter()
surface = tp.bake_surface(renderer, cloud, voxel_size=args.voxel, poses=poses,
                          max_depth=6.0 * meta["radius"])
bake_s = time.perf_counter() - t0
V = np.asarray(surface.positions, np.float32).reshape(-1, 3)
print(f"bake: {surface.triangle_count} triangles from {len(poses)} cameras, voxel {args.voxel} m, "
      f"{bake_s:.1f}s")

# The real surfaces: the sand is the plane y = 0; the rover is sampled densely.
scene, info = common.build_scene()
tris = []


def collect(o):
    if isinstance(o, tp.Mesh):
        g = o.geometry
        p = np.asarray(g.get_attribute("position"), np.float64).reshape(-1, 3)
        idx = g.get_index()
        idx = np.arange(len(p)) if idx is None else np.asarray(idx).reshape(-1)
        m = common.matrix_world_np(o)
        pw = p @ m[:3, :3].T + m[:3, 3]
        tris.append(pw[idx.reshape(-1, 3)])


info["rover"].traverse(collect)
T = np.concatenate(tris)                                     # [n, 3, 3]
area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
rng = np.random.default_rng(0)
n_samples = 3_000_000
pick = rng.choice(len(T), n_samples, p=area / area.sum())
r1, r2 = rng.random(n_samples), rng.random(n_samples)
s = np.sqrt(r1)
S = (T[pick, 0] * (1 - s)[:, None] + T[pick, 1] * (s * (1 - r2))[:, None]
     + T[pick, 2] * (s * r2)[:, None]).astype(np.float32)
spacing = math.sqrt(area.sum() / n_samples)

half = 4.0 * meta["radius"]                                  # the sand square's half side
on_sand = (np.abs(V[:, 0]) <= half) & (np.abs(V[:, 2]) <= half)
d_sand = np.where(on_sand, np.abs(V[:, 1]), np.inf)
d_rover, _ = cKDTree(S).query(V, workers=-1)
d = np.minimum(d_sand, d_rover)

lo, hi = S.min(0) - 0.05, S.max(0) + 0.05
near_rover = np.all((V >= lo) & (V <= hi), axis=1) & (V[:, 1] > 0.03)


def stats(x):
    x = x * 1000.0
    return {"vertices": int(len(x)), "median_mm": float(np.median(x)), "mean_mm": float(x.mean()),
            "p90_mm": float(np.percentile(x, 90)), "p99_mm": float(np.percentile(x, 99)),
            "within_1_voxel": float((x <= args.voxel * 1000).mean()),
            "within_2_voxels": float((x <= 2 * args.voxel * 1000).mean())}


acc_all, acc_rover = stats(d), stats(d[near_rover])

# --- 3. coverage of the rover's visible surface --------------------------------
fx, fy, cx, cy = tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]
seen = np.zeros(len(S), bool)
for fr in frames["train"][::2]:
    c2w = np.array(fr["transform_matrix"])
    w2c = np.linalg.inv(c2w)
    pc = S @ w2c[:3, :3].T + w2c[:3, 3]                     # OpenGL camera frame
    z = -pc[:, 2]
    ok = z > 1e-3
    u = np.full(len(S), -1.0)
    v = np.full(len(S), -1.0)
    u[ok] = fx * pc[ok, 0] / z[ok] + cx
    v[ok] = -fy * pc[ok, 1] / z[ok] + cy
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.asarray(Image.open(os.path.join(args.data, fr["depth_file_path"]))).astype(np.float32) / 1000.0
    dz = depth[v[inb].astype(int), u[inb].astype(int)]
    seen[np.flatnonzero(inb)] |= (dz > 0) & (np.abs(dz - z[inb]) < 0.01 + 0.004 * z[inb])
d_cov, _ = cKDTree(V).query(S[seen], workers=-1)
coverage = {"visible_share_of_rover_surface": float(seen.mean()),
            "reached_within_2_voxels": float((d_cov <= 2 * args.voxel).mean()),
            "median_gap_mm": float(np.median(d_cov) * 1000)}

print(f"bake accuracy, every vertex: median {acc_all['median_mm']:.1f} mm, p90 {acc_all['p90_mm']:.1f}, "
      f"p99 {acc_all['p99_mm']:.1f}; {acc_all['within_1_voxel'] * 100:.1f} % within one voxel")
print(f"bake accuracy, on the rover ({acc_rover['vertices']} vertices): median {acc_rover['median_mm']:.1f} mm, "
      f"p90 {acc_rover['p90_mm']:.1f}, p99 {acc_rover['p99_mm']:.1f}; "
      f"{acc_rover['within_1_voxel'] * 100:.1f} % within one voxel")
print(f"coverage: {coverage['reached_within_2_voxels'] * 100:.1f} % of the rover's visible surface "
      f"is within two voxels of the bake (median gap {coverage['median_gap_mm']:.1f} mm; "
      f"{coverage['visible_share_of_rover_surface'] * 100:.0f} % of its surface is visible at all); "
      f"rover samples {spacing * 1000:.1f} mm apart")

# --- the sheet ----------------------------------------------------------------------
bake_scene, bake_info = common.build_scene()   # the same sun, the real surfaces taken out
bake_scene.remove(bake_info["rover"])
bake_scene.remove(bake_info["ground"])
geo = surface.to_geometry()
geo.compute_vertex_normals()
mat = tp.MeshStandardMaterial()
mat.color = tp.Color(0.22, 0.22, 0.22)  # mid grey on screen under this sun; lighter washes out
mat.roughness = 0.8
baked = tp.Mesh(geo, mat)
baked.cast_shadow = True
baked.receive_shadow = True
bake_scene.add(baked)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic

labels = ["camera output (truth)", "gsplat render", "threepp splats", "baked surface", "|threepp - truth| x4"]
show = list(range(0, len(frames["test"]), max(1, len(frames["test"]) // 4)))[:4]
for j in show:
    gt = np.asarray(Image.open(os.path.join(args.data, frames["test"][j]["file_path"])).convert("RGB"))
    gs_path = os.path.join(out_dir, f"test_{j:03d}_gsplat.png")
    gs = np.asarray(Image.open(gs_path).convert("RGB")) if os.path.isfile(gs_path) else np.zeros_like(gt)
    bk = shoot(bake_scene, frames["test"][j])
    diff = np.clip(np.abs(renders[j].astype(np.int16) - gt.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    rows.append(np.concatenate([gt, gs, renders[j], bk, diff], 1))
sheet = Image.fromarray(np.concatenate(rows, 0))
sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)
draw = ImageDraw.Draw(sheet)
for k, lab in enumerate(labels):
    draw.rectangle([k * W // 2, 0, k * W // 2 + 8 * len(lab) + 8, 16], fill=(0, 0, 0))
    draw.text((k * W // 2 + 4, 2), lab, fill=(255, 255, 255))
sheet.save(os.path.join(out_dir, "compare_sheet.jpg"), quality=92)

result = {"splats": n_splats, "test_views": len(psnr_cam),
          "psnr_vs_camera_mean": float(np.mean(psnr_cam)), "psnr_vs_camera_min": float(np.min(psnr_cam)),
          "psnr_vs_gsplat_mean": float(np.mean(psnr_gsplat)) if psnr_gsplat else None,
          "bake": {"cameras": len(poses), "voxel_m": args.voxel, "triangles": surface.triangle_count,
                   "seconds": round(bake_s, 1)},
          "accuracy_all": acc_all, "accuracy_rover": acc_rover, "coverage": coverage,
          "rover_sample_spacing_mm": spacing * 1000}
with open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
print(f"wrote {os.path.join(out_dir, 'compare_sheet.jpg')} and compare.json")

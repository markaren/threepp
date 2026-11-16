"""Score a trained splat map against the scene it came from: the picture per split, in threepp, and the geometry.

  picture   threepp's Vulkan splat pass renders the .ply (COLMAP frame, turned upright by rotation.x = pi on the
            node, no tone curve: the splats are display values) from every view of every split but "train", each
            held --frames frames. PSNR and SSIM over the unmasked pixels against the camera output, the masked
            share beside them; and against gsplat's own render of the same view where train.py left one.
  geometry  splats.bake_surface from every --bake-every-th training pose, then each baked vertex's exact distance
            to the true triangles (truth .npz from the site runner: "tris" (M, 3, 3), "tri_veg" (M,) 1 = vegetation,
            "points" area-weighted samples on them and "point_tri" each sample's triangle), reported for hard
            surfaces and vegetation apart. A true sample counts as SEEN when it lands on
            a trained-on pixel (mask 255) of at least --min-seen training depth maps, within 0.1 m + 1 % of the
            depth there. Accuracy is over the baked vertices whose nearest true sample is seen (the region the
            flight saw): median, p90, the share within one voxel. Coverage is the share of the seen samples
            within two voxels of a baked vertex.

Writes score.json, score_sheet_<split>.jpg (truth | threepp | |difference| x4, masked pixels dimmed) and the
manifest beside the .ply.

    py -3.14 python/examples/splat_mapping/score.py --data RUN/dataset --splats RUN/seed0/splats.ply \
        --truth RUN/truth_surface.npz --frames N
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # python/ (threepp)
import threepp as tp  # noqa: E402

import dataset as ds  # noqa: E402
from manifest import write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--splats", required=True)
ap.add_argument("--truth", default=None, help="the true surface samples (.npz with 'points'); none = no geometry")
ap.add_argument("--frames", type=int, required=True, help="frames each view is held, as in the dataset")
ap.add_argument("--voxel", type=float, default=0.2, help="bake voxel, m")
ap.add_argument("--bake-every", type=int, default=2)
ap.add_argument("--max-depth", type=float, default=0.0, help="bake max depth, m (0 = the dataset's range cap)")
ap.add_argument("--min-seen", type=int, default=3)
ap.add_argument("--sheet-views", type=int, default=6)
ap.add_argument("--out", default=None, help="where score.json and the sheets go (default: beside the .ply)")
ap.add_argument("--gsplat-dir", default=None, help="where train.py left its renders (default: beside the .ply)")
ap.add_argument("--min-trainable", type=float, default=0.0,
                help="leave out of the means the views whose trainable (unmasked) share is under this; they stay "
                     "in the per-view lists as null")
args = ap.parse_args()

out_dir = args.out or os.path.dirname(os.path.abspath(args.splats))
gs_dir = args.gsplat_dir or os.path.dirname(os.path.abspath(args.splats))
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(args.data, "transforms.json"), encoding="utf-8") as f:
    tf = json.load(f)
W, H = tf["w"], tf["h"]
vfov = math.degrees(2.0 * math.atan(0.5 * H / tf["fl_y"]))


def vfov_of(fr):
    """A frame's vertical field of view: its own fl_y when it carries one (a dataset that mixes cameras)."""
    return math.degrees(2.0 * math.atan(0.5 * H / fr.get("fl_y", tf["fl_y"])))
unit = tf["depth_unit_scale_factor"]
splits = {}
for fr in tf["frames"]:
    splits.setdefault(fr["split"], []).append(fr)

canvas = tp.Canvas("splat-score", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)
renderer.dlss = False
renderer.fsr = False
renderer.render_scale = 1.0
renderer.auto_exposure = False
renderer.tone_mapping = tp.ToneMapping.NoToneMapping    # the splats are display values already
renderer.tone_mapping_exposure = 1.0
renderer.bloom_intensity = 0.0
renderer.sharpen_strength = 0.0
renderer.sim_time = 0.0
camera = tp.PerspectiveCamera(vfov, W / H, tf["near"] or 0.5, tf["far"] or 60000.0)

data = tp.SplatLoader.load_ply(args.splats)
n_splats = len(data)
cloud = tp.SplatCloud(data)
cloud.rotation.x = math.pi                               # COLMAP frame (+Y down) -> upright, as the editor does
scene = tp.Scene()
scene.background = tp.Color(0.0, 0.0, 0.0)
scene.add(cloud)
scene.update_matrix_world(True)


def load(fr, key, conv="RGB"):
    return np.asarray(Image.open(os.path.join(args.data, fr[key])).convert(conv) if conv else
                      Image.open(os.path.join(args.data, fr[key])))


# --- the picture ---------------------------------------------------------------
t0 = time.perf_counter()
result = {"splats": n_splats, "frames_per_view": args.frames, "splits": {}}
for split, frames in splits.items():
    if split == "train":
        continue
    rows, ps, ss, pg, shares, npx = [], [], [], [], [], []
    show = set(np.linspace(0, len(frames) - 1, min(args.sheet_views, len(frames))).round().astype(int).tolist())
    for j, fr in enumerate(frames):
        pos, tgt, up = ds.pose_of(fr)
        if abs(camera.fov - vfov_of(fr)) > 1e-6:
            camera.fov = vfov_of(fr)
            camera.update_projection_matrix()
        ds.place(camera, pos, tgt, up)
        for _ in range(args.frames):
            renderer.render(scene, camera)
        img = renderer.read_pixels()
        gt = load(fr, "file_path")
        mask = load(fr, "mask_path", None) > 127 if "mask_path" in fr else np.ones((H, W), bool)
        shares.append(1.0 - mask.mean())
        npx.append(int(mask.sum()))
        if mask.mean() < args.min_trainable:              # too little to score: kept in the lists as null
            ps.append(float("nan"))
            ss.append(float("nan"))
            continue
        ps.append(ds.masked_psnr(img, gt, mask))
        ss.append(ds.masked_ssim(img, gt, mask))
        gs = os.path.join(gs_dir, f"{ds.safe(split)}_{j:03d}_gsplat.png")
        if os.path.isfile(gs):
            pg.append(ds.masked_psnr(img, np.asarray(Image.open(gs).convert("RGB")), mask))
        if j in show:
            diff = np.clip(np.abs(img.astype(np.int16) - gt.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            dim = np.where(mask[..., None], 1.0, 0.3)
            rows.append((np.concatenate([gt, img, diff], 1) * np.concatenate([dim] * 3, 1)).astype(np.uint8))
    p = np.array(ps)
    scored = ~np.isnan(p)
    r = {"views": len(ps), "scored_views": int(scored.sum()), "left_out_under_min_trainable": int((~scored).sum()),
         "min_trainable": args.min_trainable,
         "psnr_mean": float(np.nanmean(p)) if scored.any() else None, "psnr_min": float(np.nanmin(p)) if scored.any() else None,
         "worst_view": int(np.nanargmin(p)) if scored.any() else None, "ssim_mean": float(np.nanmean(ss)) if scored.any() else None,
         "masked_share_mean": float(np.mean(shares)), "trainable_px_mean": float(np.mean(npx)),
         "trainable_px_min": int(np.min(npx)),
         "psnr_vs_gsplat_mean": float(np.mean(pg)) if pg else None,
         "psnr": [None if np.isnan(x) else round(x, 3) for x in ps], "ssim": [None if np.isnan(x) else round(x, 4) for x in ss],
         "trainable_px": npx}
    result["splits"][split] = r
    if not rows:
        result["splits"][split] = r
        continue
    sheet = Image.fromarray(np.concatenate(rows, 0))
    sheet.resize((sheet.width // 3, sheet.height // 3), Image.LANCZOS).save(
        os.path.join(out_dir, f"score_sheet_{ds.safe(split)}.jpg"), quality=90)
    print(f"  {split:18s} {r['scored_views']:3d} of {r['views']} views: threepp PSNR {r['psnr_mean']:6.2f} dB (worst {r['psnr_min']:.2f}), "
          f"SSIM {r['ssim_mean']:.4f}, masked {r['masked_share_mean'] * 100:.1f} %, trainable px mean {r['trainable_px_mean']:.0f}"
          + (f"; vs gsplat {r['psnr_vs_gsplat_mean']:.2f} dB" if pg else ""), flush=True)
result["picture_seconds"] = round(time.perf_counter() - t0, 1)

# --- the geometry ----------------------------------------------------------------
if args.truth:
    train = splits["train"]
    max_depth = args.max_depth or tf.get("range_cap_m", 300.0)
    poses = []
    for fr in train[::args.bake_every]:
        pos, tgt, up = ds.pose_of(fr)
        poses.append(tp.BakePose(tp.Vector3(*pos), tp.Vector3(*tgt), tp.Vector3(*up), vfov_of(fr)))
    t0 = time.perf_counter()
    surface = tp.bake_surface(renderer, cloud, voxel_size=args.voxel, poses=poses, max_depth=max_depth)
    bake_s = time.perf_counter() - t0
    V = np.asarray(surface.positions, np.float32).reshape(-1, 3)
    print(f"bake: {surface.triangle_count} triangles, {len(V)} vertices from {len(poses)} poses, voxel {args.voxel} m, "
          f"max depth {max_depth} m, {bake_s:.1f} s; stats " + str({k: v for k, v in surface.stats.items() if isinstance(v, (int, float))}),
          flush=True)

    S = np.load(args.truth)["points"].astype(np.float64)
    # seen: on a trained-on pixel of >= min_seen training depth maps, within 0.1 m + 1 % of the depth there
    t0 = time.perf_counter()
    # counted per platform (a frame's "platform", for a dataset that joins vehicles), and over all frames
    platforms = sorted({fr.get("platform", "all") for fr in train})
    counts = {pl: np.zeros(len(S), np.uint8) for pl in platforms}
    for fr in train:
        count = counts[fr.get("platform", "all")]
        fx, fy = fr.get("fl_x", tf["fl_x"]), fr.get("fl_y", tf["fl_y"])
        cx, cy = fr.get("cx", tf["cx"]), fr.get("cy", tf["cy"])
        w2c = np.linalg.inv(np.array(fr["transform_matrix"]))
        depth = load(fr, "depth_file_path", None).astype(np.float32) * unit
        mask = load(fr, "mask_path", None) > 127
        for a in range(0, len(S), 4_000_000):
            P = S[a:a + 4_000_000]
            pc = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = -pc[:, 2]
            ok = z > 0.5
            u = np.where(ok, fx * pc[:, 0] / np.where(ok, z, 1) + cx, -1)
            v = np.where(ok, -fy * pc[:, 1] / np.where(ok, z, 1) + cy, -1)
            inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            k = np.flatnonzero(inb)
            ui, vi = u[k].astype(int), v[k].astype(int)
            dz = depth[vi, ui]
            hit = mask[vi, ui] & (dz > 0) & (np.abs(dz - z[k]) < 0.1 + 0.01 * z[k])
            count[a + k[hit]] += 1
    seen = sum(c.astype(np.int32) for c in counts.values()) >= args.min_seen
    seen_by = {pl: c >= args.min_seen for pl, c in counts.items()} if len(platforms) > 1 else {}
    seen_s = time.perf_counter() - t0
    truth = np.load(args.truth)
    T = truth["tris"].astype(np.float64)                 # (M, 3, 3) the true triangles
    tri_of = truth["point_tri"]                           # the triangle each sample lies on
    veg_tri = truth["tri_veg"].astype(bool)
    veg_s = veg_tri[tri_of]
    tree = cKDTree(S)

    def tri_dist(P, tri):
        """Exact distance from points P (n, 3) to triangles tri (n, 3, 3) (Ericson, Real-Time Collision Detection 5.1.5)."""
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac, ap = b - a, c - a, P - a
        dot = lambda x, y: np.einsum("ij,ij->i", x, y)    # noqa: E731
        d1, d2 = dot(ab, ap), dot(ac, ap)
        bp = P - b
        d3, d4 = dot(ab, bp), dot(ac, bp)
        cp = P - c
        d5, d6 = dot(ab, cp), dot(ac, cp)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        denom = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
        q = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]                    # inside the face
        e = lambda x, y: np.where(np.abs(x - y) > 1e-30, x - y, 1e-30)                    # noqa: E731
        t_ab = np.clip(d1 / e(d1, d3), 0, 1)
        t_ac = np.clip(d2 / e(d2, d6), 0, 1)
        t_bc = np.clip((d4 - d3) / e(d4 - d3 + d5, d6), 0, 1)
        cand = np.stack([q, a + ab * t_ab[:, None], a + ac * t_ac[:, None], b + (c - b) * t_bc[:, None]], 1)
        inside = (va >= 0) & (vb >= 0) & (vc >= 0)
        dd = np.linalg.norm(cand - P[:, None], axis=2)
        dd[:, 0] = np.where(inside, dd[:, 0], np.inf)
        return dd.min(1)

    # each baked vertex: its 8 nearest samples name the candidate triangles; the distance is the exact distance to
    # the nearest of those. A vertex is in the seen region when its nearest sample is seen; its kind (hard or
    # vegetation) is the nearest candidate triangle's.
    t0 = time.perf_counter()
    KC = 8
    d = np.empty(len(V))
    kind = np.zeros(len(V), bool)
    in_region = np.zeros(len(V), bool)
    nearest0 = np.zeros(len(V), np.int64)
    for a in range(0, len(V), 400_000):
        Pv = V[a:a + 400_000].astype(np.float64)
        _, nn = tree.query(Pv, k=KC, workers=-1)
        tid = tri_of[nn]                                   # (n, KC)
        dist = np.stack([tri_dist(Pv, T[tid[:, j]]) for j in range(KC)], 1)
        best = dist.argmin(1)
        d[a:a + len(Pv)] = dist[np.arange(len(Pv)), best]
        kind[a:a + len(Pv)] = veg_tri[tid[np.arange(len(Pv)), best]]
        in_region[a:a + len(Pv)] = seen[nn[:, 0]]
        nearest0[a:a + len(Pv)] = nn[:, 0]
    tri_s = time.perf_counter() - t0
    vt = cKDTree(V)
    d_cov, _ = vt.query(S[seen], workers=-1)
    veg_cov = veg_s[seen]

    def stats(x):
        x = x * 1000.0
        if len(x) == 0:
            return {"vertices": 0}
        return {"vertices": int(len(x)), "median_mm": float(np.median(x)), "mean_mm": float(x.mean()),
                "p90_mm": float(np.percentile(x, 90)), "p99_mm": float(np.percentile(x, 99)),
                "within_1_voxel": float((x <= args.voxel * 1000).mean()),
                "within_2_voxels": float((x <= 2 * args.voxel * 1000).mean())}

    def cov(x):
        return {"samples": int(len(x)), "within_2_voxels": float((x <= 2 * args.voxel).mean()) if len(x) else None,
                "within_1_voxel": float((x <= args.voxel).mean()) if len(x) else None,
                "median_gap_mm": float(np.median(x) * 1000) if len(x) else None}
    result["geometry"] = {
        "method": "distance to the true triangles (exact, over the triangles of each vertex's 8 nearest samples)",
        "bake": {"poses": len(poses), "voxel_m": args.voxel, "max_depth_m": max_depth, "triangles": surface.triangle_count,
                 "vertices": int(len(V)), "seconds": round(bake_s, 1),
                 "stats": {k: v for k, v in surface.stats.items() if isinstance(v, (int, float))}},
        "truth": {"samples": int(len(S)), "triangles": int(len(T)), "vegetation_triangles": int(veg_tri.sum()),
                  "spacing_m": float(truth["spacing_m"]), "seen": int(seen.sum()), "seen_vegetation": int((seen & veg_s).sum()),
                  "min_seen": args.min_seen, "seen_seconds": round(seen_s, 1), "triangle_seconds": round(tri_s, 1)},
        "accuracy_seen_region": stats(d[in_region]),
        "accuracy_seen_hard": stats(d[in_region & ~kind]),
        "accuracy_seen_vegetation": stats(d[in_region & kind]),
        "accuracy_all_vertices": stats(d),
        "share_of_vertices_in_seen_region": float(in_region.mean()) if len(V) else 0.0,
        "coverage": cov(d_cov), "coverage_hard": cov(d_cov[~veg_cov]), "coverage_vegetation": cov(d_cov[veg_cov]),
    }
    # the same measures over each platform's own seen region (a sample seen in >= min_seen of THAT platform's maps)
    regions = {}
    for pl, sb in seen_by.items():
        reg = sb[nearest0]
        dc, _ = vt.query(S[sb], workers=-1)
        vc = veg_s[sb]
        regions[pl] = {"seen": int(sb.sum()), "accuracy_seen_region": stats(d[reg]), "accuracy_seen_hard": stats(d[reg & ~kind]),
                       "accuracy_seen_vegetation": stats(d[reg & kind]), "share_of_vertices_in_seen_region": float(reg.mean()),
                       "coverage": cov(dc), "coverage_hard": cov(dc[~vc]), "coverage_vegetation": cov(dc[vc])}
        a_ = regions[pl]["accuracy_seen_region"]
        print(f"geometry, seen by {pl}: median {a_.get('median_mm', float('nan')):.0f} mm, p90 {a_.get('p90_mm', float('nan')):.0f}, "
              f"coverage {regions[pl]['coverage']['within_2_voxels'] * 100:.1f} %", flush=True)
    if regions:
        result["geometry"]["regions"] = regions
    g = result["geometry"]
    for name in ("accuracy_seen_region", "accuracy_seen_hard", "accuracy_seen_vegetation"):
        a_ = g[name]
        print(f"geometry {name}: {a_.get('vertices', 0)} vertices, median {a_.get('median_mm', float('nan')):.0f} mm, "
              f"p90 {a_.get('p90_mm', float('nan')):.0f}, within one voxel {a_.get('within_1_voxel', 0) * 100:.1f} %", flush=True)
    for name in ("coverage", "coverage_hard", "coverage_vegetation"):
        c_ = g[name]
        print(f"geometry {name}: {c_['samples']} seen samples, {c_['within_2_voxels'] * 100:.1f} % within two voxels, "
              f"median gap {c_['median_gap_mm']:.0f} mm", flush=True)

with open(os.path.join(out_dir, "score.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
write_manifest(out_dir, extra={"data": os.path.basename(os.path.dirname(os.path.abspath(args.data))),
                               "splats": os.path.basename(out_dir)})
print(f"wrote {os.path.join(out_dir, 'score.json')}")

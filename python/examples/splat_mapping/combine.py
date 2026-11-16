"""Join the training frames of several datasets into one training set, each frame with its own camera.

Every training frame of each source keeps its own intrinsics (fl_x, fl_y, cx, cy on the frame: the trainer and
score.py take them per frame), its own mask and depth, and a "platform" tag (the name given to its source); its
files stay where they are (paths relative to the new dataset). The start points are the union of the sources'
points3d.ply, voxel-thinned once. Held-out splits are not copied: score the map on each source dataset instead.

    py -3.14 combine.py --out RUN/dataset --source x8=X8RUN/dataset --source mariner=MARINERRUN/dataset [--voxel 0.5]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset as ds  # noqa: E402
from manifest import write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--source", action="append", required=True, help="NAME=DATASET_DIR")
ap.add_argument("--voxel", type=float, default=0.5, help="the start points' voxel, m")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def read_points(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], 1), np.stack([rec["r"], rec["g"], rec["b"]], 1)


frames, xyz, rgb, base, info = [], [], [], None, {}
for spec in args.source:
    name, root = spec.split("=", 1)
    tf = json.load(open(os.path.join(root, "transforms.json"), encoding="utf-8"))
    if base is None:
        base = tf
    assert (tf["w"], tf["h"], tf["depth_unit_scale_factor"], tf["range_cap_m"]) == \
        (base["w"], base["h"], base["depth_unit_scale_factor"], base["range_cap_m"]), f"{name}: another frame size, depth unit or range cap"
    n = 0
    for fr in tf["frames"]:
        if fr["split"] != "train":
            continue
        g = dict(fr, platform=name, source_frame=fr["file_path"])
        for key in ("fl_x", "fl_y", "cx", "cy"):
            g[key] = fr.get(key, tf[key])
        for key in ("file_path", "depth_file_path", "mask_path", "class_path"):
            g[key] = os.path.relpath(os.path.join(root, fr[key]), args.out).replace("\\", "/")
        frames.append(g)
        n += 1
    p, c = read_points(os.path.join(root, tf["ply_file_path"]))
    xyz.append(p)
    rgb.append(c)
    info[name] = {"dataset": os.path.basename(os.path.dirname(os.path.abspath(root))), "train_frames": n,
                  "start_points": int(len(p)), "fl_x": tf["fl_x"], "fl_y": tf["fl_y"]}
P, C = np.concatenate(xyz), np.concatenate(rgb)
key = np.floor(P / args.voxel).astype(np.int64)
_, first = np.unique(key, axis=0, return_index=True)
ds.write_points_ply(os.path.join(args.out, "points3d.ply"), P[first], C[first])
out = {k: base[k] for k in ("camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h", "k1", "k2", "p1", "p2", "near", "far",
                            "depth_unit_scale_factor", "masked", "range_cap_m", "classes")}
out.update(ply_file_path="points3d.ply", frames=frames,
           start_points={"union": int(len(P)), "voxel_m": args.voxel, "written": int(len(first))},
           threepp={"combined_from": info, "note": "training frames only, each with its own intrinsics and a platform tag"})
with open(os.path.join(args.out, "transforms.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
write_manifest(os.path.dirname(os.path.abspath(args.out)), name="combine", extra={"sources": info, "start_points": out["start_points"]})
counts = ", ".join(f"{k} {v['train_frames']}" for k, v in info.items())
print(f"{len(frames)} training frames ({counts}); start points {len(P)} -> {len(first)} at {args.voxel} m")

"""The mapping harness's dataset: nerfstudio's transforms.json, extended for keyframes taken along a vehicle's path.

Per frame (beside nerfstudio's file_path and transform_matrix, camera-to-world in the OpenGL convention):
  split           "train", "on_path" or "off_path:<name>"; the trainer fits "train" and scores every other split
  t               the sim time the frame was taken at, s
  depth_file_path planar depth along the view axis, uint16 in the dataset's depth unit, 0 = none
  mask_path       uint8, 255 = train on (or score) this pixel, 0 = ignore it
  class_path      uint8, the raw class of each pixel (CLASSES below), so a later stage can degrade the masks
Global: intrinsics (fl_x, fl_y, cx, cy, w, h), "near"/"far" of the camera, depth_unit_scale_factor (metres per
unit), ply_file_path (the start points), and a "threepp" meta block the site runner fills.

The start points are back-projected from the training frames' depth, ONLY where their mask is 255, thinned to one
per voxel and capped (both arguments: a site is about a hundred times the rover's extent).
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

# The raw class map's values (class_path). The mask is derived from it plus the range cap.
STATIC, WATER, MOVING, SKY, OWN = 0, 1, 2, 3, 4
CLASSES = {"static": STATIC, "water": WATER, "moving": MOVING, "sky": SKY, "own_vehicle": OWN}
CLASS_COLOURS = {STATIC: None, WATER: (41, 120, 224), MOVING: (230, 57, 70), SKY: (120, 90, 200), OWN: (255, 200, 0)}
RANGE_COLOUR = (40, 40, 40)


def place(camera, pos, target, up=(0.0, 1.0, 0.0)):
    camera.up.set(*map(float, up))
    camera.position.set(*map(float, pos))
    camera.look_at(*map(float, target))
    camera.update_matrix_world(True)


def matrix_world_np(obj):
    """An Object3D's matrix_world as a row-major 4x4 numpy array."""
    m = obj.matrix_world
    m = m() if callable(m) else m
    return np.array(m.elements(), np.float64).reshape(4, 4).T


def pose_of(frame):
    """(position, target, up) of a dataset frame, from its OpenGL camera-to-world."""
    m = np.array(frame["transform_matrix"])
    pos = m[:3, 3]
    return pos, pos - m[:3, 2], m[:3, 1]


def safe(split):
    return split.replace(":", "_")


class Writer:
    """Writes the frames of one dataset folder, and at close() the start points and transforms.json."""

    def __init__(self, root, depth_unit=0.01, range_cap=300.0):
        self.root, self.depth_unit, self.range_cap = root, depth_unit, range_cap
        for d in ("images", "depth", "masks", "classes"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.frames, self.intr, self.counts = [], None, {}
        self.pts, self.cols = [], []

    def add(self, split, rgb, depth, cls, c2w, t, intr, seed_points=False, point_stride=3, extra=None):
        """One frame. rgb (H, W, 3) uint8; depth (H, W) float metres (planar; >= far or non-finite = none);
        cls (H, W) uint8 of CLASSES; c2w the camera's matrix_world (4x4); t the sim time. Returns its mask."""
        if self.intr is None:
            self.intr = dict(intr)
        k = self.counts.get(split, 0)
        self.counts[split] = k + 1
        name = f"{safe(split)}_{k:04d}"
        far = self.intr.get("far", np.inf)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 0.999 * far)
        units = np.where(valid, np.clip(np.round(depth / self.depth_unit), 1, 65535), 0).astype(np.uint16)
        cls = cls.copy()
        cls[~valid & (cls == STATIC)] = SKY                  # nothing drawn there: sky
        mask = ((cls == STATIC) & valid & (depth <= self.range_cap)).astype(np.uint8) * 255
        Image.fromarray(rgb).save(os.path.join(self.root, "images", name + ".png"))
        Image.fromarray(units).save(os.path.join(self.root, "depth", name + ".png"))
        Image.fromarray(mask).save(os.path.join(self.root, "masks", name + ".png"))
        Image.fromarray(cls).save(os.path.join(self.root, "classes", name + ".png"))
        fr = {"file_path": f"images/{name}.png", "depth_file_path": f"depth/{name}.png",
              "mask_path": f"masks/{name}.png", "class_path": f"classes/{name}.png",
              "transform_matrix": np.asarray(c2w).tolist(), "split": split, "t": round(float(t), 4),
              "masked_share": float(1.0 - mask.mean() / 255.0)}
        if extra:
            fr.update(extra)
        self.frames.append(fr)
        if seed_points:
            m = mask[::point_stride, ::point_stride] > 0
            v, u = np.nonzero(m)
            v, u = v * point_stride, u * point_stride
            z = depth[v, u]
            x = (u + 0.5 - self.intr["cx"]) * z / self.intr["fx"]
            y = (v + 0.5 - self.intr["cy"]) * z / self.intr["fy"]
            cam = np.stack([x, -y, -z, np.ones_like(z)], 1)          # OpenCV -> OpenGL camera axes
            self.pts.append((cam @ np.asarray(c2w).T)[:, :3].astype(np.float32))
            self.cols.append(rgb[v, u])
        return mask

    def close(self, meta, voxel=0.5, cap=1_000_000, seed=0):
        """The start cloud (voxel-thinned, then capped at random with `seed`) and transforms.json."""
        rng = np.random.default_rng(seed)
        xyz, rgb = np.concatenate(self.pts), np.concatenate(self.cols)
        raw = len(xyz)
        key = np.floor(xyz / voxel).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        xyz, rgb = xyz[first], rgb[first]
        thinned = len(xyz)
        if len(xyz) > cap:
            pick = rng.choice(len(xyz), cap, replace=False)
            xyz, rgb = xyz[pick], rgb[pick]
        write_points_ply(os.path.join(self.root, "points3d.ply"), xyz, rgb)
        i = self.intr
        tf = {"camera_model": "OPENCV", "fl_x": i["fx"], "fl_y": i["fy"], "cx": i["cx"], "cy": i["cy"],
              "w": i["w"], "h": i["h"], "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
              "near": i.get("near"), "far": i.get("far"),
              "depth_unit_scale_factor": self.depth_unit, "ply_file_path": "points3d.ply",
              "masked": True, "range_cap_m": self.range_cap, "classes": CLASSES,
              "start_points": {"raw": raw, "voxel_m": voxel, "after_voxel": thinned, "cap": cap, "written": len(xyz),
                               "seed": seed},
              "frames": self.frames, "threepp": meta}
        with open(os.path.join(self.root, "transforms.json"), "w", encoding="utf-8") as f:
            json.dump(tf, f, indent=1)
        return tf


def write_points_ply(path, xyz, rgb):
    """A colour point cloud (x y z float, red green blue uchar), binary little-endian."""
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def overlay(rgb, cls, depth_m, range_cap):
    """The frame with what the mask ignores tinted by why: water blue, moving red, sky violet, beyond the range
    cap dark grey; trained-on pixels as they are."""
    out = rgb.astype(np.float32)
    for c, col in CLASS_COLOURS.items():
        if col is None:
            continue
        m = cls == c
        out[m] = 0.45 * out[m] + 0.55 * np.float32(col)
    far = (cls == STATIC) & ~(depth_m <= range_cap)
    out[far] = 0.35 * out[far] + 0.65 * np.float32(RANGE_COLOUR)
    return out.clip(0, 255).astype(np.uint8)


def contact_sheet(path, tiles, labels, cols=4, scale=0.4):
    """Tiles (H, W, 3) uint8 with a label each, `cols` across, each scaled by `scale`."""
    h, w = tiles[0].shape[:2]
    tw, th = int(w * scale), int(h * scale)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (k % cols) * tw, (k // cols) * th
        sheet.paste(Image.fromarray(t).resize((tw, th), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + 7 * len(lab) + 8, y + 14], fill=(0, 0, 0))
        draw.text((x + 4, y + 1), lab, fill=(255, 255, 255))
    sheet.save(path, quality=90)


# --------------------------------------------------------------------------- overlap between datasets
class SeenTest:
    """Is a world point on surface a dataset's training views saw: on a trained-on pixel (mask 255) of >= min_seen
    training depth maps, within 0.1 m + 1 % of the depth there (the same rule as score.py's)."""

    def __init__(self, root, min_seen=3):
        tf = json.load(open(os.path.join(root, "transforms.json"), encoding="utf-8"))
        self.k, self.min_seen, self.maps = tf, min_seen, []
        for fr in tf["frames"]:
            if fr["split"] != "train":
                continue
            d = np.asarray(Image.open(os.path.join(root, fr["depth_file_path"]))).astype(np.float32) * tf["depth_unit_scale_factor"]
            m = np.asarray(Image.open(os.path.join(root, fr["mask_path"]))) > 127
            k = {key: fr.get(key, tf[key]) for key in ("fl_x", "fl_y", "cx", "cy")}
            self.maps.append((np.linalg.inv(np.array(fr["transform_matrix"])), np.where(m, d, 0.0).astype(np.float32), k))

    def __call__(self, P):
        count = np.zeros(len(P), np.int32)
        W_, H_ = self.k["w"], self.k["h"]
        for w2c, d, k in self.maps:
            pc = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = -pc[:, 2]
            ok = z > 0.5
            zs = np.where(ok, z, 1.0)
            u = (k["fl_x"] * pc[:, 0] / zs + k["cx"]).astype(np.int64)
            v = (-k["fl_y"] * pc[:, 1] / zs + k["cy"]).astype(np.int64)
            ok &= (u >= 0) & (u < W_) & (v >= 0) & (v < H_)
            dz = np.zeros(len(P), np.float32)
            dz[ok] = d[v[ok], u[ok]]
            count += (ok & (dz > 0) & (np.abs(dz - z) < 0.1 + 0.01 * z)).astype(np.int32)
        return count >= self.min_seen


def frame_overlap(seen, fr, root, k, step=8):
    """(share of a frame's trainable pixels (every step-th) that lie on surface `seen` (a SeenTest) counts as seen,
    the frame's trainable pixel count); k = the frame's dataset (its fl_x, fl_y, cx, cy, depth unit)."""
    d = np.asarray(Image.open(os.path.join(root, fr["depth_file_path"]))).astype(np.float32) * k["depth_unit_scale_factor"]
    m = np.asarray(Image.open(os.path.join(root, fr["mask_path"]))) > 127
    if not m.any():
        return 0.0, 0
    P = backproject(d, np.array(fr["transform_matrix"]), {"fx": fr.get("fl_x", k["fl_x"]), "fy": fr.get("fl_y", k["fl_y"]),
                                                          "cx": fr.get("cx", k["cx"]), "cy": fr.get("cy", k["cy"])}, m, step)
    return float(seen(P).mean()), int(m.sum())


def backproject(depth, c2w, k, mask, step=8):
    """World points of the mask's pixels (every step-th), from planar depth; k has fx, fy, cx, cy."""
    v, u = np.nonzero(mask[::step, ::step])
    v, u = v * step, u * step
    z = depth[v, u]
    cam = np.stack([(u + 0.5 - k["cx"]) * z / k["fx"], -(v + 0.5 - k["cy"]) * z / k["fy"], -z, np.ones_like(z)], 1)
    return (cam @ c2w.T)[:, :3]


# --------------------------------------------------------------------------- masked picture metrics (numpy)
def masked_psnr(a, b, mask):
    """PSNR in dB over the pixels where mask (H, W) is true; a and b uint8 (H, W, 3)."""
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    d = (a.astype(np.float32) - b.astype(np.float32))[m] / 255.0
    return float(-10.0 * math.log10(float((d * d).mean()) + 1e-12))


def masked_ssim(a, b, mask):
    """Mean SSIM (11x11 Gaussian window, sigma 1.5, on [0, 1]) over the pixels where mask is true."""
    from scipy.ndimage import gaussian_filter
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    x, y = a.astype(np.float64) / 255.0, b.astype(np.float64) / 255.0
    f = lambda z: gaussian_filter(z, sigma=(1.5, 1.5, 0), truncate=5.0 / 1.5)   # noqa: E731  (11 taps)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean(axis=2)[m].mean())

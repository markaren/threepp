"""Trail-floor analysis for the Calico Tanks Gaussian-splat scan (WP0).

Pure Python (PIL + numpy). No threepp import, no C++ build. Reads the streamed
SOG directory written by splat-transform v3.1.7 and writes ``shots.json``:
the walkable wash floor, a spine path along it, a spawn pose, and four camera
poses for the demo.

Frame convention
----------------
The scan is stored Y-DOWN (COLMAP style). ``SogLoader`` / ``gaussian_splats.exe``
apply a half-turn about X by default (``cloud.rotation.x = pi``), i.e.
``y -> -y, z -> -z``. Everything this script emits is in that POST-FLIP Y-UP
world frame, so the numbers can be handed straight to
``gaussian_splats.exe <asset> --vulkan --level 0 --cam x,y,z --look x,y,z``
with no ``--no-flip``. Sanity anchor: SuperSplat's initial camera in that frame
is (-11.82, 2.19, 1.66) looking at (-13.82, 2.09, 1.73); the wash runs toward -x.

Yaw convention: ``yaw = atan2(fx, fz)`` about +Y, i.e. the Three.js heading of a
node whose local +Z points along ``forward``. ``forward`` is also emitted
explicitly so no consumer has to guess.

SOG v2 decoding (per chunk ``meta.json``)
----------------------------------------
means   : 16-bit per channel, ``means_l.webp`` (low byte) | ``means_u.webp`` << 8,
          /65535, lerp between ``meta.means.mins`` and ``maxs`` per axis, then
          ``sign(v) * (exp(|v|) - 1)``.
scales  : ``exp(meta.scales.codebook[idx])`` indexed by ``scales.webp`` RGB.
opacity : ``sh0.webp`` alpha / 255.
Only the first ``count`` pixels (row-major) of each image are valid.

Usage
-----
    py -3.14 python/examples/calico/trail_analysis.py \
        [--asset C:/dev/splats/calico_tanks] [--level 2] [--out shots.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ASSET = os.environ.get("THREEPP_CALICO_ASSET", r"C:/dev/splats/calico_tanks")

# SuperSplat's initial camera, post-flip. Used only as the "near here" anchor
# when several walkable regions survive the plane test.
ANCHOR_XZ = (-11.82, 1.66)

CELL = 0.5          # XZ grid cell size, metres
FLOOR_Q = 0.15      # quantile of splat Y taken as the cell's floor height
MIN_PER_CELL = 8    # splats needed before a cell has a floor sample
# The brief asks for RMS < 5 cm / slope < 15 deg. At level 2 that splits the wash
# into two disjoint 10 m patches (94 and 78 cells) because the splat cloud is a
# noisy sample of a rock streambed, not a surveyed surface. 10 cm / 20 deg keeps
# the same corridor and joins it into one 30 m walkable region; --strict restores
# the brief's numbers.
RMS_MAX = 0.10      # plane-fit RMS over the 3x3 neighbourhood, metres
SLOPE_MAX_DEG = 20.0
OPACITY_MIN = 0.5


# --------------------------------------------------------------------------- #
# SOG decoding
# --------------------------------------------------------------------------- #
def _plane(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8).reshape(-1, 4)


def decode_chunk(chunk_dir: Path, flip: bool = True):
    """Return (means Nx3 float32, opacity N float32, scale N float32)."""
    meta = json.loads((chunk_dir / "meta.json").read_text(encoding="utf-8"))
    n = int(meta["count"])

    lo = _plane(chunk_dir / "means_l.webp")[:n, :3].astype(np.uint16)
    hi = _plane(chunk_dir / "means_u.webp")[:n, :3].astype(np.uint16)
    v = (lo | (hi << 8)).astype(np.float32) / 65535.0

    mins = np.asarray(meta["means"]["mins"], dtype=np.float32)
    maxs = np.asarray(meta["means"]["maxs"], dtype=np.float32)
    v = mins + v * (maxs - mins)
    means = np.sign(v) * (np.exp(np.abs(v)) - 1.0)

    opacity = _plane(chunk_dir / "sh0.webp")[:n, 3].astype(np.float32) / 255.0

    book = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
    sidx = _plane(chunk_dir / "scales.webp")[:n, :3].astype(np.int32)
    scale = np.exp(book[sidx]).mean(axis=1)

    if flip:  # the loader's half-turn about X
        means[:, 1] *= -1.0
        means[:, 2] *= -1.0
    return means.astype(np.float32), opacity, scale.astype(np.float32)


def load_level(asset: Path, level: int, flip: bool = True):
    dirs = sorted(
        (d for d in asset.iterdir() if d.is_dir() and d.name.startswith(f"{level}_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not dirs:
        raise SystemExit(f"no level-{level} chunks under {asset}")
    P, O, S = [], [], []
    for d in dirs:
        m, o, s = decode_chunk(d, flip)
        P.append(m)
        O.append(o)
        S.append(s)
        print(f"  {d.name}: {len(m):>8d} splats")
    return np.concatenate(P), np.concatenate(O), np.concatenate(S)


# --------------------------------------------------------------------------- #
# Floor extraction
# --------------------------------------------------------------------------- #
def floor_grid(pts: np.ndarray):
    """Low-quantile Y per 0.5 m XZ cell. Returns (height, count, x0, z0, shape)."""
    x0 = math.floor(pts[:, 0].min() / CELL)
    z0 = math.floor(pts[:, 2].min() / CELL)
    ix = (np.floor(pts[:, 0] / CELL) - x0).astype(np.int64)
    iz = (np.floor(pts[:, 2] / CELL) - z0).astype(np.int64)
    nx, nz = int(ix.max()) + 1, int(iz.max()) + 1
    flat = ix * nz + iz

    order = np.argsort(flat, kind="stable")
    flat_s, y_s = flat[order], pts[order, 1]
    starts = np.searchsorted(flat_s, np.arange(nx * nz), side="left")
    ends = np.searchsorted(flat_s, np.arange(nx * nz), side="right")

    height = np.full(nx * nz, np.nan, dtype=np.float64)
    count = (ends - starts).astype(np.int64)
    for c in np.nonzero(count >= MIN_PER_CELL)[0]:
        height[c] = np.quantile(y_s[starts[c]:ends[c]], FLOOR_Q)
    return (height.reshape(nx, nz), count.reshape(nx, nz), x0, z0, (nx, nz))


def planar_mask(height: np.ndarray):
    """3x3 local plane fit per cell -> (mask, rms, slope_deg)."""
    nx, nz = height.shape
    mask = np.zeros((nx, nz), dtype=bool)
    rms = np.full((nx, nz), np.nan)
    slope = np.full((nx, nz), np.nan)
    off = [(dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            if not np.isfinite(height[i, j]):
                continue
            A, b = [], []
            for dx, dz in off:
                h = height[i + dx, j + dz]
                if np.isfinite(h):
                    A.append((dx * CELL, dz * CELL, 1.0))
                    b.append(h)
            if len(b) < 6:
                continue
            A = np.asarray(A)
            b = np.asarray(b)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            r = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
            sl = math.degrees(math.atan(math.hypot(sol[0], sol[1])))
            rms[i, j] = r
            slope[i, j] = sl
            mask[i, j] = r < RMS_MAX and sl < SLOPE_MAX_DEG
    return mask, rms, slope


def connected_regions(mask: np.ndarray):
    """8-connected components as lists of (i, j)."""
    nx, nz = mask.shape
    lab = np.zeros((nx, nz), dtype=np.int32)
    regions = []
    for si in range(nx):
        for sj in range(nz):
            if not mask[si, sj] or lab[si, sj]:
                continue
            rid = len(regions) + 1
            stack = [(si, sj)]
            lab[si, sj] = rid
            cells = []
            while stack:
                i, j = stack.pop()
                cells.append((i, j))
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        a, b = i + dx, j + dz
                        if 0 <= a < nx and 0 <= b < nz and mask[a, b] and not lab[a, b]:
                            lab[a, b] = rid
                            stack.append((a, b))
            regions.append(cells)
    return regions


def cell_xz(i, j, x0, z0):
    return ((x0 + i + 0.5) * CELL, (z0 + j + 0.5) * CELL)


# --------------------------------------------------------------------------- #
# Spine
# --------------------------------------------------------------------------- #
def spine_from_region(cells, height, x0, z0, target_len=16.0, step=1.0,
                      bias_xz=None):
    """PCA centreline through the region, trimmed to ~target_len metres.

    ``bias_xz`` pulls the kept window toward that XZ point (the anchor pose),
    clamped so the window stays inside the region.
    """
    pts = np.array([cell_xz(i, j, x0, z0) for i, j in cells])
    hs = np.array([height[i, j] for i, j in cells])
    mu = pts.mean(axis=0)
    d = pts - mu
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    if axis[0] > 0:  # orient toward -x: the wash runs that way
        axis = -axis
    perp = np.array([-axis[1], axis[0]])

    t = d @ axis
    s = d @ perp
    lo, hi = float(np.quantile(t, 0.02)), float(np.quantile(t, 0.98))
    if hi - lo > target_len:
        mid = 0.5 * (lo + hi)
        if bias_xz is not None:
            tb = float((np.asarray(bias_xz) - mu) @ axis)
            mid = min(max(tb, lo + target_len / 2), hi - target_len / 2)
        lo, hi = mid - target_len / 2, mid + target_len / 2

    way = []
    for tv in np.arange(lo, hi + 1e-6, step):
        sel = np.abs(t - tv) <= max(step, 1.0)
        if sel.sum() < 2:
            continue
        sv = float(np.median(s[sel]))
        p = mu + axis * tv + perp * sv
        # floor height from the nearest region cells
        near = np.argsort((pts[sel] - p) ** 2 @ np.ones(2))[: min(5, int(sel.sum()))]
        y = float(np.median(hs[sel][near]))
        way.append([float(p[0]), y, float(p[1])])
    return way, axis, perp


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).with_name("shots.json")))
    ap.add_argument("--roi", type=float, default=30.0,
                    help="half-extent in XZ around the anchor to analyse")
    ap.add_argument("--strict", action="store_true",
                    help="use the brief's RMS < 5 cm / slope < 15 deg thresholds")
    args = ap.parse_args()

    global RMS_MAX, SLOPE_MAX_DEG
    if args.strict:
        RMS_MAX, SLOPE_MAX_DEG = 0.05, 15.0

    asset = Path(args.asset)
    print(f"decoding level {args.level} of {asset}")
    pts, opa, _scale = load_level(asset, args.level)
    print(f"total {len(pts)} splats; bbox min {pts.min(0)} max {pts.max(0)}")

    keep = opa > OPACITY_MIN
    ax, az = ANCHOR_XZ
    keep &= (np.abs(pts[:, 0] - ax) < args.roi) & (np.abs(pts[:, 2] - az) < args.roi)
    keep &= np.abs(pts[:, 1]) < 40.0
    p = pts[keep]
    print(f"{len(p)} splats in the ROI with opacity > {OPACITY_MIN}")

    height, count, x0, z0, shape = floor_grid(p)
    mask, rms, slope = planar_mask(height)
    print(f"grid {shape}, {int(np.isfinite(height).sum())} cells with a floor sample, "
          f"{int(mask.sum())} planar")

    regions = connected_regions(mask)
    regions.sort(key=len, reverse=True)
    if not regions:
        raise SystemExit("no planar region found")

    # largest region whose centroid is within 25 m of the anchor
    def near_anchor(cells):
        c = np.array([cell_xz(i, j, x0, z0) for i, j in cells]).mean(axis=0)
        return math.hypot(c[0] - ax, c[1] - az)

    chosen = next((r for r in regions if near_anchor(r) < 25.0), regions[0])
    print(f"{len(regions)} regions; chosen has {len(chosen)} cells "
          f"({len(chosen) * CELL * CELL:.1f} m^2), centroid {near_anchor(chosen):.1f} m "
          f"from the anchor")

    cells_xz = [cell_xz(i, j, x0, z0) for i, j in chosen]
    hs = [float(height[i, j]) for i, j in chosen]
    arr = np.array(cells_xz)
    poly_min = arr.min(axis=0) - CELL / 2
    poly_max = arr.max(axis=0) + CELL / 2

    spine, axis, perp = spine_from_region(chosen, height, x0, z0,
                                          bias_xz=np.array(ANCHOR_XZ))
    if len(spine) < 4:
        raise SystemExit("spine too short")
    spine_len = float(sum(math.dist(spine[k][::2], spine[k + 1][::2])
                          for k in range(len(spine) - 1)))
    print(f"spine {len(spine)} waypoints, {spine_len:.1f} m")

    a = np.array(spine[0])
    b = np.array(spine[-1])
    fwd = b - a
    fwd[1] = 0.0
    fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[2], 0.0, -fwd[0]])  # +Y cross forward, Y-up
    yaw = math.atan2(fwd[0], fwd[2])
    floor_y = float(spine[0][1])

    # --- camera poses -------------------------------------------------------
    # Derived from the spine, then PINNED to the values that actually rendered
    # clean (shots/*.png). Two of the derived poses failed by eye and are kept
    # here only as a record of why the pinned numbers differ:
    #   * the brief's establishing pose (-6,3.6,1.5)->(-14,1.8,1.7) fills the
    #     left half of frame with a bush; it is not clean.
    #   * a low camera placed at spawn - fwd*2.6 + right*1.6 lands inside the
    #     rock/brush shoulder at (12.8, 1.5), whose splat column runs y 0.3..1.9.
    # Every pinned pose was checked against the local splat column: the camera
    # sits above that column's 99th percentile, i.e. in free air.
    mid = np.array(spine[len(spine) // 2])
    derived = {
        "establishing": (list(a + fwd * -5.0 + np.array([0.0, 3.2, 0.0])),
                         list(mid + np.array([0.0, 0.0, 0.0]))),
        "low_following": (list(a + right * 1.2 + np.array([0.0, 0.6, 0.0])),
                          list(np.array(spine[3]) + np.array([0.0, 0.4, 0.0]))),
        "topdown": ([float(mid[0]), float(mid[1]) + 29.9, float(mid[2]) + 6.3],
                    [float(mid[0]), float(mid[1]), float(mid[2])]),
        "lookback": (list(b + np.array([0.0, 1.6, 0.0])),
                     list(np.array(spine[5]) + np.array([0.0, 0.5, 0.0]))),
    }
    pinned = {
        "establishing": ([16.0, 2.6, -2.5], [2.0, -0.9, -1.3]),
        "low_following": ([10.39, -0.069, 0.54], [7.65, -0.6, -0.7]),
        "topdown": ([2.68, 29.0, 5.0], [2.68, -0.864, -1.259]),
        "lookback": ([-4.984, 1.373, -4.314], [5.636, -0.42, -0.732]),
    }
    notes = {
        "establishing": "three-quarter from the +x end of the spine, 3.3 m above "
                        "the floor, looking down the wash",
        "low_following": "0.6 m above the spawn's floor, 1.2 m to the right and "
                         "level with the spawn: a three-quarter follow of a robot "
                         "walking away down the spine",
        "topdown": "map view of the walkable region, 30 m up with a small +z "
                   "offset so the up vector is not degenerate",
        "lookback": "at the -x end of the spine, eye height, looking back along "
                    "the trail toward the spawn",
    }
    cams = {}
    for name, (p_, l_) in pinned.items():
        d_ = derived[name]
        cams[name] = {
            "pos": [round(float(v), 3) for v in p_],
            "look": [round(float(v), 3) for v in l_],
            "note": notes[name],
            "derived_pos": [round(float(v), 3) for v in d_[0]],
            "derived_look": [round(float(v), 3) for v in d_[1]],
            "verified_png": f"shots/{name}.png",
        }

    # --- v2: the stretch WP2 actually walks -------------------------------- #
    # WP2 found the wp0 end of the spine has a patchy collider and brush (the
    # robot passes through it), and spawns at wp4 instead, walking to wp16
    # (~12 m). These poses frame that stretch. Same pinning discipline: derived
    # from the waypoints, then checked with a SIGHTLINE clearance test -- march
    # the camera->target segment and compare its height to the 98th percentile
    # of the splat column within 0.5 m. A negative worst clearance means the
    # view is buried, which is exactly how the first establishing_v2 attempt
    # (14, 2.5, -1.5) failed: -1.37 m at (11.4, -1.5), the brush mound on the
    # -z bank. Every pose below has positive worst clearance and was rendered.
    w4 = np.array(spine[4]); w5 = np.array(spine[5])
    w6 = np.array(spine[6]); w8 = np.array(spine[8]); w16 = np.array(spine[16])
    f4 = w5 - w4
    f4[1] = 0.0
    f4 /= np.linalg.norm(f4)
    cams2 = {
        "establishing_v2": {
            "pos": [14.0, 3.5, 0.5], "look": [0.5, -0.87, -1.8],
            "note": "wide view down the wash from above the +x bank; frames wp4 "
                    "(7.4 m ahead, centre) through wp16. Worst sightline "
                    "clearance +0.68 m. The first try at (14, 2.5, -1.5) put the "
                    "-z brush mound across the right third and was rejected.",
        },
        "low_following_v2": {
            "pos": [8.44, -0.354, -0.55], "look": [2.68, -0.5, -1.259],
            "note": "0.6 m above wp4's floor (-0.954), 1.8 m behind wp4 and a "
                    "little off the centreline, looking down-spine past wp8. "
                    "Worst clearance +0.16 m. Offsetting the full 1.2 m to the "
                    "+z side puts the lens inside the bank (top -0.30 there).",
        },
        "follow_v2": {
            "pos": [7.131, 0.099, -0.698], "look": [2.68, -0.5, -1.259],
            "note": "chase: 2.5 m behind wp6 along the polyline, 1.0 m above "
                    "wp6's floor (-0.901), looking at wp8 raised 0.36 m to the "
                    "robot's body. Worst clearance +0.47 m.",
        },
        "lookback": dict(cams["lookback"]),
    }
    cams2["lookback"]["note"] = "unchanged from v1 (the hero); still frames the " \
                                "wp4..wp16 stretch from its far end"
    spawn2 = {
        "position": [round(float(w4[0]), 3), round(float(w4[1]), 3),
                     round(float(w4[2]), 3)],
        "floor_y": round(float(w4[1]), 3),
        "yaw_rad": round(math.atan2(float(f4[0]), float(f4[2])), 4),
        "forward": [round(float(v), 4) for v in f4],
        "waypoint_index": 4,
        "walk_to_index": 16,
        "walk_length_m": round(float(sum(
            math.dist(spine[k][::2], spine[k + 1][::2]) for k in range(4, 16))), 2),
        "note": "WP2's spawn: wp0's collider is patchy and brushy, wp4 is open rock.",
    }

    doc = {
        "frame": {
            "up": "Y",
            "note": "post-flip world (loader half-turn about X: y->-y, z->-z). "
                    "Feed --cam/--look to gaussian_splats.exe without --no-flip.",
            "yaw": "atan2(forward.x, forward.z), radians, about +Y",
            "anchor_supersplat": {"pos": [-11.82, 2.19, 1.66], "look": [-13.82, 2.09, 1.73]},
        },
        "source": {"asset": str(asset), "level": args.level, "cell_m": CELL,
                   "opacity_min": OPACITY_MIN, "floor_quantile": FLOOR_Q,
                   "rms_max_m": RMS_MAX, "slope_max_deg": SLOPE_MAX_DEG},
        "walkable_region": {
            "cell_size_m": CELL,
            "cell_count": len(chosen),
            "area_m2": round(len(chosen) * CELL * CELL, 2),
            "bounds_xz": {"min": [round(float(poly_min[0]), 3), round(float(poly_min[1]), 3)],
                          "max": [round(float(poly_max[0]), 3), round(float(poly_max[1]), 3)]},
            "height_range_y": [round(min(hs), 3), round(max(hs), 3)],
            "cells": [[round(float(x), 3), round(float(y), 3), round(float(z), 3)]
                      for (x, z), y in zip(cells_xz, hs)],
        },
        "spine": {
            "length_m": round(spine_len, 2),
            "waypoints": [[round(v, 3) for v in w] for w in spine],
        },
        "spawn": {
            "position": [round(float(a[0]), 3), round(floor_y, 3), round(float(a[2]), 3)],
            "floor_y": round(floor_y, 3),
            "yaw_rad": round(yaw, 4),
            "forward": [round(float(v), 4) for v in fwd],
        },
        "cameras": cams,
        "spawn_v2": spawn2,
        "cameras_v2": cams2,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for label, table in (("v1", cams), ("v2", cams2)):
        for name, c in table.items():
            print(f"  {label} {name:18s} --cam {c['pos'][0]},{c['pos'][1]},{c['pos'][2]} "
                  f"--look {c['look'][0]},{c['look'][1]},{c['look'][2]}")
    print(f"  spawn_v2 {spawn2['position']} yaw {spawn2['yaw_rad']} rad, "
          f"walk {spawn2['walk_length_m']} m")


if __name__ == "__main__":
    main()

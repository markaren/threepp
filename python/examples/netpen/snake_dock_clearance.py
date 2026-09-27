"""Clearance between the snake's rendered body and the cradle's solid parts, per telemetry row
(negative = the body passes through a part).

    python snake_dock_clearance.py D:/dev/snake_out/film/raw_seed0_telemetry.npz [more.npz ...]

Exit code 1 if any row overlaps any part. Body: per link, points along the link axis at the visual
radius (the head's lamp pods, the side thrusters' ducts on link 2 from the head, the tail cap to
-0.19 m). Cradle (snake_mission / snake_netpen): a tube of 6 bars + 4 frame rings (bore BORE_R), a
funnel at each end flaring over FUN_L to the MOUTH_R hoop, a charging pad under the middle.
Input: a scene telemetry npz (links + quats, visual +x along the body) or a CPU-harness npz (links
head first + axes + lats)."""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import snake_mission as M

LINK_L, MOD_R, LINK_R = 0.18, 0.049, 0.0525
BAR_R, BAR_RR, RING_RR = M.BORE_R + 0.012, 0.010, 0.012
BAR_ANG = np.radians(30.0 + 60.0 * np.arange(6))
RINGS = (-0.75, -0.25, 0.25, 0.75)
PAD = dict(a=0.0, h=-(BAR_R + 0.035), half=(0.12, 0.02, 0.06))    # along, up, side half extents


def rot(q, v):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    u = np.stack([x, y, z], -1)
    t = 2.0 * np.cross(u, v)
    return v + w[..., None] * t + np.cross(u, t)


def body_points(links, ax, lat):
    """(T, P, 3) points, (P,) radii; links head first."""
    up = np.cross(lat, ax)
    pts, rad = [], []
    n = links.shape[1]
    for i in range(n):
        xs = np.linspace(-0.09, 0.09, 7)
        r = np.full(7, MOD_R if i > 0 else LINK_R + 0.002)
        if i == n - 1:
            xt = np.linspace(-0.10, -0.19, 5)
            xs = np.concatenate([xs, xt])
            r = np.concatenate([r, np.maximum(MOD_R * (1.0 - (-xt - 0.062) / 0.12), 0.012)])
        for x, rr in zip(xs, r):
            pts.append(links[:, i] + x * ax[:, i]); rad.append(rr)
        if i == 0:
            for side in (-1.0, 1.0):
                for dx in (-0.017, 0.017):
                    pts.append(links[:, 0] + dx * ax[:, 0] - 0.004 * up[:, 0] + side * (LINK_R + 0.010) * lat[:, 0])
                    rad.append(0.011)
        if i == n - 1 - M.SIDE_THRUSTER_LINK:              # scene index of the side-thruster module
            for side in (-1.0, 1.0):
                for dx in (-0.035, 0.0, 0.035):
                    pts.append(links[:, i] + dx * ax[:, i] + side * 0.079 * lat[:, i]); rad.append(0.03)
    return np.stack(pts, 1), np.array(rad)


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def clearance(links, ax, lat, dock_c, dock_u):
    upw = np.array([0.0, 1.0, 0.0])
    side = np.cross(dock_u, upw); side /= np.linalg.norm(side)
    P, R = body_points(links, ax, lat)
    d = P - dock_c
    a, s, h = d @ dock_u, d @ side, d @ upw
    rho = np.hypot(s, h)
    aa = np.abs(a)
    out = {}
    ar = np.stack([aa, rho], -1)
    out["funnel"] = seg_dist(ar, np.array([M.A_THROAT, M.BORE_R + 0.005]), np.array([0.5 * M.DOCK_LEN, M.MOUTH_R])) - R
    out["hoop"] = np.hypot(aa - 0.5 * M.DOCK_LEN, rho - M.MOUTH_R) - 0.017 - R
    sh = np.stack([s, h], -1)
    bars = np.full_like(a, 9.0)
    for th in BAR_ANG:
        q = BAR_R * np.array([math.cos(th), math.sin(th)])
        dd = np.hypot(np.linalg.norm(sh - q, axis=-1), np.maximum(aa - M.A_THROAT, 0.0)) - BAR_RR - R
        bars = np.minimum(bars, dd)
    out["tube_bar"] = bars
    rings = np.full_like(a, 9.0)
    for a0 in RINGS:
        rings = np.minimum(rings, np.hypot(a - a0, rho - BAR_R) - RING_RR - R)
    out["ring"] = rings
    q = np.stack([a - PAD["a"], h - PAD["h"], s], -1)
    ext = np.maximum(np.abs(q) - np.array(PAD["half"]), 0.0)
    out["pad"] = np.linalg.norm(ext, axis=-1) - R
    return out


def main(path):
    z = np.load(path, allow_pickle=True)
    if "axes" in z.files:                                 # the CPU harness
        t, ph = z["t"], z["phase"].astype(int)
        links, ax, lat = z["links"], z["axes"], z["lats"]
        phases = list(z["phases"])
    else:                                                 # the scene telemetry: visual +x along the body
        cols = list(z["cols"]); log = z["log"]
        t, ph = log[:, cols.index("t")], log[:, cols.index("phase")].astype(int)
        phases = list(z["phases"])
        links, q = z["links"].astype(float), z["quats"].astype(float)
        hx = log[:, cols.index("head_x")]
        if np.abs(links[:, 0, 0] - hx).mean() > np.abs(links[:, -1, 0] - hx).mean():
            links, q = links[:, ::-1], q[:, ::-1]
        ax, lat = rot(q, np.array([1.0, 0.0, 0.0])), rot(q, np.array([0.0, 0.0, 1.0]))
    out = clearance(links, ax, lat, z["dock_c"], z["dock_u"])
    ok = True
    for k, v in out.items():
        m = v.min(1)
        bad = m < 0
        if bad.any():
            ok = False
            idx = np.nonzero(bad)[0]
            spans, start = [], idx[0]
            for p_, q_ in zip(idx[:-1], idx[1:]):
                if q_ != p_ + 1:
                    spans.append((start, p_)); start = q_
            spans.append((start, idx[-1]))
            desc = ", ".join(f"{t[i0]:.1f}-{t[i1]:.1f}s {phases[ph[i0]]} ({m[i0:i1 + 1].min() * 100:.1f} cm)" for i0, i1 in spans[:6])
            print(f"  {k:9s} min {m.min() * 100:6.1f} cm  PENETRATES: {desc}")
        else:
            print(f"  {k:9s} min {m.min() * 100:6.1f} cm  ok")
    print(f"dock clearance {'OK' if ok else 'FAIL'}: {path}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([main(p) for p in sys.argv[1:]]) else 1)

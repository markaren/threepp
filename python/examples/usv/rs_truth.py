"""Ground truth for usv_radar_sonar.py's headless shots: where the radar and the
sonar put their echoes, against where things really are.

The truth here never comes from the sensors' rays. The terrain is the
GeoScene's height field (the survey the meshes were built from, sampled on its
own grid), the targets are the poses the demo gave them, and the sensors'
poses are recorded when they fired. What it measures:

  radar   every target's strongest echo: range and bearing error, and whether
          it was seen at all; per spoke, the first land echo against the first
          land on the height field along that bearing (the shoreline the scope
          should draw), allowing for the beam smearing a nearer headland into
          the spoke;
  sonar   per beam, the seabed's leading edge against where the vertical fan
          first meets the height field; every target's strongest echo against
          its true range and bearing.

Each check has a tolerance stated from the sensor's resolution, not tuned to
pass: the range resolution (c tau / 2, or the chirp's), the beamwidth, and the
size of the target.
"""
import math

import numpy as np


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def line_source_two_way_db(el_deg, null_deg):
    """Two-way gain of an untapered line source at `el_deg` off its axis, its
    first null at `null_deg` (the sonar's vertical stave)."""
    x = math.sin(math.radians(abs(el_deg))) / math.sin(math.radians(null_deg))
    if x >= 1.0:
        return -math.inf
    a = 1.0 if x < 1e-9 else math.sin(math.pi * x) / (math.pi * x)
    return 40.0 * math.log10(max(abs(a), 1e-12))


class HeightGrid:
    """The height field sampled once on a regular grid; bilinear after that."""

    def __init__(self, geo, cx, cz, half, step):
        n = int(math.ceil(2.0 * half / step)) + 1
        self.x0, self.z0, self.step, self.n = cx - half, cz - half, step, n
        xs = (self.x0 + step * np.arange(n)).tolist()
        zs = (self.z0 + step * np.arange(n)).tolist()
        h = geo.height_at
        self.h = np.array([[h(x, z) for x in xs] for z in zs], np.float32)

    def __call__(self, x, z):
        fx = (np.asarray(x, float) - self.x0) / self.step
        fz = (np.asarray(z, float) - self.z0) / self.step
        i = np.clip(np.floor(fx).astype(np.int64), 0, self.n - 2)
        j = np.clip(np.floor(fz).astype(np.int64), 0, self.n - 2)
        tx = np.clip(fx - i, 0.0, 1.0)
        tz = np.clip(fz - j, 0.0, 1.0)
        h = self.h
        return ((1 - tz) * ((1 - tx) * h[j, i] + tx * h[j, i + 1])
                + tz * ((1 - tx) * h[j + 1, i] + tx * h[j + 1, i + 1]))


# --------------------------------------------------------------------------- #
#  Radar
# --------------------------------------------------------------------------- #
def line_of_sight(grid, ax, ay, az, tx, ty, tz, step=2.0):
    """True if the straight line from the antenna to (tx, ty, tz) clears the
    height field (the terrain and the quays; not the waves)."""
    d = math.hypot(tx - ax, tz - az)
    n = max(2, int(d / step))
    f = (np.arange(1, n) / n)
    gx, gz = ax + f * (tx - ax), az + f * (tz - az)
    ray_y = ay + f * (ty - ay)
    return bool(np.all(grid(gx, gz) < ray_y))

def radar_truth(img, model, history, grid, targets, not_land, threshold_db):
    """`history`: (t, x, y, z) of the antenna, one per frame. `targets`: {id:
    (name, pose(t) -> (x, z), radius, top)}, `top` the target's highest point
    above the sea. `not_land`: label ids that are not terrain or buildings
    (sea, own ship, targets, pier decks and moored boats, bridges)."""
    snr, lab = img.snr_db, img.label
    times, wbear = img.time, img.world_bearing_deg
    spokes, bins = img.spokes, img.bins
    bin_w = (img.max_range - img.min_range) / bins
    rng = img.min_range + (np.arange(bins) + 0.5) * bin_w
    hist = np.asarray(history, float)
    ok = times >= 0.0
    ax = np.interp(times, hist[:, 0], hist[:, 1])
    ay = np.interp(times, hist[:, 0], hist[:, 2])
    az = np.interp(times, hist[:, 0], hist[:, 3])
    res = 0.5 * 299792458.0 * model.pulse_length          # the pulse's range resolution
    beam = model.horizontal_beamwidth
    above = snr > threshold_db
    out = {"spokes_filled": int(ok.sum()), "range_resolution_m": res, "beamwidth_deg": beam, "targets": {}}

    # ── Targets: the strongest echo of each, against its pose at that spoke ──
    last = int(np.argmax(np.where(ok, times, -1.0)))
    for ident, (name, pose, radius, top) in targets.items():
        cells = (lab == ident) & above & ok[:, None]
        tx0, tz0 = pose(float(times[last]))
        rec = {"name": name, "seen": bool(cells.any()), "cells": int(cells.sum()),
               "line_of_sight": line_of_sight(grid, ax[last], ay[last], az[last], tx0, model.sea_level + top, tz0)}
        if cells.any():
            s, j = np.unravel_index(int(np.argmax(np.where(cells, snr, -1e9))), snr.shape)
            tx, tz = pose(float(times[s]))
            dx, dz = tx - ax[s], tz - az[s]
            r_true = math.hypot(dx, dz)
            b_true = math.degrees(math.atan2(dx, -dz))
            rec.update(snr_db=float(snr[s, j]), range_m=float(rng[j]), range_true_m=r_true,
                       range_err_m=float(rng[j] - r_true), bearing_err_deg=float(wrap180(wbear[s] - b_true)),
                       range_tol_m=radius + 0.5 * res + bin_w,
                       bearing_tol_deg=0.5 * beam + math.degrees(radius / max(r_true, 1.0)))
            rec["ok"] = bool(abs(rec["range_err_m"]) <= rec["range_tol_m"]
                             and abs(rec["bearing_err_deg"]) <= rec["bearing_tol_deg"])
        # What the scope holds where the target really is: its bearing within
        # half a beamwidth, its range within half a pulse and its size.
        tb = np.empty(spokes)
        tr = np.empty(spokes)
        for s in np.where(ok)[0]:
            px, pz = pose(float(times[s]))
            tb[s] = math.degrees(math.atan2(px - ax[s], -(pz - az[s])))
            tr[s] = math.hypot(px - ax[s], pz - az[s])
        near_b = ok & (np.abs(wrap180(wbear - np.where(ok, tb, 0.0))) <= 0.5 * beam)
        if near_b.any():
            win = near_b[:, None] & (np.abs(rng[None, :] - np.where(ok, tr, 0.0)[:, None]) <= 0.5 * res + radius)
            if win.any():
                s2, j2 = np.unravel_index(int(np.argmax(np.where(win, snr, -1e9))), snr.shape)
                labs, cnt = np.unique(lab[win & above], return_counts=True)
                rec["at_truth"] = {"max_snr_db": float(snr[s2, j2]), "label_at_max": int(lab[s2, j2]),
                                   "range_true_m": float(tr[s2]),
                                   "labels_above_threshold": {int(a): int(b) for a, b in zip(labs, cnt)}}
        out["targets"][str(ident)] = rec

    # ── The coast, spoke by spoke ────────────────────────────────────────────
    step = 2.0
    rs = np.arange(img.min_range, img.max_range, step)
    th = np.radians(wbear)[:, None]
    hx = ax[:, None] + rs[None, :] * np.sin(th)
    hz = az[:, None] - rs[None, :] * np.cos(th)
    landed = grid(hx, hz) > 0.0
    first = np.where(landed.any(axis=1), np.argmax(landed, axis=1), -1)
    r_shore = np.where(first >= 0, rs[np.maximum(first, 0)], np.inf)
    # A nearer shore inside the beam paints into this spoke too: the smear window.
    m = int(math.ceil(0.5 * beam / (360.0 / spokes)))
    r_near = np.full(spokes, np.inf)
    for k in range(-m, m + 1):
        r_near = np.minimum(r_near, np.roll(r_shore, k))

    land_echo = above & ~np.isin(lab, np.asarray(sorted(not_land), np.int32)) & (lab >= 0)
    has = land_echo.any(axis=1)
    r_echo = np.where(has, rng[np.argmax(land_echo, axis=1)], np.inf)

    early_tol = res + bin_w          # a strong edge's CW response starts c tau / 2 before it
    late_tol = 15.0                  # the first hit sits up the shore's slope, not at the waterline
    coast = ok & np.isfinite(r_shore) & (r_shore < img.max_range - late_tol)
    early = coast & (r_echo < r_near - early_tol)
    at_edge = coast & (r_echo >= r_near - early_tol) & (r_echo <= r_shore + late_tol)
    behind = coast & (r_echo > r_shore + late_tol)
    err = (r_echo - r_shore)[coast & np.isfinite(r_echo)]

    # Where every land echo falls: on land, within one resolution cell (half a
    # pulse in range, half a beamwidth in bearing), or over open water.
    bx = ax[:, None] + rng[None, :] * np.sin(th)
    bz = az[:, None] - rng[None, :] * np.cos(th)
    land_bin = grid(bx, bz) > 0.0
    near = land_bin.copy()
    for k in range(1, int(math.ceil((0.5 * res + bin_w) / bin_w)) + 1):
        near[:, k:] |= land_bin[:, :-k]
        near[:, :-k] |= land_bin[:, k:]
    cell = near.copy()
    for k in range(1, m + 1):
        cell |= np.roll(near, k, axis=0) | np.roll(near, -k, axis=0)
    echoes = land_echo & ok[:, None]
    on_land = echoes & cell
    unlabelled = above & ok[:, None] & (lab == -1)
    out["coast"] = {
        "land_echo_cells": int(echoes.sum()),
        "on_land_share": float(on_land.sum() / max(1, echoes.sum())),
        "spokes_with_shore": int(coast.sum()),
        "first_echo_at_shore": int(at_edge.sum()),
        "first_echo_early": int(early.sum()),
        "first_echo_behind_shore": int(behind.sum()),
        "median_err_m": float(np.median(err)) if err.size else None,
        "p10_err_m": float(np.percentile(err, 10)) if err.size else None,
        "p90_err_m": float(np.percentile(err, 90)) if err.size else None,
        "unlabelled_cells_above_threshold": int(unlabelled.sum()),
        "early_tol_m": early_tol, "late_tol_m": late_tol,
    }
    # The worst early spokes, for a look at what they are.
    ex = np.where(early)[0]
    if ex.size:
        worst = ex[np.argsort((r_echo - r_near)[ex])[:5]]
        # Past the smear window but inside one full beamwidth: the shore that
        # comes nearest the echo's range, and how far off the spoke it lies.
        m2 = int(math.ceil(beam / (360.0 / spokes)))

        def side_shore(s):
            ks = np.arange(-m2, m2 + 1)
            r = r_shore[(s + ks) % spokes]
            k = int(ks[int(np.argmin(np.abs(r - r_echo[s])))])
            return wrap180(float(wbear[(s + k) % spokes] - wbear[s])), float(r_shore[(s + k) % spokes])

        out["coast"]["early_examples"] = []
        for s in worst:
            off, r_side = side_shore(s)
            out["coast"]["early_examples"].append(
                {"world_bearing_deg": float(wbear[s]), "echo_m": float(r_echo[s]), "shore_m": float(r_shore[s]),
                 "label": int(lab[s, int(np.argmax(land_echo[s]))]),
                 "beam_shore_offset_deg": off, "beam_shore_m": r_side})
    out["_arrays"] = {"r_shore": r_shore, "r_near": r_near, "r_echo": r_echo, "ax": ax, "az": az, "rng": rng}
    return out


def radar_bscan(img, truth, targets, threshold_db):
    targets = {k: v[:3] for k, v in targets.items()}
    """Spoke (across) by range (up) in SNR, with the height field's shoreline
    (cyan), the smear window's near edge (dim cyan) and each target's true
    range (magenta). RGB uint8."""
    snr = img.snr_db
    v = np.clip((snr - threshold_db) / 40.0, 0.0, 1.0)
    rgb = np.stack([v * 255, v * 230, v * 160], -1)
    a = truth["_arrays"]
    rng = a["rng"]
    bin_w = rng[1] - rng[0]
    for arr, col in ((a["r_near"], (0, 90, 110)), (a["r_shore"], (0, 230, 255))):
        j = np.round((arr - rng[0]) / bin_w).astype(np.int64)
        s = np.where(np.isfinite(arr) & (j >= 0) & (j < len(rng)))[0]
        for dj in (-1, 0, 1):
            rgb[s, np.clip(j[s] + dj, 0, len(rng) - 1)] = col
    for ident, (name, pose, radius) in targets.items():
        for s in range(0, img.spokes, 4):
            if img.time[s] < 0:
                continue
            tx, tz = pose(float(img.time[s]))
            dx, dz = tx - a["ax"][s], tz - a["az"][s]
            b = math.degrees(math.atan2(dx, -dz))
            if abs(wrap180(img.world_bearing_deg[s] - b)) < 1.0:
                j = int(round((math.hypot(dx, dz) - rng[0]) / bin_w))
                if 0 <= j < len(rng):
                    rgb[max(0, s - 3):s + 4, max(0, j - 3):j + 4] = (255, 0, 255)
    img8 = rgb.astype(np.uint8).transpose(1, 0, 2)[::-1]       # range up, spokes across
    return np.ascontiguousarray(img8[::2, ::2])


# --------------------------------------------------------------------------- #
#  Sonar
# --------------------------------------------------------------------------- #
def sonar_truth(frame, model, world, grid, objects, not_seabed=(), above_noise_db=10.0):
    """`world`: the sonar's (4, 4) world matrix when it pinged. `objects`:
    {id: (name, (x, y, z), radius)}. `not_seabed`: other ids that are not the
    bottom (the sea surface, the own hull)."""
    o = world[:3, 3]
    R = world[:3, :3] / np.linalg.norm(world[:3, :3], axis=0)[None, :]
    beams, bins = frame.beams, frame.bins
    bear = np.asarray(frame.bearings_deg, float)
    bin_w = (frame.max_range - frame.min_range) / bins
    rng = frame.min_range + (np.arange(bins) + 0.5) * bin_w
    lvl, lab, kind = frame.level_db, frame.label, frame.label_kind
    thr = frame.noise_level_db + above_noise_db
    out = {"noise_db": float(frame.noise_level_db), "threshold_db": float(thr), "objects": {}}

    # ── The seabed's leading edge, beam by beam ──────────────────────────────
    half3 = 0.5 * model.vertical_beamwidth
    null = math.degrees(math.asin(min(0.999, 2.26 * math.sin(math.radians(half3)))))
    els = np.radians(np.linspace(-null, null, 31))
    u = np.sin(np.radians(bear))
    rs = np.arange(max(0.3, frame.min_range), frame.max_range, 0.1)
    hit = np.full((beams, els.size), np.inf)
    for e, el in enumerate(els):
        dy = np.sin(el)
        dz = -np.sqrt(np.maximum(0.0, 1.0 - u * u - dy * dy))
        d = R @ np.stack([u, np.full(beams, dy), dz])            # (3, beams) world directions
        p = o[:, None, None] + d[:, :, None] * rs[None, None, :]
        below = p[1] < grid(p[0], p[2])
        hit[:, e] = np.where(below.any(axis=1), rs[np.argmax(below, axis=1)], np.inf)
    in3 = np.abs(np.degrees(els)) <= half3 + 1e-6
    r_first3 = hit[:, in3].min(axis=1)
    r_first_null = hit.min(axis=1)
    r_axis = hit[:, els.size // 2]

    known = np.asarray(sorted(set(objects) | set(not_seabed)), np.int32)
    bed = (lvl > thr) & (lab >= 0) & (kind == 1) & ~np.isin(lab, known)
    # Three bins in a row: the edge, not a speckle spike.
    run = bed[:, :-2] & bed[:, 1:-1] & bed[:, 2:]
    has = run.any(axis=1)
    r_edge = np.where(has, rng[np.argmax(run, axis=1)], np.inf)
    tol = 0.5
    seen = np.isfinite(r_first3) & (r_first3 < frame.max_range - 1.0)
    inside = seen & (r_edge >= r_first_null - tol) & (r_edge <= r_first3 + tol)
    err = (r_edge - r_first3)[seen & np.isfinite(r_edge)]
    out["seabed"] = {
        "beams_with_bottom": int(seen.sum()),
        "edge_between_null_and_3db": int(inside.sum()),
        "share": float(inside.sum() / max(1, seen.sum())),
        "median_err_vs_3db_m": float(np.median(err)) if err.size else None,
        "p10_err_m": float(np.percentile(err, 10)) if err.size else None,
        "p90_err_m": float(np.percentile(err, 90)) if err.size else None,
        "tol_m": tol, "vertical_null_deg": null,
    }

    # ── Objects: the strongest echo of each ──────────────────────────────────
    for ident, (name, pos, radius) in objects.items():
        cells = (lab == ident) & (lvl > thr) & (kind == 1)
        dv = R.T @ (np.asarray(pos, float) - o)
        r_true = float(np.linalg.norm(dv))
        b_true = math.degrees(math.asin(max(-1.0, min(1.0, dv[0] / max(r_true, 1e-6)))))
        # Inside the vertical fan: its centre within the pattern's first null?
        el = math.degrees(math.asin(max(-1.0, min(1.0, dv[1] / max(r_true, 1e-6)))))
        rec = {"name": name, "seen": bool(cells.any()), "cells": int(cells.sum()), "range_true_m": r_true,
               "bearing_true_deg": b_true, "elevation_deg": el, "in_fan": bool(abs(el) <= null),
               "in_main_lobe": bool(abs(el) <= half3), "pattern_two_way_db": line_source_two_way_db(el, null)}
        if cells.any():
            b, j = np.unravel_index(int(np.argmax(np.where(cells, lvl, -1e9))), lvl.shape)
            rec.update(level_db=float(lvl[b, j]), range_m=float(rng[j]), bearing_deg=float(bear[b]),
                       range_err_m=float(rng[j] - r_true), bearing_err_deg=float(bear[b] - b_true),
                       range_tol_m=radius + 2.0 * model.range_resolution + bin_w,
                       bearing_tol_deg=model.horizontal_beamwidth + math.degrees(radius / max(r_true, 0.5)))
            rec["ok"] = bool(abs(rec["range_err_m"]) <= rec["range_tol_m"]
                             and abs(rec["bearing_err_deg"]) <= rec["bearing_tol_deg"])
        out["objects"][str(ident)] = rec
    out["_arrays"] = {"r_first3": r_first3, "r_first_null": r_first_null, "r_axis": r_axis, "r_edge": r_edge,
                      "rng": rng}
    return out


def sonar_bscan(frame, truth, objects_true):
    """Beam (across) by range (up) in echo level above noise, with the vertical
    fan's first meeting with the seabed at its -3 dB edge (cyan) and its first
    null (dim cyan), the beam axis's (yellow), the measured edge (green ticks)
    and the targets' true positions (magenta). RGB uint8."""
    lvl = frame.level_db
    v = np.clip((lvl - frame.noise_level_db) / 50.0, 0.0, 1.0)
    rgb = np.stack([v * 255, v * 200, v * 150], -1)
    a = truth["_arrays"]
    rng = a["rng"]
    bin_w = rng[1] - rng[0]
    beams = frame.beams
    for key, col in (("r_first_null", (0, 90, 110)), ("r_first3", (0, 230, 255)), ("r_axis", (240, 220, 0)),
                     ("r_edge", (40, 255, 60))):
        arr = a[key]
        j = np.round((np.where(np.isfinite(arr), arr, rng[0] - bin_w) - rng[0]) / bin_w).astype(np.int64)
        b = np.where((j >= 0) & (j < len(rng)))[0]
        if key == "r_edge":
            b = b[b % 4 == 0]
        rgb[b, j[b]] = col
    bear = np.asarray(frame.bearings_deg, float)
    for name, b_true, r_true in objects_true:
        bi = int(np.argmin(np.abs(bear - b_true)))
        j = int(round((r_true - rng[0]) / bin_w))
        if 0 <= j < len(rng):
            rgb[max(0, bi - 4):bi + 5, j] = (255, 0, 255)
            rgb[bi, max(0, j - 6):j + 7] = (255, 0, 255)
    return np.ascontiguousarray(rgb.astype(np.uint8).transpose(1, 0, 2)[::-1])


def strip_arrays(d):
    """The JSON-able part of a truth dict."""
    if isinstance(d, dict):
        return {str(k): strip_arrays(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, float) and not math.isfinite(d):
        return None
    return d

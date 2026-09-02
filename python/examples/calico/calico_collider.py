"""A waterproof collider for the Calico Tanks scan, split off from the sensor mesh.

The demo used to give PhysX the raw `bakeSurface` marching-cubes shell. That shell
is the right SENSOR surface -- it is what the scan looks like, brush and all -- and
the wrong COLLIDER, for three separate reasons, all of which the user saw:

  * it has HOLES, wherever the bake's poses did not agree (weight floor 2, brush
    occluding the rock behind it). A foot lands in a hole and the robot falls
    through the world.
  * it has ZERO THICKNESS. A fast foot plant at 2 ms substeps and no CCD tunnels
    straight through a single sheet of triangles.
  * it has SPIKES: twigs and single-pose slivers fused into near-vertical shards
    standing above the floor (WP2 measured 20 cm of fused brush over the rock at
    the original spawn). A foot that lands on one is kicked into the air.

So: keep the bake as the sensor surface, unchanged, and derive a SEPARATE collider
from it -- a 2.5D height grid, watertight by construction. One height per cell means
no hole is representable; the extraction picks the lowest RELIABLE surface per cell,
so brush never enters it; a slope clamp means no spike is representable either.

    build_height_grid(positions, indices, bounds)  ->  HeightGrid
    HeightGrid.to_geometry()                       ->  a regular-grid trimesh
    drop_test(world, grid, samples)                ->  the waterproofness metric

`drop_test` is the acceptance instrument, not an assert: it drops a small hard ball
on every 10 cm cell of the walkable corridor and reports where each one came to
rest relative to the grid. Holes and spikes are counted, not argued about, and the
same test run with the balls thrown DOWN at 4 m/s is the tunneling test.

Run it as a script to get the whole table plus the hole/spike maps:

    set PYTHONPATH=C:/dev/threepp/python
    py -3.14 python/examples/calico/calico_collider.py --bake-cache   # bake once
    py -3.14 python/examples/calico/calico_collider.py --out shots
"""
import argparse
import gc
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))       # repo/python

import threepp as tp

DEFAULT_ASSET = os.environ.get("THREEPP_CALICO_ASSET", "C:/dev/splats/calico_tanks")
BAKE_CACHE = os.path.join(DEFAULT_ASSET, "calico_bake_cache.npz")

# ── grid parameters ───────────────────────────────────────────────────────────
CELL = 0.05             # collider cell, matched to the bake's 5 cm voxel
MARGIN = 1.0            # metres of grid past the walkable region (never step off an edge)
FLOOR_CELL = 1.0        # the local-floor field's cell, as in calico_slam_pip
# Reject samples more than this far over the local floor. 0.40 was the first
# guess and it is not tight enough at the wp0 end of the spine: the cell under
# Spot's front-left foot there holds 17 vertices in a 5 cm band at +0.33..+0.38
# with NOTHING under them -- an isolated brush shelf whose rock the bake never
# saw -- and the local floor of its 1 m cell is +0.05, so 0.40 lets it through.
# The result is a 30 cm bank inside one footprint and a robot that topples on
# the spot. 0.30 cuts that cell to +0.19 and moves the wp4 spawn by 0.000 m.
BRUSH_ABOVE = 0.30
LOW_PCT = 20.0          # per-cell percentile: the lowest RELIABLE surface, not the min
SLOPE_DEG = 50.0        # neighbour-to-neighbour slope clamp
CORRIDOR_HALF_W = 1.25  # half width of the walkable corridor around the spine


# ── the grid ──────────────────────────────────────────────────────────────────
class HeightGrid:
    """A regular XY height field: H[i, j] is the collider's z at (x0 + i*cell, y0 + j*cell).

    Every cell is finite -- that is the whole point. `stats` carries the build's
    provenance (how many cells came from samples, how many were inpainted, how much
    the median filter and the slope clamp moved).
    """

    def __init__(self, H, x0, y0, cell, stats=None):
        self.H = np.ascontiguousarray(H, np.float32)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.cell = float(cell)
        self.stats = dict(stats or {})

    @property
    def shape(self):
        return self.H.shape

    @property
    def bounds(self):
        nx, ny = self.H.shape
        return (self.x0, self.x0 + (nx - 1) * self.cell,
                self.y0, self.y0 + (ny - 1) * self.cell)

    def height_at(self, x, y):
        """Bilinear height, clamped to the edge outside the grid. Array or scalar."""
        nx, ny = self.H.shape
        fx = np.clip((np.asarray(x, np.float64) - self.x0) / self.cell, 0.0, nx - 1.0001)
        fy = np.clip((np.asarray(y, np.float64) - self.y0) / self.cell, 0.0, ny - 1.0001)
        ix = fx.astype(np.int64); iy = fy.astype(np.int64)
        tx = fx - ix; ty = fy - iy
        H = self.H
        h = ((1 - tx) * (1 - ty) * H[ix, iy] + tx * (1 - ty) * H[ix + 1, iy] +
             (1 - tx) * ty * H[ix, iy + 1] + tx * ty * H[ix + 1, iy + 1])
        return float(h) if np.isscalar(x) or np.ndim(x) == 0 else h

    def to_geometry(self):
        """A non-indexed regular-grid triangle soup, `build_hf_geom` style.

        (nx-1)*(ny-1) quads, two triangles each, wound counter-clockwise seen from
        +z so the normals point up. `add_static_trimesh` auto-indexes a soup.
        """
        nx, ny = self.H.shape
        xs = self.x0 + np.arange(nx, dtype=np.float64) * self.cell
        ys = self.y0 + np.arange(ny, dtype=np.float64) * self.cell
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        V = np.stack([X, Y, self.H.astype(np.float64)], axis=-1)      # [nx, ny, 3]
        a = V[:-1, :-1]; b = V[1:, :-1]; c = V[1:, 1:]; d = V[:-1, 1:]
        soup = np.concatenate([np.stack([a, b, c], axis=2),
                               np.stack([a, c, d], axis=2)], axis=2)  # [nx-1, ny-1, 6, 3]
        g = tp.BufferGeometry()
        g.set_attribute("position", np.ascontiguousarray(soup.reshape(-1, 3), np.float32))
        g.compute_vertex_normals()
        return g

    def to_mesh(self, material=None):
        return tp.Mesh(self.to_geometry(), material or tp.MeshStandardMaterial())

    def save(self, path):
        np.savez_compressed(path, H=self.H, x0=self.x0, y0=self.y0, cell=self.cell)

    @staticmethod
    def load(path):
        z = np.load(path)
        return HeightGrid(z["H"], float(z["x0"]), float(z["y0"]), float(z["cell"]))


# ── the build ─────────────────────────────────────────────────────────────────
def _sample_triangles(positions, indices):
    """Vertices PLUS the centroid and three edge midpoints of every triangle.

    A 5 cm bake has triangles up to ~7 cm across, so on a 5 cm grid there are cells
    that a triangle crosses without putting a vertex in. Those cells come back empty
    and get inpainted from a neighbour, which is fine in the middle of a slab and
    wrong on a step. Four extra samples per triangle close the gap with no random
    numbers (this build has to stay deterministic -- the film depends on it).
    """
    P = np.asarray(positions, np.float64)
    if indices is None or len(indices) == 0:
        return P
    T = P[np.asarray(indices, np.int64)]                    # [M, 3, 3]
    mid = np.concatenate([(T[:, 0] + T[:, 1]) * 0.5,
                          (T[:, 1] + T[:, 2]) * 0.5,
                          (T[:, 2] + T[:, 0]) * 0.5], axis=0)
    cen = T.mean(axis=1)
    return np.concatenate([P, cen, mid], axis=0)


def _local_floor(S, cell, bounds):
    """Per-cell minimum over a coarse grid, 3x3-smoothed. The trail's own datum.

    Same construction as `SlamSurface._floor_grid`: a cell's minimum is its floor
    (brush and boulders sit above it and cannot pull it down), and the 3x3 mean of
    those minima is a surface, not a staircase, so a threshold measured against it
    means the same thing on the wash floor as on the slab a metre higher up.
    """
    x0, x1, y0, y1 = bounds
    nx = int(math.ceil((x1 - x0) / cell)) + 1
    ny = int(math.ceil((y1 - y0) / cell)) + 1
    ix = np.clip(((S[:, 0] - x0) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((S[:, 1] - y0) / cell).astype(np.int64), 0, ny - 1)
    F = np.full(nx * ny, np.inf, np.float64)
    np.minimum.at(F, ix * ny + iy, S[:, 2])
    F = F.reshape(nx, ny)
    F[~np.isfinite(F)] = np.nan
    pad = np.full((nx + 2, ny + 2), np.nan, np.float64)
    pad[1:-1, 1:-1] = F
    stack = np.stack([pad[a:a + nx, b:b + ny] for a in range(3) for b in range(3)])
    with np.errstate(invalid="ignore"):
        F = np.nanmean(stack, axis=0)
    F[~np.isfinite(F)] = float(np.median(S[:, 2]))
    return F, x0, y0, cell, nx, ny


def _floor_lookup(fl, x, y):
    F, x0, y0, cell, nx, ny = fl
    ix = np.clip(((x - x0) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell).astype(np.int64), 0, ny - 1)
    return F[ix, iy]


def _cell_percentile(key, z, ncells, pct):
    """Nearest-rank percentile of z within each cell key. NaN where a cell is empty.

    np.lexsort is stable, so two samples at the same height in the same cell keep
    their input order and the result does not depend on anything but the inputs.
    """
    out = np.full(ncells, np.nan, np.float64)
    if key.size == 0:
        return out
    order = np.lexsort((z, key))
    ks = key[order]; zs = z[order]
    uk, starts, counts = np.unique(ks, return_index=True, return_counts=True)
    rank = np.floor(pct / 100.0 * (counts - 1) + 0.5).astype(np.int64)
    out[uk] = zs[starts + rank]
    return out


def _inpaint_nearest(H):
    """Fill every non-finite cell from the nearest finite one. -> (H, n_filled).

    scipy's exact EDT when it is there; otherwise an iterative dilation that takes
    the mean of a hole's finite 4-neighbours and repeats. Both are deterministic;
    the dilation is a little smoother at the centre of a big hole, which is the
    conservative direction for a floor.
    """
    valid = np.isfinite(H)
    n_hole = int((~valid).sum())
    if n_hole == 0:
        return H, 0
    if valid.sum() == 0:
        raise ValueError("height grid has no valid cell at all -- the bake missed the region")
    try:
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        return H[tuple(idx)], n_hole
    except Exception:
        pass
    F = H.copy()
    for _ in range(10000):
        bad = ~np.isfinite(F)
        if not bad.any():
            break
        pad = np.full((F.shape[0] + 2, F.shape[1] + 2), np.nan, np.float64)
        pad[1:-1, 1:-1] = F
        nb = np.stack([pad[0:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, 0:-2], pad[1:-1, 2:]])
        with np.errstate(invalid="ignore"):
            m = np.nanmean(nb, axis=0)
        fillable = bad & np.isfinite(m)
        if not fillable.any():                    # nothing adjacent to a valid cell
            F[bad] = float(np.nanmedian(F))
            break
        F[fillable] = m[fillable]
    return F, n_hole


def _median3(H):
    pad = np.pad(H, 1, mode="edge")
    stack = np.stack([pad[a:a + H.shape[0], b:b + H.shape[1]]
                      for a in range(3) for b in range(3)])
    return np.median(stack, axis=0)


def _slope_clamp(H, maxd, rounds=4):
    """Lower any cell that stands more than `maxd` above a 4-neighbour, and repeat.

    Only DOWNWARD: a spike is a launch hazard and gets shaved to a walkable slope,
    while a pit is left alone (raising pits would build walls out of the wash's own
    drainage). Four rounds of the four sweeps propagate a clamp diagonally as far as
    anything on this trail needs.
    """
    F = H.copy()
    nx, ny = F.shape
    for _ in range(rounds):
        before = F.copy()
        for i in range(1, nx):
            np.minimum(F[i], F[i - 1] + maxd, out=F[i])
        for i in range(nx - 2, -1, -1):
            np.minimum(F[i], F[i + 1] + maxd, out=F[i])
        for j in range(1, ny):
            np.minimum(F[:, j], F[:, j - 1] + maxd, out=F[:, j])
        for j in range(ny - 2, -1, -1):
            np.minimum(F[:, j], F[:, j + 1] + maxd, out=F[:, j])
        if np.array_equal(before, F):
            break
    return F


def build_height_grid(positions, indices=None, bounds=None, cell=CELL, margin=MARGIN,
                      floor_cell=FLOOR_CELL, brush_above=BRUSH_ABOVE, low_pct=LOW_PCT,
                      slope_deg=SLOPE_DEG, verbose=True):
    """The bake's triangles -> a watertight 2.5D collider.

    `bounds` is the WALKABLE region (x0, x1, y0, y1); the grid is built `margin`
    metres beyond it on every side so the robot never walks off an edge, and cells
    outside the sampled data inherit the nearest sampled height (an edge clamp, not
    a cliff).

    The pipeline, in order, and why each step is there:

      1. sample the surface (vertices + triangle centroids + edge midpoints)
      2. local floor: 1 m-cell minima, 3x3 smoothed
      3. REJECT samples more than `brush_above` over that floor  -> no brush
      4. per-cell `low_pct`-th percentile                        -> the lowest
         RELIABLE surface, immune to the one stray sliver a pure minimum takes
      5. nearest-valid inpainting                                -> no holes
      6. 3x3 median                                              -> no single-cell spikes
      7. slope clamp at tan(slope_deg)*cell                      -> no ridges either
    """
    t0 = time.perf_counter()
    S = _sample_triangles(positions, indices)
    if bounds is None:
        bounds = (float(S[:, 0].min()), float(S[:, 0].max()),
                  float(S[:, 1].min()), float(S[:, 1].max()))
    gx0 = bounds[0] - margin; gx1 = bounds[1] + margin
    gy0 = bounds[2] - margin; gy1 = bounds[3] + margin
    nx = int(math.floor((gx1 - gx0) / cell)) + 1
    ny = int(math.floor((gy1 - gy0) / cell)) + 1

    keep = ((S[:, 0] >= gx0 - cell) & (S[:, 0] <= gx0 + nx * cell) &
            (S[:, 1] >= gy0 - cell) & (S[:, 1] <= gy0 + ny * cell))
    S = S[keep]
    n_in = S.shape[0]
    if n_in == 0:
        raise ValueError("no baked geometry inside the requested bounds")

    fl = _local_floor(S, floor_cell, (gx0, gx0 + nx * cell, gy0, gy0 + ny * cell))
    dz = S[:, 2] - _floor_lookup(fl, S[:, 0], S[:, 1])
    on_floor = dz <= brush_above
    n_brush = int((~on_floor).sum())
    S = S[on_floor]

    ix = np.clip(((S[:, 0] - gx0) / cell + 0.5).astype(np.int64), 0, nx - 1)
    iy = np.clip(((S[:, 1] - gy0) / cell + 0.5).astype(np.int64), 0, ny - 1)
    H = _cell_percentile(ix * ny + iy, S[:, 2], nx * ny, low_pct).reshape(nx, ny)
    n_seen = int(np.isfinite(H).sum())

    H, n_filled = _inpaint_nearest(H)
    Hm = _median3(H)
    med_moved = float(np.abs(Hm - H).max())
    H = Hm
    maxd = math.tan(math.radians(slope_deg)) * cell
    Hs = _slope_clamp(H, maxd)
    clamp_moved = float(np.abs(H - Hs).max())
    n_clamped = int((np.abs(H - Hs) > 1e-6).sum())
    H = Hs

    stats = {
        "cells": nx * ny, "cells_seen": n_seen, "cells_filled": n_filled,
        "samples": n_in, "samples_brush": n_brush,
        "median_max_move": med_moved, "slope_cells": n_clamped,
        "slope_max_move": clamp_moved, "max_step": maxd,
        "build_s": time.perf_counter() - t0,
    }
    if verbose:
        print(f"[grid] {nx}x{ny} cells at {cell} m over "
              f"x {gx0:+.2f}..{gx0 + (nx - 1) * cell:+.2f}  "
              f"y {gy0:+.2f}..{gy0 + (ny - 1) * cell:+.2f}")
        print(f"[grid]   {n_in} samples, {n_brush} ({100.0 * n_brush / max(n_in, 1):.1f}%) "
              f"rejected as > {brush_above} m over the local floor")
        print(f"[grid]   {n_seen}/{nx * ny} cells measured ({100.0 * n_seen / (nx * ny):.1f}%), "
              f"{n_filled} inpainted")
        print(f"[grid]   median moved <= {med_moved:.3f} m; slope clamp "
              f"(<= {maxd * 100:.1f} cm/cell = {slope_deg:.0f} deg) touched {n_clamped} cells, "
              f"max {clamp_moved:.3f} m")
        print(f"[grid]   z {H.min():+.2f}..{H.max():+.2f}, built in {stats['build_s']:.2f}s")
    return HeightGrid(H, gx0, gy0, cell, stats)


# ── the walkable corridor ─────────────────────────────────────────────────────
def corridor_samples(spine, half_w=1.25, spacing=0.1):
    """Every `spacing` cell of the grid whose centre is within `half_w` of the spine.

    The spine is a polyline; a cell is in the corridor when its distance to the
    nearest SEGMENT is under half_w. Returned sorted, so the drop test's ordering
    (and therefore its PNG and its counts) does not depend on anything else.
    """
    P = np.asarray(spine, np.float64)[:, :2]
    x0, x1 = P[:, 0].min() - half_w, P[:, 0].max() + half_w
    y0, y1 = P[:, 1].min() - half_w, P[:, 1].max() + half_w
    xs = np.arange(x0, x1 + spacing * 0.5, spacing)
    ys = np.arange(y0, y1 + spacing * 0.5, spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Q = np.stack([X.ravel(), Y.ravel()], axis=1)
    d2 = np.full(Q.shape[0], np.inf)
    for a, b in zip(P[:-1], P[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((Q - a) @ ab) / max(L2, 1e-9), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = np.minimum(d2, ((Q - proj) ** 2).sum(axis=1))
    m = d2 <= half_w * half_w
    return np.ascontiguousarray(Q[m]), (xs, ys, m.reshape(X.shape))


# ── the drop test ─────────────────────────────────────────────────────────────
PROBE_HALF = 0.03       # a 6 cm cube, about a Spot foot
PROBE_WINDOW = 0.8      # seconds from release to the reading


def drop_test(world, grid, samples, half=PROBE_HALF, drop_h=0.5, vz0=0.0,
              t_window=PROBE_WINDOW, dt=0.02, batch=1200, verbose=True, label=""):
    """Drop a hard probe on every sample point and report its height minus the grid's.

    This is the waterproofness metric. `vz0` < 0 throws the probes down instead of
    releasing them: at -4 m/s a probe crosses 8 mm per 2 ms substep, which is what a
    fast foot plant does, and a zero-thickness sheet lets it through.

    TWO deliberate choices, both MEASURED rather than assumed:

      * a CUBE, not a ball. A 3 cm ball on this trail lands correctly (dz within
        1.4 cm at t = 0.34 s) and then ROLLS: at t = 2 s the same ten probes read
        -0.11 .. -1.07, and at t = 4 s one of them had rolled clear off the grid to
        -19.7 m. "Rest height" on a real wash measures the SLOPE, not the collider.
        A 6 cm cube on a grippy material stays where it lands, which is also what a
        foot does.
      * a fixed 0.8 s WINDOW rather than "step to rest". 0.8 s is past the 0.33 s
        free fall from 0.5 m with room for the contact to settle, and a probe that
        found nothing has fallen 3.1 m by then -- five times the hole threshold. It
        is not long enough for anything to creep downhill.

    Probes go in batches so PhysX is never asked for ten thousand actors at once;
    they are 10 cm apart and 6 cm across so they never touch each other.
    """
    S = np.asarray(samples, np.float64)
    gh = grid.height_at(S[:, 0], S[:, 1])
    out = np.empty(S.shape[0], np.float64)
    geom = tp.BoxGeometry(2 * half, 2 * half, 2 * half)
    mat = tp.MeshStandardMaterial()
    grippy = world.create_material(1.2, 1.2, 0.0)
    n = int(round(t_window / dt))
    t0 = time.perf_counter()
    for s in range(0, S.shape[0], batch):
        e = min(s + batch, S.shape[0])
        bodies = []
        for k in range(s, e):
            b = tp.Mesh(geom, mat)
            b.position.set(float(S[k, 0]), float(S[k, 1]), float(gh[k]) + drop_h + half)
            rb = world.add(b, 500.0, grippy)
            if vz0 != 0.0:
                rb.set_linear_velocity(tp.Vector3(0.0, 0.0, float(vz0)))
            bodies.append(rb)
        for _ in range(n):
            world.step(dt)
        for k, rb in zip(range(s, e), bodies):
            out[k] = float(rb.position.z) - half - gh[k]
            world.remove(rb)
        if verbose:
            print(f"  [drop{(' ' + label) if label else ''}] {e}/{S.shape[0]} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def classify(dz, hole=-0.5, spike=0.15):
    return dz < hole, dz > spike


def report(name, dz, hole=-0.5, spike=0.15):
    h, s = classify(dz, hole, spike)
    ok = int((~h & ~s).sum())
    n = dz.size
    row = {
        "name": name, "n": n,
        "holes": int(h.sum()), "spikes": int(s.sum()), "ok": ok,
        # A probe 1 m under the grid is THROUGH the collider, full stop; between
        # -0.5 and -1 m it could still be one that slid down a steep clamped face
        # inside the 0.8 s window, so the two are counted apart.
        "gone": int((dz < -1.0).sum()),
        "hole_pct": 100.0 * h.sum() / max(n, 1),
        "spike_pct": 100.0 * s.sum() / max(n, 1),
        "ok_pct": 100.0 * ok / max(n, 1),
        "p50": float(np.percentile(dz, 50)), "p99": float(np.percentile(dz, 99)),
        "worst_low": float(dz.min()), "worst_high": float(dz.max()),
    }
    return row


def print_table(rows, title):
    print()
    print(f"  {title}")
    print(f"  {'collider':<14}{'n':>7}{'ok %':>9}{'holes':>8}{'hole %':>9}"
          f"{'spikes':>8}{'spike %':>9}{'gone<-1m':>10}{'median dz':>11}{'lowest':>9}{'highest':>9}")
    for r in rows:
        print(f"  {r['name']:<14}{r['n']:>7}{r['ok_pct']:>9.2f}{r['holes']:>8}{r['hole_pct']:>9.2f}"
              f"{r['spikes']:>8}{r['spike_pct']:>9.2f}{r['gone']:>10}{r['p50']:>+11.3f}"
              f"{r['worst_low']:>+9.2f}{r['worst_high']:>+9.2f}")


def hole_spike_png(path, samples, dz, grid_xy=None, hole=-0.5, spike=0.15, px=6, title=""):
    """green = the ball rested on the collider, red = hole, blue = spike.

    One pixel block per sample, x to the right and y UP, so the picture reads like
    a plan view of the trail.
    """
    from PIL import Image, ImageDraw
    S = np.asarray(samples, np.float64)
    sp = 0.1
    ix = np.round((S[:, 0] - S[:, 0].min()) / sp).astype(np.int64)
    iy = np.round((S[:, 1] - S[:, 1].min()) / sp).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    img = np.full((ny, nx, 3), 26, np.uint8)
    h, s = classify(dz, hole, spike)
    col = np.tile(np.array([[60, 170, 70]], np.uint8), (S.shape[0], 1))
    col[h] = (220, 50, 40)
    col[s] = (60, 110, 235)
    img[ny - 1 - iy, ix] = col
    im = Image.fromarray(img, "RGB").resize((nx * px, ny * px), Image.NEAREST)
    d = ImageDraw.Draw(im)
    if title:
        d.rectangle([0, 0, nx * px, 18], fill=(0, 0, 0))
        d.text((4, 4), title, fill=(255, 255, 255))
    im.save(path)
    return path


# ── the script: bake once, build both colliders, drop on each ─────────────────
def _load_frame_and_spine():
    """shots.json's Y-up waypoints in this demo's Z-up world.

    Deliberately NOT `from spot_calico import Frame`: importing that module drags in
    torch and the whole policy stack, and this script has to bake a 19 M-splat cloud
    in the same process. (The first version did import it and the bake died with
    "LLVM ERROR: out of memory".) The map is three lines and is quoted verbatim from
    spot_calico's header, which stays the single source of truth for it.
    """
    import json
    with open(os.path.join(_HERE, "shots.json")) as f:
        shots = json.load(f)
    sx, sy, sz = shots["spawn"]["position"]
    floor_y = float(shots["spawn"]["floor_y"])
    t = np.array([-sx, sz, -floor_y], np.float64)          # (x, y, z)_yup -> (x, -z, y)_zup + t

    def F(p):
        return (float(p[0]) + t[0], -float(p[2]) + t[1], float(p[1]) + t[2])

    F.t = t
    spine = np.array([F(w) for w in shots["spine"]["waypoints"]], np.float64)
    return shots, F, spine


def spine_bake_poses(wp, eye_up=1.6, side_m=4.0, side_up=0.5, side_every=3, fov=75.0):
    """The judged bake recipe (plans/calico-splat-demo.md, WP1 findings), verbatim.

    Kept here rather than imported from spot_calico for the same reason as above:
    this file must not pull torch into the baking process.
    """
    poses = []
    up = tp.Vector3(0, 0, 1)
    n = len(wp)
    for i, w in enumerate(wp):
        eye = np.array([w[0], w[1], w[2] + eye_up])
        nxt = wp[min(i + 1, n - 1)]
        prv = wp[max(i - 1, 0)]
        if i + 1 < n:
            poses.append(tp.BakePose(_v3(eye), _v3(nxt), up, fov))
        if i > 0:
            poses.append(tp.BakePose(_v3(eye), _v3(prv), up, fov))
        poses.append(tp.BakePose(_v3(eye), _v3(w), tp.Vector3(-1, 0, 0), fov))
        if i % side_every == 0:
            tan = np.array(nxt, float) - np.array(prv, float)
            tan[2] = 0.0
            nrm = np.linalg.norm(tan)
            tan = tan / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
            left = np.array([-tan[1], tan[0], 0.0])
            for s in (+1.0, -1.0):
                tgt = np.array(w, float) + left * (s * side_m) + [0, 0, side_up]
                poses.append(tp.BakePose(_v3(eye), _v3(tgt), up, fov))
    return poses


def _v3(p):
    return tp.Vector3(float(p[0]), float(p[1]), float(p[2]))


BAKE_VOXEL, BAKE_MAX_DEPTH = 0.05, 15.0


def bake_to_cache(asset, cache_path, spine, verbose=True):
    """Run the demo's own bake recipe once and cache its triangles to an npz.

    Identical to `spot_calico`'s: finest resident level forced into submit_ranges,
    the judged spine pose set, voxel 0.05, max_depth 15. Cached because it costs a
    Vulkan context, a 3 s SOG parse and ~25 s of fusing, and the collider work needs
    to iterate on the SAME triangles.
    """
    scene = tp.Scene()
    cloud = tp.SplatCloud.from_sog_lod(asset)
    cloud.rotation.x = math.pi + math.pi / 2
    _, F, _ = _load_frame_and_spine()
    cloud.position.set(float(F.t[0]), float(F.t[1]), float(F.t[2]))
    scene.add(cloud)
    lod = list(cloud.lod_levels)
    if lod:
        cloud.submit_ranges = [(int(lod[0]["base"]), int(lod[0]["count"]))]
    canvas = tp.Canvas("calico-collider-bake", width=960, height=600, headless=True, vsync=False)
    rend = tp.VulkanRenderer(canvas)
    cam = tp.PerspectiveCamera(75, 1.6, 0.05, 400)
    cam.up.set(0, 0, 1)
    cam.position.set(2.0, -2.0, 2.0)
    cam.look_at(-4.0, 1.0, 0.0)
    t0 = time.perf_counter()
    rend.render(scene, cam)
    if verbose:
        print(f"[bake] first render {time.perf_counter() - t0:.1f}s (shader compile included)")
    poses = spine_bake_poses(spine)
    t0 = time.perf_counter()
    surface = tp.bake_surface(rend, cloud, poses=poses,
                              voxel_size=BAKE_VOXEL, max_depth=BAKE_MAX_DEPTH)
    if verbose:
        print(f"[bake] {surface.triangle_count} triangles, {surface.vertex_count} vertices "
              f"in {time.perf_counter() - t0:.1f}s")
    np.savez_compressed(cache_path, positions=surface.positions, indices=surface.indices)
    canvas.close()
    return surface.positions, surface.indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--cache", default=BAKE_CACHE)
    ap.add_argument("--bake-cache", action="store_true",
                    help="run the bake and write the cache, then exit")
    ap.add_argument("--out", default=os.path.join(_HERE, "shots"))
    ap.add_argument("--cell", type=float, default=CELL)
    ap.add_argument("--half-w", type=float, default=CORRIDOR_HALF_W,
                    help="half width of the walkable corridor the drop test covers")
    ap.add_argument("--spacing", type=float, default=0.1)
    ap.add_argument("--colliders", default="bake,grid",
                    help="comma list of bake,grid,heightfield")
    ap.add_argument("--grid-out", default=None, help="also save the height grid as npz")
    args = ap.parse_args()

    shots, F, spine = _load_frame_and_spine()
    if args.bake_cache or not os.path.exists(args.cache):
        bake_to_cache(args.asset, args.cache, spine)
        if args.bake_cache:
            return
    z = np.load(args.cache)
    pos, idx = z["positions"], z["indices"]
    print(f"[bake] cache {args.cache}: {pos.shape[0]} vertices, {idx.shape[0]} triangles")

    samples, _ = corridor_samples(spine, args.half_w, args.spacing)
    # the WALKABLE region; build_height_grid adds its own 1 m of margin past this
    bx = (float(spine[:, 0].min()) - args.half_w, float(spine[:, 0].max()) + args.half_w)
    by = (float(spine[:, 1].min()) - args.half_w, float(spine[:, 1].max()) + args.half_w)
    grid = build_height_grid(pos, idx, bounds=(bx[0], bx[1], by[0], by[1]), cell=args.cell)
    if args.grid_out:
        grid.save(args.grid_out)
        print(f"[grid] saved {args.grid_out}")
    print(f"[drop] {samples.shape[0]} sample points over the "
          f"{args.half_w * 2:.1f} m-wide corridor at {args.spacing} m")

    os.makedirs(args.out, exist_ok=True)
    rows_rest, rows_fast = [], []
    for name in [c.strip() for c in args.colliders.split(",") if c.strip()]:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81),
                              fixed_timestep=0.002, max_substeps=20)
        if name == "bake":
            g = tp.BufferGeometry()
            g.set_attribute("position", np.ascontiguousarray(pos, np.float32))
            g.set_index(np.ascontiguousarray(idx.reshape(-1), np.uint32))
            body = world.add_static_trimesh(tp.Mesh(g, tp.MeshStandardMaterial()))
        elif name == "grid":
            body = world.add_static_trimesh(grid.to_mesh())
        elif name == "heightfield":
            if not hasattr(world, "add_static_heightfield"):
                print("[skip] heightfield: this pyd has no add_static_heightfield")
                continue
            body = world.add_static_heightfield(grid.H, cell=grid.cell,
                                                origin=tp.Vector3(grid.x0, grid.y0, 0.0))
        else:
            raise SystemExit(f"unknown collider {name!r}")
        assert body is not None, f"{name}: PhysX refused the collider"
        print(f"[{name}] collider in the world")
        for tag, vz0, rows in (("rest", 0.0, rows_rest), ("fast", -4.0, rows_fast)):
            dz = drop_test(world, grid, samples, vz0=vz0, label=f"{name}/{tag}")
            np.savez_compressed(os.path.join(args.out, f"hf_dz_{name}_{tag}.npz"),
                                dz=dz, xy=samples)
            r = report(f"{name}", dz)
            rows.append(r)
            p = os.path.join(args.out, f"hf_map_{name}_{tag}.png")
            hole_spike_png(p, samples, dz,
                           title=f"{name} / {tag}  ok {r['ok_pct']:.1f}%  "
                                 f"holes {r['holes']}  spikes {r['spikes']}")
            print(f"  -> {p}  ok {r['ok_pct']:.2f}%  holes {r['holes']}  spikes {r['spikes']}")
        # PhysX allows exactly one foundation per process, so the next collider's
        # world cannot be created until this one is actually gone.
        body = None
        del world
        gc.collect()

    print_table(rows_rest, "DROP TEST (released from 0.5 m, v0 = 0)")
    print_table(rows_fast, "TUNNELING TEST (released from 0.5 m at vz = -4 m/s)")


if __name__ == "__main__":
    main()

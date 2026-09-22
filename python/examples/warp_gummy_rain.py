"""Gummy rain: hundreds of soft jelly candies pour into a glass bowl. Warp + threepp.

Every candy is its own deformable body: a 3x3x3 lattice of particles held
together by XPBD distance constraints (axis edges, face and body diagonals,
and skip-one bending struts), solved with graph-coloured Gauss-Seidel. All
candies share one constraint template, so one colour is one launch across
every candy at once. The rendered candy is a rounded cube EMBEDDED in its
lattice: each surface vertex is a fixed triquadratic (27-node Lagrange)
combination of the lattice particles, so the surface bends, shears and
wobbles smoothly with the body.

Body-body contact: the lattice particles are spheres, found through a
wp.HashGrid rebuilt every substep, with position-level Coulomb friction. The
glass bowl and the table are analytic colliders. The whole substep loop is
captured once into a CUDA graph and replayed each frame.

    python warp_gummy_rain.py                 # window; drag to orbit
    python warp_gummy_rain.py --video 10      # headless Vulkan film -> C:/dev/_softbody_films/gummy_rain
    python warp_gummy_rain.py --video 10 --size 960x540 --n 300
    python warp_gummy_rain.py --glow-light    # the glow also lights the scene (much slower)
    python warp_gummy_rain.py --no-interop    # soup via numpy + update_attribute
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (Encoder, SoupInterop, accum_normals, cli_arg, load_font,
                         normalize_vec3, parse_size, resize_handler, scatter_soup)

VIDEO = cli_arg("--video", 0.0, float)
W, H = parse_size(cli_arg("--size", "1920x1080", str))
OUTDIR = cli_arg("--out", "C:/dev/_softbody_films/gummy_rain", str)
N_CANDY = cli_arg("--n", 600, int)
SUBSTEPS = cli_arg("--substeps", 20, int)
FPS = 60
DT = 1.0 / (FPS * SUBSTEPS)
TRANSMISSION = cli_arg("--transmission", 0.0, float)
EXPOSURE = cli_arg("--exposure", 0.9, float)
TRANSL = cli_arg("--transl", 0.9, float)
GLOW = cli_arg("--glow", 0.18, float)
ROUGH = cli_arg("--rough", 0.3, float)
CLEARCOAT = cli_arg("--clearcoat", 0.3, float)
GLOW_LIGHT = "--glow-light" in sys.argv      # candies' glow also lights the scene (slow)

# --- physical tunables -------------------------------------------------------
A = 0.042                    # nominal candy edge, m
R_CONTACT = A / 7.0          # contact sphere radius of every lattice particle
LAT_H = A / 2.0 - R_CONTACT  # lattice half extent of a nominal cube
SURF_E = (A / 2.0) / LAT_H   # render surface in lattice parameter units (> 1)
MASS_P = 1200.0 * A ** 3 / 27.0      # gelatine density
INV_M = 1.0 / MASS_P
COMPLIANCE = cli_arg("--compliance", 8e-4, float)   # m/N per distance constraint
DAMPING = 0.0004             # per substep
MU_PP = 0.5                  # candy-candy friction
MU_WALL = 0.25               # candy-glass / table friction
GRAVITY = wp.vec3(0.0, -9.81, 0.0)

# Bowl: a sphere section standing on the table.
R_IN = 0.30
THICK = 0.009
R_OUT = R_IN + THICK
BOWL_CY = 0.80 * R_OUT       # sphere centre height; outer sphere meets the table
Y_BIN = 0.016                # inner floor of the bowl (thick glass bottom)
Y_RIM = BOWL_CY + 0.22 * R_IN
R_MID = R_IN + 0.5 * THICK
# Candies stop a few mm short of the glass. The renderer starts each refracted
# ray 1 mm past the glass, so a candy face pressed flat against the inside of
# the bowl was skipped by the ray and rendered with a hole in its middle.
GLASS_GAP = 0.005

SPAWN_Y = 1.05
WAVE_EVERY = 6               # frames between release waves
POUR_START = 12              # frame of the first wave

# --- candy template -----------------------------------------------------------
nodes = np.array([(i - 1, j - 1, k - 1) for i in range(3) for j in range(3)
                  for k in range(3)], np.float32)          # parameter coords
NP = 27
pairs = []
for a in range(NP):
    for b in range(a + 1, NP):
        d = np.abs(nodes[a] - nodes[b])
        if d.max() == 1:
            pairs.append((a, b))
        elif d.max() == 2 and (d == 0).sum() == 2:          # skip-one bending strut
            pairs.append((a, b))
used = [0] * NP
col = []
for a, b in pairs:
    m = used[a] | used[b]
    c = 0
    while (m >> c) & 1:
        c += 1
    col.append(c)
    used[a] |= 1 << c
    used[b] |= 1 << c
col = np.array(col)
order = np.argsort(col, kind="stable")
n_colors = int(col.max()) + 1
color_count = np.bincount(col, minlength=n_colors)
color_start = np.concatenate(([0], np.cumsum(color_count)[:-1]))
ci_np = np.array([pairs[k][0] for k in order], np.int32)
cj_np = np.array([pairs[k][1] for k in order], np.int32)
dparam_np = (nodes[cj_np] - nodes[ci_np]).astype(np.float32)
N_CONS = len(pairs)


def rounded_cube(n, rho):
    """Rounded-cube surface in [-1,1]^3 parameter space: verts, faces (outward)."""
    t = -np.cos(np.linspace(0.0, math.pi, n + 1))          # clustered at the edges
    t = 0.55 * t + 0.45 * np.sign(t) * np.abs(t) ** 0.6    # soften the clustering
    verts, faces, index = [], [], {}

    def vid(p):
        key = tuple(np.round(p, 5))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]

    for axis in range(3):
        for sgn in (-1.0, 1.0):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            grid = np.empty((n + 1, n + 1), np.int64)
            for iu in range(n + 1):
                for iv in range(n + 1):
                    p = np.zeros(3)
                    p[axis] = sgn
                    p[u_ax] = t[iu]
                    p[v_ax] = t[iv]
                    grid[iu, iv] = vid(p)
            for iu in range(n):
                for iv in range(n):
                    a, b = grid[iu, iv], grid[iu + 1, iv]
                    c, d = grid[iu + 1, iv + 1], grid[iu, iv + 1]
                    faces.append((a, b, c))
                    faces.append((a, c, d))
    q = np.array(verts)
    core = np.clip(q, -(1 - rho), 1 - rho)
    off = q - core
    ln = np.linalg.norm(off, axis=1, keepdims=True)
    p = core + rho * off / np.maximum(ln, 1e-9)
    faces = np.array(faces, np.int32)
    # Outward winding.
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    flip = (nrm * (a + b + c)).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return p.astype(np.float32), faces


surf_p, surf_f = rounded_cube(cli_arg("--res", 6, int), 0.42)
surf_p *= SURF_E                                            # lattice parameter units
NV = len(surf_p)
NF = len(surf_f)


def lag(u):
    return np.stack([0.5 * u * (u - 1.0), 1.0 - u * u, 0.5 * u * (u + 1.0)], -1)


Lx, Ly, Lz = lag(surf_p[:, 0]), lag(surf_p[:, 1]), lag(surf_p[:, 2])
weights_np = np.einsum("vi,vj,vk->vijk", Lx, Ly, Lz).reshape(NV, 27).astype(np.float32)

# --- the candy population -----------------------------------------------------
rng = np.random.default_rng(7)
PALETTE = [(0.92, 0.03, 0.06),     # cherry
           (1.00, 0.30, 0.01),     # orange
           (1.00, 0.72, 0.02),     # lemon
           (0.18, 0.80, 0.06),     # lime
           (0.42, 0.06, 0.78)]     # grape
NG = len(PALETTE)
# One mesh per chunk of candies (the renderer device-lost on a single dynamic
# mesh of ~90k vertices; 60 candies = 23k is known good). Chunk c is colour c % NG.
per_group = cli_arg("--chunk", 60, int)
N_CHUNK = max(NG, int(round(N_CANDY / per_group / NG)) * NG)
N_CANDY = per_group * N_CHUNK
# Per-candy dimensions: mostly cubes, some bricks and slabs, a little size spread.
dims = np.ones((N_CANDY, 3), np.float32)
kind = rng.random(N_CANDY)
dims[kind < 0.25, rng.integers(0, 3)] = 1.45
dims[(kind >= 0.25) & (kind < 0.40), 1] = 0.7
dims *= rng.uniform(0.9, 1.1, (N_CANDY, 1)).astype(np.float32)
half_np = (dims * LAT_H).astype(np.float32)

# Release schedule: waves of slots on a disk above the bowl; order shuffled so
# every wave mixes colours.
slot_xy = []
for ring, count in ((0, 1), (1, 6), (2, 11)):
    for s in range(count):
        ang = 2 * math.pi * s / max(count, 1) + 0.3 * ring
        slot_xy.append((0.078 * ring * math.cos(ang), 0.078 * ring * math.sin(ang)))
slot_xy = np.array(slot_xy, np.float32)
perm = rng.permutation(N_CANDY)
release_frame = np.zeros(N_CANDY, np.int64)
spawn_pos = np.zeros((N_CANDY, 3), np.float32)
per_wave = len(slot_xy)
for rank, b in enumerate(perm):
    w, s = divmod(rank, per_wave)
    release_frame[b] = POUR_START + w * WAVE_EVERY
    swirl = 0.05 * np.array([math.cos(0.35 * w), math.sin(0.35 * w)])
    xy = slot_xy[s] + swirl + rng.uniform(-0.012, 0.012, 2)
    spawn_pos[b] = (xy[0], SPAWN_Y + rng.uniform(-0.02, 0.02), xy[1])
N_WAVES = (N_CANDY + per_wave - 1) // per_wave


def rand_quat(n):
    q = rng.normal(size=(n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


spawn_rot = rand_quat(N_CANDY)          # (x, y, z, w)
spawn_spin = rng.normal(0.0, 6.0, (N_CANDY, 3)).astype(np.float32)

# Parked (unreleased) candies wait far under the table, spread out.
park = np.zeros((N_CANDY, 3), np.float32)
park[:, 0] = (np.arange(N_CANDY) % 25) * 0.12 - 1.5
park[:, 2] = (np.arange(N_CANDY) // 25) * 0.12 - 1.5
park[:, 1] = -30.0
x0 = (park[:, None, :] + nodes[None, :, :] * half_np[:, None, :]).reshape(-1, 3)

# --- warp ---------------------------------------------------------------------
wp.init()
device = wp.get_preferred_device()
NPART = N_CANDY * NP
print(f"gummy rain: {N_CANDY} candies, {NPART} particles, {N_CANDY * N_CONS} constraints "
      f"in {n_colors} colours, {N_CANDY * NV} render verts, {N_CANDY * NF} tris on {device}")

x = wp.array(x0, dtype=wp.vec3, device=device)
prev = wp.array(x0, dtype=wp.vec3, device=device)
delta = wp.zeros(NPART, dtype=wp.vec3, device=device)
active = wp.zeros(N_CANDY, dtype=int, device=device)
half = wp.array(half_np, dtype=wp.vec3, device=device)
ci = wp.array(ci_np, dtype=int, device=device)
cj = wp.array(cj_np, dtype=int, device=device)
dparam = wp.array(dparam_np, dtype=wp.vec3, device=device)
weights = wp.array(weights_np.reshape(-1), dtype=float, device=device)
rpos = wp.zeros(N_CANDY * NV, dtype=wp.vec3, device=device)
rnrm = wp.zeros(N_CANDY * NV, dtype=wp.vec3, device=device)
tris_all = (surf_f[None, :, :] + (np.arange(N_CANDY) * NV)[:, None, None]).reshape(-1)
tris = wp.array(tris_all.astype(np.int32), dtype=int, device=device)
# The renderer draws a de-indexed triangle soup: the ray-traced hit shading
# (what the glass bowl refracts) read an indexed dynamic mesh's normals wrong.
soup_pos = wp.zeros(N_CANDY * NF * 3, dtype=wp.vec3, device=device)
soup_nrm = wp.zeros(N_CANDY * NF * 3, dtype=wp.vec3, device=device)
bowl_x = wp.zeros(1, dtype=float, device=device)
grid = wp.HashGrid(96, 96, 96, device=device)


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
            active: wp.array(dtype=int), dt: float, damping: float, g: wp.vec3):
    i = wp.tid()
    p = x[i]
    if active[i // 27] == 0:
        prev[i] = p
        return
    v = (p - prev[i]) * (1.0 - damping)
    prev[i] = p
    x[i] = p + v + g * dt * dt


@wp.kernel
def solve_color(x: wp.array(dtype=wp.vec3), ci: wp.array(dtype=int), cj: wp.array(dtype=int),
                dparam: wp.array(dtype=wp.vec3), half: wp.array(dtype=wp.vec3),
                start: int, count: int, alpha_t: float, w: float):
    t = wp.tid()
    b = t // count
    k = start + t % count
    i = b * 27 + ci[k]
    j = b * 27 + cj[k]
    hb = half[b]
    dp = dparam[k]
    rest = wp.length(wp.vec3(dp[0] * hb[0], dp[1] * hb[1], dp[2] * hb[2]))
    d = x[j] - x[i]
    l = wp.length(d)
    if l < 1.0e-9:
        return
    c = l - rest
    dl = -c / (2.0 * w + alpha_t)
    n = d / l
    x[i] = x[i] - n * (dl * w)
    x[j] = x[j] + n * (dl * w)


@wp.kernel
def contacts(grid: wp.uint64, x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
             active: wp.array(dtype=int), delta: wp.array(dtype=wp.vec3),
             r: float, mu: float):
    i = wp.tid()
    bi = i // 27
    if active[bi] == 0 or i % 27 == 13:
        delta[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    p = x[i]
    vi = p - prev[i]
    corr = wp.vec3(0.0, 0.0, 0.0)
    cnt = float(0.0)
    q = wp.hash_grid_query(grid, p, 2.0 * r)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j // 27 == bi or j % 27 == 13:
            continue
        d = p - x[j]
        dist = wp.length(d)
        if dist < 2.0 * r and dist > 1.0e-9:
            n = d / dist
            pen = 2.0 * r - dist
            c = n * (0.5 * pen)
            rel = vi - (x[j] - prev[j])
            tang = rel - n * wp.dot(rel, n)
            tl = wp.length(tang)
            if tl > 1.0e-9:
                c = c - tang * (0.5 * wp.min(mu * pen / tl, 1.0))
            corr = corr + c
            cnt = cnt + 1.0
    if cnt > 0.0:
        corr = corr * wp.min(1.0, 1.6 / cnt)
    delta[i] = corr


@wp.kernel
def apply_delta(x: wp.array(dtype=wp.vec3), delta: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    x[i] = x[i] + delta[i]


@wp.kernel
def walls(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
          active: wp.array(dtype=int), bowl_x: wp.array(dtype=float), r: float, mu: float):
    i = wp.tid()
    if active[i // 27] == 0:
        return
    p = x[i]
    bx = bowl_x[0]
    c = wp.vec3(bx, BOWL_CY, 0.0)
    hit = int(0)
    nrm = wp.vec3(0.0, 1.0, 0.0)
    rel = p - c
    d = wp.length(rel)
    if p[1] < Y_RIM:
        if d < R_MID:
            # inside the bowl
            if d > R_IN - r - GLASS_GAP:
                nrm = -rel / d
                p = c + rel * ((R_IN - r - GLASS_GAP) / d)
                hit = 1
            if p[1] < Y_BIN + r + GLASS_GAP:
                p = wp.vec3(p[0], Y_BIN + r + GLASS_GAP, p[2])
                nrm = wp.vec3(0.0, 1.0, 0.0)
                hit = 1
        else:
            if d < R_OUT + r + GLASS_GAP:
                nrm = rel / d
                p = c + rel * ((R_OUT + r + GLASS_GAP) / d)
                hit = 1
    # rim: a torus of tube radius THICK/2 at the top edge
    rr = wp.sqrt(wp.max(R_MID * R_MID - (Y_RIM - BOWL_CY) * (Y_RIM - BOWL_CY), 0.0))
    h = wp.vec3(p[0] - bx, 0.0, p[2])
    hl = wp.length(h)
    if hl > 1.0e-6:
        ring = wp.vec3(bx, Y_RIM, 0.0) + h * (rr / hl)
        e = p - ring
        el = wp.length(e)
        if el < 0.5 * THICK + r and el > 1.0e-9:
            nrm = e / el
            p = ring + nrm * (0.5 * THICK + r)
            hit = 1
    if p[1] < r:
        p = wp.vec3(p[0], r, p[2])
        nrm = wp.vec3(0.0, 1.0, 0.0)
        hit = 1
    if hit == 1:
        t = p - prev[i]
        t = t - nrm * wp.dot(t, nrm)
        p = p - t * mu
    x[i] = p


@wp.kernel
def release(ids: wp.array(dtype=int), spawn: wp.array(dtype=wp.vec3),
            rot: wp.array(dtype=wp.quat), spin: wp.array(dtype=wp.vec3),
            half: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
            prev: wp.array(dtype=wp.vec3), active: wp.array(dtype=int),
            v0: wp.vec3, dt: float):
    t = wp.tid()
    b = ids[t // 27]
    k = t % 27
    u = wp.vec3(float(k // 9 - 1), float((k // 3) % 3 - 1), float(k % 3 - 1))
    hb = half[b]
    local = wp.quat_rotate(rot[b], wp.vec3(u[0] * hb[0], u[1] * hb[1], u[2] * hb[2]))
    p = spawn[b] + local
    v = v0 + wp.cross(spin[b], local)
    x[b * 27 + k] = p
    prev[b * 27 + k] = p - v * dt
    if k == 0:
        active[b] = 1


@wp.kernel
def skin(x: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float), nv: int,
         out: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    b = t // nv
    v = t % nv
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(27):
        acc = acc + x[b * 27 + k] * weights[v * 27 + k]
    out[t] = acc


alpha_t = COMPLIANCE / (DT * DT)


def substeps():
    for _ in range(SUBSTEPS):
        wp.launch(predict, dim=NPART, device=device,
                  inputs=[x, prev, active, DT, DAMPING, GRAVITY])
        for c in range(n_colors):
            cnt = int(color_count[c])
            wp.launch(solve_color, dim=N_CANDY * cnt, device=device,
                      inputs=[x, ci, cj, dparam, half, int(color_start[c]), cnt,
                              alpha_t, INV_M])
        grid.build(x, 2.0 * R_CONTACT)
        wp.launch(contacts, dim=NPART, device=device,
                  inputs=[grid.id, x, prev, active, delta, R_CONTACT, MU_PP])
        wp.launch(apply_delta, dim=NPART, device=device, inputs=[x, delta])
        wp.launch(walls, dim=NPART, device=device,
                  inputs=[x, prev, active, bowl_x, R_CONTACT, MU_WALL])


def surface():
    wp.launch(skin, dim=N_CANDY * NV, device=device, inputs=[x, weights, NV, rpos])
    rnrm.zero_()
    wp.launch(accum_normals, dim=N_CANDY * NF, device=device, inputs=[rpos, tris, rnrm])
    wp.launch(normalize_vec3, dim=N_CANDY * NV, device=device, inputs=[rnrm])
    wp.launch(scatter_soup, dim=N_CANDY * NF * 3, device=device,
              inputs=[rpos, rnrm, tris, soup_pos, soup_nrm])


graph = None
if device.is_cuda and "--no-graph" not in sys.argv:
    try:
        grid.build(x, 2.0 * R_CONTACT)   # allocate before capture
        with wp.ScopedCapture(device) as cap:
            substeps()
            surface()
        graph = cap.graph
    except Exception as e:                      # noqa: BLE001
        print(f"graph capture failed ({e}); eager launches")
        graph = None

spawn_d = wp.array(spawn_pos, dtype=wp.vec3, device=device)
rot_d = wp.array(spawn_rot, dtype=wp.quat, device=device)
spin_d = wp.array(spawn_spin, dtype=wp.vec3, device=device)
frame_no = 0


def shake_offset(t):
    """The table gets a knock: the bowl slides and rings back (m)."""
    t0 = SHAKE_T
    if t < t0:
        return 0.0
    s = t - t0
    return 0.022 * math.sin(2 * math.pi * 3.2 * s) * math.exp(-s / 0.35) * min(s / 0.04, 1.0)


SHAKE_T = cli_arg("--shake", 6.6, float)


def step_frame():
    global frame_no
    ids = np.nonzero(release_frame == frame_no)[0].astype(np.int32)
    if len(ids):
        wp.launch(release, dim=len(ids) * 27, device=device,
                  inputs=[wp.array(ids, dtype=int, device=device), spawn_d, rot_d, spin_d,
                          half, x, prev, active, wp.vec3(0.0, -1.4, 0.0), DT])
    bx = shake_offset(frame_no / FPS)
    bowl_x.assign(np.float32([bx]))
    if graph is not None:
        wp.capture_launch(graph)
    else:
        substeps()
        surface()
    frame_no += 1
    soup.publish()
    bowl.position.x = bx


# --- scene ---------------------------------------------------------------------
headless = VIDEO > 0
canvas = tp.Canvas("threepp x warp - gummy rain", width=W, height=H,
                   antialiasing=4, vsync=False, headless=headless)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = EXPOSURE

scene = tp.Scene()


def studio_env(w=1024, h=512):
    """Dark studio with two warm softboxes and cool strip lights, as a float equirect."""
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    E, Z = np.meshgrid(elev, az, indexing="ij")
    y = np.sin(E)
    col = np.zeros((h, w, 3), np.float32)
    base = 0.035 + 0.05 * np.clip(y, 0, 1)
    col += base[..., None] * np.float32([1.0, 0.86, 0.74])

    def box(az0, az1, el0, el1, rgb, k):
        m = ((Z > az0) & (Z < az1) & (E > el0) & (E < el1)).astype(np.float32)
        return m[..., None] * np.float32(rgb) * k

    d = math.radians
    col += box(d(-25), d(20), d(48), d(78), (1.0, 0.86, 0.68), 9.0)      # key softbox
    col += box(d(140), d(150), d(28), d(62), (0.70, 0.82, 1.0), 12.0)    # rim strip
    col += box(d(-150), d(-140), d(28), d(62), (1.0, 0.78, 0.55), 12.0)  # warm rim strip
    col += box(d(70), d(110), d(30), d(50), (1.0, 0.92, 0.85), 3.0)      # side fill
    col[y < 0.35] *= 0.5
    col[y < 0] *= 0.4
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


env = studio_env()
scene.environment = env
scene.background = tp.Color(0.018, 0.014, 0.016)

key = tp.DirectionalLight(0xfff0dc, 3.2)
key.position.set(1.2, 3.0, 1.6)
key.cast_shadow = True
scene.add(key)

# Table: dark lacquered wood-ish, glossy enough to mirror the candies.
table = tp.Mesh(tp.BoxGeometry(60, 0.1, 60), tp.MeshStandardMaterial())
table.material.color = tp.Color(0.030, 0.019, 0.016)
table.material.roughness = 0.26
table.position.y = -0.05
table.receive_shadow = True
scene.add(table)


def lathe(segments, n_theta=160):
    """Revolve profile segments [(r, y, nr, ny) arrays] into one indexed mesh."""
    P, N, F = [], [], []
    th = np.linspace(0.0, 2 * math.pi, n_theta + 1)
    ct, st = np.cos(th), np.sin(th)
    base = 0
    for prof in segments:
        r, yv, nr, ny = [np.asarray(a, np.float64) for a in prof]
        m = len(r)
        px = r[:, None] * ct[None, :]
        pz = r[:, None] * st[None, :]
        py = np.repeat(yv[:, None], n_theta + 1, 1)
        nx = nr[:, None] * ct[None, :]
        nz = nr[:, None] * st[None, :]
        nyy = np.repeat(ny[:, None], n_theta + 1, 1)
        P.append(np.stack([px, py, pz], -1).reshape(-1, 3))
        N.append(np.stack([nx, nyy, nz], -1).reshape(-1, 3))
        idx = np.arange(m * (n_theta + 1)).reshape(m, n_theta + 1) + base
        a, b = idx[:-1, :-1], idx[1:, :-1]
        c, d2 = idx[1:, 1:], idx[:-1, 1:]
        F.append(np.stack([a, b, c], -1).reshape(-1, 3))
        F.append(np.stack([a, c, d2], -1).reshape(-1, 3))
        base += m * (n_theta + 1)
    P = np.concatenate(P).astype(np.float32)
    N = np.concatenate(N).astype(np.float32)
    F = np.concatenate(F).astype(np.int64)
    # Orient every triangle to agree with its supplied normals.
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    gn = np.cross(b - a, c - a)
    sn = N[F[:, 0]] + N[F[:, 1]] + N[F[:, 2]]
    flip = (gn * sn).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    keep = np.linalg.norm(gn, axis=1) > 1e-12
    F = F[keep]
    g = tp.BufferGeometry()
    g.set_attribute("position", P)
    g.set_attribute("normal", N)
    g.set_index(np.ascontiguousarray(F.reshape(-1), np.uint32))
    return g


def bowl_geometry():
    segs = []
    # inner floor disc
    r_bin = math.sqrt(R_IN ** 2 - (BOWL_CY - Y_BIN) ** 2)
    rr = np.linspace(0.0, r_bin, 12)
    segs.append((rr, np.full_like(rr, Y_BIN), np.zeros_like(rr), np.ones_like(rr)))
    # inner sphere, floor to rim
    a0 = math.atan2(Y_BIN - BOWL_CY, r_bin)
    a1 = math.atan2(Y_RIM - BOWL_CY, math.sqrt(R_IN ** 2 - (Y_RIM - BOWL_CY) ** 2))
    ang = np.linspace(a0, a1, 64)
    segs.append((R_IN * np.cos(ang), BOWL_CY + R_IN * np.sin(ang), -np.cos(ang), -np.sin(ang)))
    # rim: half torus from inner to outer, over the top
    rc = math.sqrt(R_MID ** 2 - (Y_RIM - BOWL_CY) ** 2)
    phi = np.linspace(math.pi, 0.0, 24)
    segs.append((rc + 0.5 * THICK * np.cos(phi), Y_RIM + 0.5 * THICK * np.sin(phi),
                 np.cos(phi), np.sin(phi)))
    # outer sphere, rim to table
    b1 = math.atan2(Y_RIM - BOWL_CY, math.sqrt(R_OUT ** 2 - (Y_RIM - BOWL_CY) ** 2))
    b0 = math.asin(-BOWL_CY / R_OUT)
    ang = np.linspace(b1, b0, 64)
    segs.append((R_OUT * np.cos(ang), BOWL_CY + R_OUT * np.sin(ang), np.cos(ang), np.sin(ang)))
    # foot
    rf = R_OUT * math.cos(b0)
    rr = np.linspace(rf, 0.0, 12)
    segs.append((rr, np.full_like(rr, 0.004), np.zeros_like(rr), -np.ones_like(rr)))
    return lathe(segs)


glass = tp.MeshPhysicalMaterial()
glass.color = tp.Color(0.97, 0.98, 1.0)
glass.roughness = 0.02
glass.metalness = 0.0
glass.transmission = 1.0
glass.ior = 1.5
bowl = tp.Mesh(bowl_geometry(), glass)
bowl.cast_shadow = False
scene.add(bowl)

surface()
wp.synchronize_device(device)
_rp0 = soup_pos.numpy()
_rn0 = soup_nrm.numpy()
geoms = []
meshes = []
f_local = (surf_f[None, :, :] + (np.arange(per_group) * NV)[:, None, None]).reshape(-1)
for g in range(N_CHUNK):
    rgb = PALETTE[g % NG]
    geo = tp.BufferGeometry()
    # Real (parked) candy shapes, not a collapsed point: the ray-tracing BLAS
    # is built from these and only refit afterwards.
    sl = slice(g * per_group * NF * 3, (g + 1) * per_group * NF * 3)
    geo.set_attribute("position", np.ascontiguousarray(_rp0[sl]))
    geo.set_attribute("normal", np.ascontiguousarray(_rn0[sl]))
    geo.set_attribute("uv", np.zeros((per_group * NF * 3, 2), np.float32))
    mat = tp.MeshPhysicalMaterial()
    mat.color = tp.Color(*rgb)
    mat.roughness = ROUGH
    mat.metalness = 0.0
    mat.ior = 1.45
    mat.clearcoat = CLEARCOAT
    mat.clearcoat_roughness = 0.08
    if GLOW > 0.0:
        # Gummies glow: light scatters through the gelatine and comes back out
        # in the candy's own colour.
        mat.emissive = tp.Color(*rgb)
        mat.emissive_intensity = GLOW
    if TRANSMISSION > 0.0:
        mat.transmission = TRANSMISSION
    elif TRANSL > 0.0:
        mat.translucency = TRANSL
        mat.translucency_color = tp.Color(*[min(1.0, 0.25 + v) for v in rgb])
    mesh = tp.Mesh(geo, mat)
    mesh.cast_shadow = True
    mesh.receive_shadow = True
    mesh.frustum_culled = False
    scene.add(mesh)
    geoms.append(geo)
    meshes.append(mesh)
    if not GLOW_LIGHT:
        # The glow is self-illumination, not a lamp: 259k emitter triangles cost
        # most of the frame when sampled as lights, and 16 coherent strata stood
        # in for all of them as point-like highlights on the table.
        renderer.set_emissive_casts_light(mesh, False)

# The soup goes straight from Warp into the renderer's vertex buffers (host copy
# only as a fallback, or with --no-interop).
soup = SoupInterop(renderer, device, soup_pos, soup_nrm,
                   [(meshes[g], g * per_group * NF * 3, per_group * NF * 3) for g in range(N_CHUNK)],
                   interop="--no-interop" not in sys.argv)

camera = tp.PerspectiveCamera(34, W / H, 0.02, 30)


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def camera_at(t, total):
    """High over the pour, sweeping down to bowl height as the pile settles."""
    s = smooth(t / total)
    az = math.radians(-35 + 70 * s)
    el = math.radians(52 - 44 * s)
    dist = 1.65 - 0.45 * s
    ty = 0.30 - 0.10 * s
    camera.position.set(dist * math.cos(el) * math.sin(az), ty + dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    camera.look_at(0.0, ty, 0.0)


def caption(rgb, font):
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    text = f"threepp + NVIDIA Warp  \u00b7  {N_CANDY} soft bodies  \u00b7  GPU XPBD"
    m = int(0.022 * H)
    dr.text((m, H - m - font.size), text, font=font, fill=(235, 225, 215))
    return np.asarray(im)


if VIDEO > 0:
    from PIL import Image
    os.makedirs(OUTDIR, exist_ok=True)
    tag = cli_arg("--tag", "gummy_rain", str)
    total = int(round(VIDEO * FPS))
    font = load_font(max(12, int(H * 0.020)))
    stills_at = [int(total * f) for f in (0.18, 0.40, 0.72, 0.92)]
    sheet_at = [int(total * (k + 0.5) / 9) for k in range(9)]
    sheet = []
    enc = Encoder(os.path.join(OUTDIR, f"{tag}.mp4"), W, H, FPS, crf=17, preset="slow")
    camera_at(0.0, VIDEO)
    for _ in range(30):
        renderer.render(scene, camera)
    t0 = time.perf_counter()
    for k in range(total):
        ts = time.perf_counter()
        step_frame()
        wp.synchronize_device(device)
        tsim = time.perf_counter() - ts
        camera_at(k / FPS, VIDEO)
        renderer.render(scene, camera)
        rgb = caption(renderer.read_pixels(), font)
        enc.send(rgb)
        if k in stills_at:
            Image.fromarray(rgb).save(os.path.join(OUTDIR, f"{tag}_still_{k:04d}.png"))
        if k in sheet_at:
            sheet.append(np.asarray(Image.fromarray(rgb).resize((W // 3, H // 3), Image.LANCZOS)))
        if k % 60 == 0:
            xs = x.numpy()
            bad = int((~np.isfinite(xs)).sum())
            print(f"  frame {k}/{total} sim {tsim * 1e3:.1f} ms  "
                  f"({time.perf_counter() - t0:.0f}s) nan={bad} "
                  f"ymin={np.nanmin(xs[:, 1]):.3f}", flush=True)
    enc.close()
    if len(sheet) == 9:
        rows = [np.concatenate(sheet[r * 3:(r + 1) * 3], 1) for r in range(3)]
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(OUTDIR, f"{tag}_sheet.png"))
    print(f"wrote {OUTDIR}/{tag}.mp4 in {time.perf_counter() - t0:.0f}s")
else:
    camera_at(0.0, 1.0)
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(0.0, 0.25, 0.0)
    canvas.on_window_resize(resize_handler(camera, renderer))

    def animate():
        step_frame()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)

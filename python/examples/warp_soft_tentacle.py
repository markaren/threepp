"""Soft robotic tentacles that wrap a ball each and lift it: Warp + threepp.

The tentacle is a tapered silicone continuum body, 1 m long, meshed as a stack
of rings (a centre particle plus a ring of surface particles each) split into
tetrahedra. Every tet keeps its volume and every lattice edge its length, both
as XPBD constraints with a compliance taken from the local cross-section, so
the thick base is stiff and the thin tip is floppy -- which is what makes it
curl like an octopus arm instead of bending like a hose.

It is driven like a muscular hydrostat (or a fluidic soft actuator): four
longitudinal bands (+x, +z, -x, -z) run along the skin, a long pair routed out
to the skin over the distal half and a short pair over the proximal half.
Contracting a band shortens the axial rest length of the lattice and tets on
that side; the body then bends, sags and wobbles its own way there. Because
the tip is thin and floppy and the base thick and stiff, one contraction
command coils the tip into a spiral while the base only leans. Nothing is
keyframed -- the only inputs are eight scalar contraction commands.

Each ball is a rigid sphere with contact push-out and friction against every
skin particle; it rides up only because the coil squeezes it. Three arms by
default (--arms N), started a little apart; every arm is the same tables tiled
with a vertex offset, so one launch per colour still covers them all.

Everything runs in raw Warp kernels (graph-coloured Gauss-Seidel XPBD, one CUDA
graph per frame); threepp's Vulkan path tracer draws it.

    python warp_soft_tentacle.py                 # window, drag to orbit
    python warp_soft_tentacle.py --video 12.2    # headless film -> D:/dev/_softbody_films/soft_tentacle
    python warp_soft_tentacle.py --video 12.2 --size 960x540   # quick preview
    python warp_soft_tentacle.py --probe 10 --arms 1           # sim only, prints
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (Encoder, accum_normals, cli_arg, load_font, parse_size,
                         resize_handler, scatter_soup, standard_material)

VIDEO = cli_arg("--video", 0.0, float)
PROBE = cli_arg("--probe", 0.0, float)      # sim only, print tip/ball, no render
OUTDIR = cli_arg("--outdir", r"D:\dev\_softbody_films\soft_tentacle", str)
W, H = parse_size(cli_arg("--size", "1920x1080", str))
NOBALL = "--noball" in sys.argv

# --- body -------------------------------------------------------------------------
LENGTH = 1.0
N_RINGS = 64                 # rings along the arm
K = 16                       # surface particles per ring
R_BASE = cli_arg("--rbase", 0.050, float)
R_TIP = 0.008
BASE_Y = 0.10                # base ring height (inside the mount)
DENSITY = 1100.0             # silicone
E_MOD = cli_arg("--E", 5.0e6, float)       # Young's modulus, Pa
VOL_COMPLIANCE_SCALE = 0.2   # tets resist volume change harder than shape change
SUBSTEPS = cli_arg("--substeps", 40, int)
DT = 1.0 / (60.0 * SUBSTEPS)
DAMPING = cli_arg("--damp", 0.0012, float)
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
SKIN_R = 0.004               # contact radius of a skin particle

# --- ball and tee -------------------------------------------------------------------
BALL_R = cli_arg("--ballr", 0.080, float)
BALL_MASS = cli_arg("--ballm", 0.12, float)
TEE_X = cli_arg("--teex", 0.37, float)
TEE_Z = cli_arg("--teez", -0.015, float)
TEE_TOP = cli_arg("--teey", 0.225, float)
TEE_R = 0.012
MU = cli_arg("--mu", 1.0, float)

FILM_T = VIDEO if VIDEO > 0 else 11.0


def smooth(a, b, t):
    s = min(max((t - a) / (b - a), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


# Muscle programme: axial contraction of each band (fraction of rest length).
# Band order: long +x, +z, -x, -z; short +x, +z, -x, -z.
SWAY_AMP = cli_arg("--sway", 0.012, float)
T_SWAY_END = cli_arg("--tsway", 3.0, float)
T_REACH = cli_arg("--treach", 3.2, float)
T_WRAP = cli_arg("--twrap", 4.4, float)
T_LIFT = cli_arg("--tlift", 6.6, float)
C_REACH = cli_arg("--creach", 0.10, float)   # short +x: lean over the ball
C_WRAP = cli_arg("--cwrap", 0.25, float)     # long +x: coil
C_LIFT = cli_arg("--clift", 0.15, float)
C_SQUEEZE = cli_arg("--csqueeze", 0.07, float)     # short -x: stand back up


def cable_program(t):
    c = np.zeros(8, np.float32)
    # Sway: a travelling circular wave on the long cables, dying out as it reaches.
    sway = 1.0 - smooth(T_SWAY_END - 0.6, T_REACH + 0.4, t)
    amp = SWAY_AMP * sway * smooth(0.0, 0.8, t)
    ph = 2.0 * math.pi * 0.55 * t
    c[0] += amp * max(math.cos(ph), 0.0)
    c[2] += amp * max(-math.cos(ph), 0.0)
    c[1] += amp * max(math.sin(ph), 0.0)
    c[3] += amp * max(-math.sin(ph), 0.0)
    # a slower proximal sway on the short cables, out of phase
    ph2 = ph * 0.7 + 1.3
    c[4] += 0.6 * amp * max(math.cos(ph2), 0.0)
    c[6] += 0.6 * amp * max(-math.cos(ph2), 0.0)
    c[5] += 0.6 * amp * max(math.sin(ph2), 0.0)
    c[7] += 0.6 * amp * max(-math.sin(ph2), 0.0)
    # Reach: lean the proximal half over the ball.
    lean = smooth(T_REACH, T_REACH + 1.4, t) * (1.0 - 0.9 * smooth(T_LIFT, T_LIFT + 2.0, t))
    c[4] += C_REACH * lean
    # Wrap: the long +x band contracts; the tip coils round the ball.
    c[0] += C_WRAP * smooth(T_WRAP, T_WRAP + 1.8, t)
    c[0] += C_SQUEEZE * smooth(T_LIFT - 0.6, T_LIFT + 0.4, t)      # squeeze before lifting
    # Lift: the short -x band stands the base back up, carrying the coil.
    c[6] += C_LIFT * smooth(T_LIFT, T_LIFT + 2.2, t)
    # Hold: a gentle proximal sway with the ball in the coil.
    hold = smooth(T_LIFT + 2.4, T_LIFT + 3.2, t)
    ph3 = 2.0 * math.pi * 0.35 * (t - T_LIFT)
    c[5] += 0.03 * hold * max(math.sin(ph3), 0.0)
    c[7] += 0.03 * hold * max(-math.sin(ph3), 0.0)
    return c


# --- mesh construction (numpy, once) ------------------------------------------------

def ring_radius(s):
    return R_TIP + (R_BASE - R_TIP) * (1.0 - s) ** 1.15


ds = LENGTH / (N_RINGS - 1)
stride = K + 1
n_verts = N_RINGS * stride + 1
TIP = N_RINGS * stride
p0 = np.zeros((n_verts, 3), np.float32)
radius_of = np.zeros(n_verts, np.float32)
ang = 2.0 * math.pi * np.arange(K) / K
for r in range(N_RINGS):
    s = r / (N_RINGS - 1)
    rr = ring_radius(s)
    y = BASE_Y + s * LENGTH
    c = r * stride
    p0[c] = (0.0, y, 0.0)
    radius_of[c] = rr
    # alternate rings twist half a cell: triangles instead of quads, rounder look
    a = ang
    p0[c + 1:c + 1 + K, 0] = rr * np.cos(a)
    p0[c + 1:c + 1 + K, 1] = y
    p0[c + 1:c + 1 + K, 2] = rr * np.sin(a)
    radius_of[c + 1:c + 1 + K] = rr
p0[TIP] = (0.0, BASE_Y + LENGTH + R_TIP * 0.9, 0.0)
radius_of[TIP] = R_TIP


def cidx(r):
    return r * stride


def sidx(r, k):
    return r * stride + 1 + (k % K)


# Mass from the local cross-section: m = rho * pi r^2 * ds shared over the ring.
mass = np.zeros(n_verts, np.float64)
for r in range(N_RINGS):
    rr = ring_radius(r / (N_RINGS - 1))
    m_ring = DENSITY * math.pi * rr * rr * ds
    mass[cidx(r)] = 0.25 * m_ring
    for k in range(K):
        mass[sidx(r, k)] = 0.75 * m_ring / K
mass[TIP] = mass[sidx(N_RINGS - 1, 0)]
inv_mass = (1.0 / mass).astype(np.float32)
PINNED_RINGS = 2
for r in range(PINNED_RINGS):
    inv_mass[cidx(r):cidx(r) + stride] = 0.0
total_mass = mass.sum()

# Distance constraints: ring edges, spokes, axial (surface + centre), diagonals.
dist = []


def add_d(i, j):
    dist.append((i, j))


for r in range(N_RINGS):
    for k in range(K):
        add_d(sidx(r, k), sidx(r, k + 1))
        add_d(cidx(r), sidx(r, k))
        add_d(sidx(r, k), sidx(r, k + K // 2)) if k < K // 2 else None
for r in range(N_RINGS - 1):
    add_d(cidx(r), cidx(r + 1))
    for k in range(K):
        add_d(sidx(r, k), sidx(r + 1, k))
        add_d(sidx(r, k), sidx(r + 1, k + 1))
        add_d(sidx(r, k + 1), sidx(r + 1, k))
        add_d(cidx(r), sidx(r + 1, k))
        add_d(sidx(r, k), cidx(r + 1))
last = N_RINGS - 1
for k in range(K):
    add_d(sidx(last, k), TIP)
add_d(cidx(last), TIP)

# Tets: each wedge (centre, k, k+1) x (ring r, ring r+1) -> 3 tets.
tets = []
for r in range(N_RINGS - 1):
    for k in range(K):
        a0, b0, c0 = cidx(r), sidx(r, k), sidx(r, k + 1)
        a1, b1, c1 = cidx(r + 1), sidx(r + 1, k), sidx(r + 1, k + 1)
        tets += [(a0, b0, c0, a1), (b0, c0, a1, b1), (c0, a1, b1, c1)]
for k in range(K):
    tets.append((cidx(last), sidx(last, k), sidx(last, k + 1), TIP))
tets = np.array(tets, np.int32)


def tet_vol(t, p):
    a, b, c, d = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], p[t[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


v0 = tet_vol(tets, p0)
flip = v0 < 0
tets[flip] = tets[flip][:, [1, 0, 2, 3]]
v0 = tet_vol(tets, p0).astype(np.float32)


def color_constraints(items):
    """Greedy colouring so no two constraints in a colour share a particle."""
    used = [0] * n_verts
    col = np.empty(len(items), np.int32)
    for n, it in enumerate(items):
        m = 0
        for v in it:
            m |= used[v]
        c = 0
        while (m >> c) & 1:
            c += 1
        col[n] = c
        for v in it:
            used[v] |= 1 << c
    order = np.argsort(col, kind="stable")
    nc = int(col.max()) + 1
    cnt = np.bincount(col, minlength=nc)
    start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    return order, nc, cnt, start


dist = np.array(dist, np.int32)
d_order, d_nc, d_cnt, d_start = color_constraints([tuple(x) for x in dist])
dist = dist[d_order]
d_rest = np.linalg.norm(p0[dist[:, 0]] - p0[dist[:, 1]], axis=1).astype(np.float32)
# Compliance of a lattice edge = l / (E A): A is the ring's section shared over the
# edges that cross it (~ 3K + 2 of them).
r_loc = 0.5 * (radius_of[dist[:, 0]] + radius_of[dist[:, 1]])
A_edge = math.pi * r_loc ** 2 / (3 * K + 2)
d_comp = (d_rest / (E_MOD * A_edge)).astype(np.float32)

t_order, t_nc, t_cnt, t_start = color_constraints([tuple(x) for x in tets])
tets = tets[t_order]
v0 = v0[t_order]
r_tet = radius_of[tets].mean(axis=1)
# Volumetric compliance: 1 / (K_bulk * V), scaled.
t_comp = (VOL_COMPLIANCE_SCALE / (E_MOD * np.maximum(np.abs(v0), 1e-12))).astype(np.float32)

# Tendons: four sides; long to the tip ring, short to mid-length.
SIDE_K = [0, K // 4, K // 2, 3 * K // 4]      # +x, +z, -x, -z (ring 0 orientation)
MID = int(0.45 * N_RINGS)
DIST_S0 = cli_arg("--ds0", 0.35, float)
DIST_S1 = cli_arg("--ds1", 0.65, float)
SWAY_FLOOR = 0.35



# Render surface: two triangles per quad between consecutive rings + a tip fan.
faces = []
for r in range(N_RINGS - 1):
    for k in range(K):
        a, b = sidx(r, k), sidx(r, k + 1)
        c, d = sidx(r + 1, k), sidx(r + 1, k + 1)
        faces += [(a, c, b), (b, c, d)]
for k in range(K):
    faces.append((sidx(last, k), TIP, sidx(last, k + 1)))
faces = np.array(faces, np.int32)
# Outward winding check against the radial direction.
fa = p0[faces]
fn = np.cross(fa[:, 1] - fa[:, 0], fa[:, 2] - fa[:, 0])
radial = fa.mean(axis=1) - np.stack([np.zeros(len(faces)), fa.mean(axis=1)[:, 1],
                                     np.zeros(len(faces))], 1)
if (np.einsum("ij,ij->i", fn, radial) < 0).mean() > 0.5:
    faces = faces[:, ::-1].copy()

# --- warp kernels ---------------------------------------------------------------------


@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
              w: wp.array(dtype=float), dt: float, damping: float, g: wp.vec3):
    i = wp.tid()
    p = x[i]
    v = (p - prev[i]) * (1.0 - damping)
    prev[i] = p
    if w[i] > 0.0:
        x[i] = p + v + g * dt * dt


@wp.kernel
def solve_dist(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float),
               ci: wp.array(dtype=int), cj: wp.array(dtype=int),
               rest: wp.array(dtype=float), comp: wp.array(dtype=float),
               start: int, inv_dt2: float):
    k = start + wp.tid()
    i = ci[k]
    j = cj[k]
    wi = w[i]
    wj = w[j]
    d = x[i] - x[j]
    l = wp.length(d)
    denom = wi + wj + comp[k] * inv_dt2
    if l < 1.0e-9 or denom < 1.0e-12:
        return
    n = d / l
    dl = -(l - rest[k]) / denom
    x[i] = x[i] + n * (dl * wi)
    x[j] = x[j] - n * (dl * wj)


@wp.kernel
def solve_tet(x: wp.array(dtype=wp.vec3), w: wp.array(dtype=float),
              tet: wp.array(dtype=wp.vec4i), v0: wp.array(dtype=float),
              comp: wp.array(dtype=float), start: int, inv_dt2: float):
    k = start + wp.tid()
    t = tet[k]
    a = x[t[0]]
    b = x[t[1]]
    c = x[t[2]]
    d = x[t[3]]
    vol = wp.dot(wp.cross(b - a, c - a), d - a) / 6.0
    g1 = wp.cross(c - a, d - a) / 6.0
    g2 = wp.cross(d - a, b - a) / 6.0
    g3 = wp.cross(b - a, c - a) / 6.0
    g0 = -(g1 + g2 + g3)
    w0 = w[t[0]]
    w1 = w[t[1]]
    w2 = w[t[2]]
    w3 = w[t[3]]
    denom = (w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1) + w2 * wp.dot(g2, g2)
             + w3 * wp.dot(g3, g3) + comp[k] * inv_dt2)
    if denom < 1.0e-20:
        return
    lam = -(vol - v0[k]) / denom
    x[t[0]] = a + g0 * (lam * w0)
    x[t[1]] = b + g1 * (lam * w1)
    x[t[2]] = c + g2 * (lam * w2)
    x[t[3]] = d + g3 * (lam * w3)


@wp.kernel
def ball_integrate(bp: wp.array(dtype=wp.vec3), bprev: wp.array(dtype=wp.vec3),
                   dt: float, g: wp.vec3, active: int):
    b = wp.tid()
    p = bp[b]
    v = (p - bprev[b]) * 0.9995
    bprev[b] = p
    if active != 0:
        bp[b] = p + v + g * dt * dt


@wp.kernel
def contacts(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
             w: wp.array(dtype=float), rad: wp.array(dtype=float), n_per: int,
             bp: wp.array(dtype=wp.vec3), bprev: wp.array(dtype=wp.vec3),
             ball_r: float, wb: float, mu: float, active: int,
             tees: wp.array(dtype=wp.vec3), tee_r: float,
             bdelta: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    wi = w[i]
    if wi == 0.0:
        return
    body = i / n_per
    p = x[i]
    r = rad[i]
    # floor
    if p[1] < r:
        p = wp.vec3(p[0], r, p[2])
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * 0.6
    # tee post (vertical capsule from floor to tee top)
    tee = tees[body]
    if p[1] < tee[1]:
        dx = p[0] - tee[0]
        dz = p[2] - tee[2]
        dh = wp.sqrt(dx * dx + dz * dz)
        if dh < tee_r + r and dh > 1.0e-6:
            s = (tee_r + r) / dh
            p = wp.vec3(tee[0] + dx * s, p[1], tee[2] + dz * s)
    # ball
    if active != 0:
        c = bp[body]
        d = p - c
        l = wp.length(d)
        rr = ball_r + r
        if l < rr and l > 1.0e-6:
            n = d / l
            pen = rr - l
            wsum = wi + wb
            p = p + n * (pen * wi / wsum)
            dball = -n * (pen * wb / wsum)
            # friction: kill a share of the tangential slip against the ball
            rel = (p - prev[i]) - (c - bprev[body])
            tang = rel - n * wp.dot(rel, n)
            corr = tang * mu
            p = p - corr * (wi / wsum)
            dball = dball + corr * (wb / wsum)
            wp.atomic_add(bdelta, body, dball)
    x[i] = p


@wp.kernel
def ball_finish(bp: wp.array(dtype=wp.vec3), bprev: wp.array(dtype=wp.vec3),
                bdelta: wp.array(dtype=wp.vec3), ball_r: float,
                tees: wp.array(dtype=wp.vec3), active: int):
    b = wp.tid()
    if active == 0:
        return
    tee = tees[b]
    p = bp[b] + bdelta[b]
    # tee cup: the ball sits on the post top while it is over it
    dx = p[0] - tee[0]
    dz = p[2] - tee[2]
    if dx * dx + dz * dz < (0.6 * ball_r) * (0.6 * ball_r):
        if p[1] < tee[1] + ball_r * 0.97:
            p = wp.vec3(p[0], tee[1] + ball_r * 0.97, p[2])
            t = p - bprev[b]
            p = p - wp.vec3(t[0], 0.0, t[2]) * 0.5
    if p[1] < ball_r:
        p = wp.vec3(p[0], ball_r, p[2])
        t = p - bprev[b]
        p = p - wp.vec3(t[0], 0.0, t[2]) * 0.3
    bp[b] = p


# --- device state ----------------------------------------------------------------------

# Several arms: the single-arm tables tiled with a vertex offset per arm. Arms
# share no particle, so every colour stays race-free when all arms' members of
# that colour are launched together -- one launch per colour for all arms.
NB = cli_arg("--arms", 3, int)
RING_R = cli_arg("--ring", 0.62, float)
DELAYS = [0.0, 0.6, 1.2, 0.3, 0.9][:NB]
if NB == 1:
    placements = [(0.0, 0.0, 0.0)]
else:
    # bases on a circle, each arm's +x (its grasp side) pointing outward
    placements = [(RING_R * math.cos(2 * math.pi * b / NB + 0.35),
                   RING_R * math.sin(2 * math.pi * b / NB + 0.35),
                   2 * math.pi * b / NB + 0.35) for b in range(NB)]


def place(p, b):
    """Arm b's copy of local points: rotate about y by psi, then move to the base.
    psi turns local +x onto (cos psi, sin psi) in (x, z)."""
    bx, bz, psi = placements[b]
    c, s_ = math.cos(psi), math.sin(psi)
    q = np.empty_like(p)
    q[:, 0] = c * p[:, 0] - s_ * p[:, 2] + bx
    q[:, 1] = p[:, 1]
    q[:, 2] = s_ * p[:, 0] + c * p[:, 2] + bz
    return q


def tile_coloured(items, cnt, start, off):
    """Colour-major layout over all arms: colour c = arm0's c, arm1's c, ..."""
    out = []
    for c in range(len(cnt)):
        blk = items[start[c]:start[c] + cnt[c]]
        for b in range(NB):
            out.append(blk + b * off)
    return np.concatenate(out)


def tile_values(v, cnt, start):
    out = []
    for c in range(len(cnt)):
        blk = v[start[c]:start[c] + cnt[c]]
        for b in range(NB):
            out.append(blk)
    return np.concatenate(out)


N1 = n_verts
P0 = np.concatenate([place(p0, b) for b in range(NB)]).astype(np.float32)
INV_MASS = np.tile(inv_mass, NB)
D_IJ = tile_coloured(dist, d_cnt, d_start, N1)
D_REST = tile_values(d_rest, d_cnt, d_start)
D_COMP = tile_values(d_comp, d_cnt, d_start)
D_CNT = d_cnt * NB
D_START = np.concatenate(([0], np.cumsum(D_CNT)[:-1]))
T_IDX = tile_coloured(tets, t_cnt, t_start, N1)
T_V0 = tile_values(v0, t_cnt, t_start)
T_COMP = tile_values(t_comp, t_cnt, t_start)
T_CNT = t_cnt * NB
T_START = np.concatenate(([0], np.cumsum(T_CNT)[:-1]))
FACES = np.concatenate([faces + b * N1 for b in range(NB)])

wp.init()
device = wp.get_preferred_device()
print(f"soft tentacle: {NB} arms, {NB * n_verts} particles, {NB * len(tets)} tets "
      f"({t_nc} colours), {NB * len(dist)} edges ({d_nc} colours), "
      f"{total_mass:.2f} kg per arm, {SUBSTEPS} substeps on {device}", flush=True)

rng = np.random.default_rng(3)
p_init = P0 + rng.uniform(-2e-4, 2e-4, P0.shape).astype(np.float32)
p_init[INV_MASS == 0] = P0[INV_MASS == 0]
x = wp.array(p_init, dtype=wp.vec3, device=device)
prev = wp.array(p_init, dtype=wp.vec3, device=device)
w_arr = wp.array(INV_MASS, dtype=float, device=device)
rad_arr = wp.array(np.full(NB * N1, SKIN_R, np.float32), dtype=float, device=device)
d_i = wp.array(D_IJ[:, 0].copy(), dtype=int, device=device)
d_j = wp.array(D_IJ[:, 1].copy(), dtype=int, device=device)
d_r = wp.array(D_REST, dtype=float, device=device)
d_c = wp.array(D_COMP, dtype=float, device=device)
t_idx = wp.array(T_IDX, dtype=wp.vec4i, device=device)
t_v0 = wp.array(T_V0, dtype=float, device=device)
t_c = wp.array(T_COMP, dtype=float, device=device)

# Actuation: four longitudinal muscle bands (+x, +z, -x, -z). Contracting a band
# shortens the axial rest length of the lattice on that side -- the skin there
# WANTS to be shorter, and the body bends, sags and wobbles its way to it.
# Each vertex gets a cosine-lobe weight per side; an edge or tet contracts by
# the mean of its vertices' axial factors.
theta_v = np.arctan2(p0[:, 2], p0[:, 0])
band = np.zeros((n_verts, 4), np.float32)
for side, k in enumerate(SIDE_K):
    th = 2.0 * math.pi * k / K
    band[:, side] = np.clip(np.cos(theta_v - th), 0.0, 1.0) ** 1.5
_rad = np.linalg.norm(p0[:, [0, 2]], axis=1) / np.maximum(radius_of, 1e-9)
band *= np.clip(_rad, 0, 1)[:, None]
s_v = np.clip((p0[:, 1] - BASE_Y) / LENGTH, 0, 1)
# long bands: routed out to the skin distally; short bands: proximal only
long_w = np.clip((s_v - DIST_S0) / (DIST_S1 - DIST_S0), 0, 1)
long_w = np.maximum(long_w * long_w * (3 - 2 * long_w), SWAY_FLOOR).astype(np.float32)
short_w = (1.0 - np.clip((s_v - 0.40) / 0.15, 0, 1)).astype(np.float32)
# rotation about y keeps each edge's vertical and horizontal parts
D_DY = (P0[D_IJ[:, 0], 1] - P0[D_IJ[:, 1], 1]).astype(np.float32)
D_DXZ2 = ((P0[D_IJ[:, 0], 0] - P0[D_IJ[:, 1], 0]) ** 2
          + (P0[D_IJ[:, 0], 2] - P0[D_IJ[:, 1], 2]) ** 2).astype(np.float32)


def contraction_field(c):
    """Per-vertex axial contraction of one arm from its eight band commands."""
    return (band @ c[:4]) * long_w + (band @ c[4:]) * short_w


def actuate(t):
    f = np.concatenate([1.0 - contraction_field(cable_program(t - DELAYS[b]))
                        for b in range(NB)]).astype(np.float32)
    fe = 0.5 * (f[D_IJ[:, 0]] + f[D_IJ[:, 1]])
    d_r.assign(np.sqrt((D_DY * fe) ** 2 + D_DXZ2).astype(np.float32))
    t_v0.assign((T_V0 * f[T_IDX].mean(axis=1)).astype(np.float32))


ball_local = np.array([[TEE_X, TEE_TOP + BALL_R * 0.97, TEE_Z]], np.float32)
tee_local = np.array([[TEE_X, TEE_TOP, TEE_Z]], np.float32)
ball0 = np.concatenate([place(ball_local, b) for b in range(NB)]).astype(np.float32)
tee0 = np.concatenate([place(tee_local, b) for b in range(NB)]).astype(np.float32)
bp = wp.array(ball0, dtype=wp.vec3, device=device)
bprev = wp.array(ball0, dtype=wp.vec3, device=device)
bdelta = wp.zeros(NB, dtype=wp.vec3, device=device)
tees = wp.array(tee0, dtype=wp.vec3, device=device)
BALL_ON = 0 if NOBALL else 1
NV = NB * N1

tris = wp.array(FACES.reshape(-1), dtype=int, device=device)
n_corners = FACES.size
nrm = wp.zeros(NV, dtype=wp.vec3, device=device)
soup_pos = wp.zeros(n_corners, dtype=wp.vec3, device=device)
soup_nrm = wp.zeros(n_corners, dtype=wp.vec3, device=device)
INV_DT2 = 1.0 / (DT * DT)


def frame_launches(nsub=SUBSTEPS):
    for _ in range(nsub):
        wp.launch(integrate, dim=NV, device=device,
                  inputs=[x, prev, w_arr, DT, DAMPING, GRAVITY])
        wp.launch(ball_integrate, dim=NB, device=device,
                  inputs=[bp, bprev, DT, GRAVITY, BALL_ON])
        for c in range(t_nc):
            wp.launch(solve_tet, dim=int(T_CNT[c]), device=device,
                      inputs=[x, w_arr, t_idx, t_v0, t_c, int(T_START[c]), INV_DT2])
        for c in range(d_nc):
            wp.launch(solve_dist, dim=int(D_CNT[c]), device=device,
                      inputs=[x, w_arr, d_i, d_j, d_r, d_c, int(D_START[c]), INV_DT2])
        bdelta.zero_()
        wp.launch(contacts, dim=NV, device=device,
                  inputs=[x, prev, w_arr, rad_arr, N1, bp, bprev, BALL_R, 1.0 / BALL_MASS,
                          MU, BALL_ON, tees, TEE_R, bdelta])
        wp.launch(ball_finish, dim=NB, device=device,
                  inputs=[bp, bprev, bdelta, BALL_R, tees, BALL_ON])
    nrm.zero_()
    wp.launch(accum_normals, dim=len(FACES), device=device, inputs=[x, tris, nrm])
    wp.launch(scatter_soup, dim=n_corners, device=device,
              inputs=[x, nrm, tris, soup_pos, soup_nrm])


# Slow motion is FEWER substeps per rendered frame at the same dt: the physics
# is identical, the film just samples it more densely.
# The speed ramps smoothly, so there is one graph per substep count in between.
NSUBS = tuple(range(SUBSTEPS // 2, SUBSTEPS + 1))
graphs = {}
if device.is_cuda:
    frame_launches()        # compile kernels outside the capture
    wp.copy(x, wp.array(p_init, dtype=wp.vec3, device=device))
    wp.copy(prev, wp.array(p_init, dtype=wp.vec3, device=device))
    bp.assign(ball0)
    bprev.assign(ball0)
    for ns in NSUBS:
        with wp.ScopedCapture(device) as cap:
            frame_launches(ns)
        graphs[ns] = cap.graph

sim_t = 0.0
SLOMO = "--no-slomo" not in sys.argv


def film_speed(t):
    """Sim seconds per film second: slow down through the wrap."""
    if not SLOMO:
        return 1.0
    # smooth ease 1 -> 0.5 through the wrap, ease back out after the lift starts
    down = smooth(T_WRAP + 0.1, T_WRAP + 0.7, t)
    up = smooth(T_WRAP + 1.9, T_WRAP + 2.6, t)
    return 1.0 - 0.5 * down * (1.0 - up)


def step_frame():
    global sim_t
    sp = film_speed(sim_t) if VIDEO > 0 else 1.0
    nsub = min(max(int(round(SUBSTEPS * sp)), SUBSTEPS // 2), SUBSTEPS)
    sim_t += nsub * DT
    actuate(sim_t)
    if graphs:
        wp.capture_launch(graphs[nsub])
    else:
        frame_launches(nsub)


if PROBE > 0:
    t0 = time.perf_counter()
    for f in range(int(PROBE * 60)):
        step_frame()
        if f % 15 == 0:
            xs = x.numpy()
            b = bp.numpy()[0]
            print(f"t={sim_t:5.2f} tip=({xs[TIP,0]:+.3f},{xs[TIP,1]:+.3f},{xs[TIP,2]:+.3f}) "
                  f"balls_y={np.round(bp.numpy()[:, 1], 3)} "
                  f"ball=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f}) nan={np.isnan(xs).any()} "
                  "",
                  flush=True)
    wp.synchronize_device(device)
    print(f"sim {1e3 * (time.perf_counter() - t0) / (PROBE * 60):.2f} ms/frame")
    os.makedirs(OUTDIR, exist_ok=True)
    np.save(os.path.join(OUTDIR, "probe_last.npy"), x.numpy())
    sys.exit(0)

# --- threepp scene ----------------------------------------------------------------------

headless = VIDEO > 0
canvas = tp.Canvas("threepp x warp - soft tentacle", width=W, height=H,
                   antialiasing=4, vsync=False, headless=headless)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = cli_arg("--exposure", 1.1, float)

scene = tp.Scene()
SUN_DIR = np.array([0.55, 0.78, 0.30])
SUN_DIR /= np.linalg.norm(SUN_DIR)
BACK_Y = cli_arg("--backy", 0.17, float)     # backlight elevation (0.53 = the wreck's)


def studio_env(w=1024, hgt=512):
    """Procedural studio: dark cyclorama, three big softboxes (the jelly-wreck
    studio: a backlight just above frame pools on the glossy floor)."""
    elev = ((np.arange(hgt, dtype=np.float32) + 0.5) / hgt - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    E, A = np.meshgrid(elev, az, indexing="ij")
    d = np.stack([np.cos(E) * np.cos(A), np.sin(E), np.cos(E) * np.sin(A)], -1)
    y = d[..., 1:2]
    col = np.where(y > 0, np.float32([0.05, 0.055, 0.065]) * (1 - y) + np.float32([0.02, 0.022, 0.028]) * y,
                   np.float32([0.045, 0.043, 0.042]))
    col = col.astype(np.float32)

    def box(center, half_w, half_h, inten, tint):
        c = np.asarray(center, np.float32)
        c /= np.linalg.norm(c)
        up = np.float32([0, 1, 0])
        r = np.cross(up, c)
        if np.linalg.norm(r) < 1e-4:
            r = np.float32([1, 0, 0])
        r /= np.linalg.norm(r)
        u = np.cross(c, r)
        dd = d @ c
        a = np.arctan2(d @ r, dd)
        b = np.arctan2(d @ u, dd)
        m = (dd > 0) & (np.abs(a) < half_w) & (np.abs(b) < half_h)
        soft = np.clip((half_w - np.abs(a)) / 0.03, 0, 1) * np.clip((half_h - np.abs(b)) / 0.03, 0, 1)
        return (m * soft)[..., None] * np.float32(tint) * inten

    col += box(SUN_DIR, 0.42, 0.30, 9.0, (1.0, 0.95, 0.88))           # key
    # backlight: the jelly-wreck one, lowered. This camera looks down ~20 deg, so
    # at the wreck's height its mirror image washed the whole stage; this low it
    # pools on the far floor behind the arms and the stage stays dark.
    col += box((-0.379, BACK_Y, -0.758), 0.9, 0.12, 5.0, (0.85, 0.9, 1.0))
    col += box((-0.5, 0.35, 0.8), 0.35, 0.25, 2.2, (1.0, 1.0, 1.0))    # fill
    col += box((0.0, 1.0, 0.0), 0.5, 0.5, 2.5, (1.0, 1.0, 1.0))        # top
    out = np.ones((hgt, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


env = studio_env()
scene.environment = env
scene.background = tp.Color(0.016, 0.017, 0.020)

key = tp.DirectionalLight(0xfff2e0, 2.2)
key.position.set(*(SUN_DIR * 12.0))
key.cast_shadow = True
scene.add(key)

floor_mat = standard_material(tp.Color(0.030, 0.032, 0.036), 0.16, 0.0)
floor = tp.Mesh(tp.PlaneGeometry(600, 600), floor_mat)
floor.rotation.x = -math.pi / 2
floor.receive_shadow = True
scene.add(floor)

# Per arm: a dark mount with a steel collar, and a tee holding a chrome ball.
steel = standard_material(tp.Color(0.55, 0.57, 0.60), 0.28, 1.0)
dark = standard_material(tp.Color(0.05, 0.055, 0.06), 0.4, 0.3)
ball_mat = standard_material(tp.Color(0.92, 0.92, 0.94), 0.06, 1.0)
balls = []
for b_ in range(NB):
    bx, bz, _ = placements[b_]
    mount = tp.Mesh(tp.CylinderGeometry(R_BASE * 1.35, R_BASE * 1.7, 0.16, 64), dark)
    mount.position.set(bx, 0.08, bz)
    mount.cast_shadow = True
    mount.receive_shadow = True
    scene.add(mount)
    collar = tp.Mesh(tp.TorusGeometry(R_BASE * 1.25, 0.012, 16, 64), steel)
    collar.rotation.x = math.pi / 2
    collar.position.set(bx, 0.162, bz)
    scene.add(collar)
    tx, ty, tz = (float(v) for v in tee0[b_])
    post = tp.Mesh(tp.CylinderGeometry(TEE_R, TEE_R * 1.3, TEE_TOP, 32), steel)
    post.position.set(tx, TEE_TOP / 2, tz)
    post.cast_shadow = True
    scene.add(post)
    cup = tp.Mesh(tp.CylinderGeometry(0.035, TEE_R, 0.025, 48, 1, True), steel)
    cup.position.set(tx, TEE_TOP - 0.004, tz)
    scene.add(cup)
    foot = tp.Mesh(tp.CylinderGeometry(0.09, 0.1, 0.015, 64), dark)
    foot.position.set(tx, 0.0075, tz)
    scene.add(foot)
    ball = tp.Mesh(tp.SphereGeometry(BALL_R, 96, 64), ball_mat)
    ball.cast_shadow = True
    ball.position.set(*(float(v) for v in ball0[b_]))
    ball.visible = not NOBALL
    scene.add(ball)
    balls.append(ball)
plate = tp.Mesh(tp.CylinderGeometry(RING_R + 0.28 if NB > 1 else 0.34,
                                    RING_R + 0.30 if NB > 1 else 0.36, 0.02, 128), dark)
plate.position.set(0.0, 0.01, 0.0)
plate.receive_shadow = True
scene.add(plate)

# The tentacles: one triangle-soup mesh per arm (the soup is contiguous per arm).
# --look glass (default): clear jewel-coloured jelly, one colour per arm, the
# jelly-wreck recipe; --look jewel: glossy opaque silicone in the same colours;
# --look salmon: the octopus coral of the first cut.
LOOK = cli_arg("--look", "glass", str)
JEWELS = [(1.00, 0.16, 0.24),   # raspberry
          (1.00, 0.52, 0.10),   # amber
          (0.22, 0.45, 1.00),   # sapphire
          (0.42, 0.92, 0.14),   # lime
          (0.74, 0.22, 1.00)]   # grape
s_along = np.clip((p0[:, 1] - BASE_Y) / LENGTH, 0, 1)
side = np.clip(p0[:, 0] / np.maximum(radius_of, 1e-6), -1, 1)     # +x = sucker side
mix = (0.5 + 0.5 * side)[:, None] ** 1.6
q = p0 * np.float32([90.0, 55.0, 90.0])
mott = (np.sin(q[:, 0] + 1.7 * np.sin(q[:, 1] * 0.7)) * np.sin(q[:, 1] + 2.1 * np.cos(q[:, 2]))
        * np.sin(q[:, 2] * 1.3 + q[:, 1] * 0.4))


def arm_colours(b_):
    """Per-vertex colour of one arm: darker back, lighter sucker side, paler tip."""
    if LOOK == "salmon":
        dorsal = np.float32([0.48, 0.055, 0.045])
        ventral = np.float32([0.95, 0.46, 0.36])
        tip_gain = 0.35
    else:
        j = np.float32(JEWELS[b_ % len(JEWELS)])
        dorsal = j * 0.62
        ventral = np.clip(j * 0.92 + 0.10, 0, 1)
        tip_gain = 0.18
    vc = dorsal[None] * (1 - mix) + ventral[None] * mix
    vc = vc * (1.0 + tip_gain * s_along[:, None] ** 1.5)
    vc = vc * (1.0 - 0.22 * np.clip(mott, 0, 1)[:, None] * (1 - mix))
    return np.clip(vc, 0, 1).astype(np.float32)


CPA = len(faces) * 3                  # soup corners per arm
arm_geoms = []
for b_ in range(NB):
    g_ = tp.BufferGeometry()
    fb = faces.reshape(-1) + b_ * N1
    g_.set_attribute("position", P0[fb])
    g_.set_attribute("normal", P0[fb])
    tm = tp.MeshPhysicalMaterial()
    tm.metalness = 0.0
    if LOOK == "glass":
        tm.color = tp.Color(*JEWELS[b_ % len(JEWELS)])
        tm.roughness = 0.06
        tm.transmission = 1.0
        tm.ior = 1.35
        tm.transparent = False
    else:
        g_.set_attribute("color", arm_colours(b_)[faces.reshape(-1)])
        tm.color = tp.Color(1.0, 1.0, 1.0)
        tm.vertex_colors = True
        tm.roughness = 0.42 if LOOK == "salmon" else 0.30
        tm.clearcoat = 1.0
        tm.clearcoat_roughness = 0.10 if LOOK == "salmon" else 0.05
    arm = tp.Mesh(g_, tm)
    arm.cast_shadow = True
    arm.receive_shadow = True
    arm.frustum_culled = False
    scene.add(arm)
    arm_geoms.append(g_)

sucker_rows = []
for r in range(8, N_RINGS - 3):
    if r < N_RINGS - 14:
        sucker_rows.append((r, 1 if r % 2 else K - 1))
    else:
        sucker_rows.append((r, 0))
sucker_rows = sucker_rows[::2] if len(sucker_rows) > 40 else sucker_rows
n_suck = len(sucker_rows) * NB
suck_v = np.concatenate([np.array([sidx(r, k) for r, k in sucker_rows], np.int32) + b_ * N1
                         for b_ in range(NB)])
suck_c = np.concatenate([np.array([cidx(r) for r, _ in sucker_rows], np.int32) + b_ * N1
                         for b_ in range(NB)])
suck_s = np.tile(np.array([0.34 * ring_radius(r / (N_RINGS - 1)) for r, _ in sucker_rows],
                          np.float32), NB)
sucker_mat = tp.MeshPhysicalMaterial()
sucker_mat.color = tp.Color(1.0, 0.78, 0.68) if LOOK == "salmon" else tp.Color(1.0, 1.0, 1.0)
sucker_mat.roughness = 0.30
sucker_mat.clearcoat = 1.0
sucker_mat.clearcoat_roughness = 0.1
suckers = tp.InstancedMesh(tp.SphereGeometry(1.0, 20, 14), sucker_mat, n_suck)
suckers.frustum_culled = False
suckers.cast_shadow = True
scene.add(suckers)
if LOOK != "salmon":
    # suckers: a pale tint of their arm's jewel so the rows read against the body
    per_arm = len(sucker_rows)
    for i in range(n_suck):
        j = np.float32(JEWELS[(i // per_arm) % len(JEWELS)])
        c = 0.45 * j + 0.55 * np.float32([1.0, 0.94, 0.90])
        suckers.set_color_at(i, tp.Color(*[float(v) for v in c]))
    suckers.instance_color_needs_update()
_m = tp.Matrix4()
_q = tp.Quaternion()
_pv = tp.Vector3()
_sv = tp.Vector3()


def push_suckers(xs):
    pv = xs[suck_v]
    nrm_s = pv - xs[suck_c]
    nrm_s /= np.maximum(np.linalg.norm(nrm_s, axis=1, keepdims=True), 1e-9)
    cen = pv - nrm_s * (0.12 * suck_s[:, None])
    # quaternion taking +y onto the skin normal: a flattened sphere = a sucker disc
    qx, qy, qz, qw = nrm_s[:, 2], np.zeros(n_suck), -nrm_s[:, 0], 1.0 + nrm_s[:, 1]
    qn = np.maximum(np.sqrt(qx * qx + qz * qz + qw * qw), 1e-9)
    for i in range(n_suck):
        sc = float(suck_s[i])
        _q.set(float(qx[i] / qn[i]), 0.0, float(qz[i] / qn[i]), float(qw[i] / qn[i]))
        _pv.set(float(cen[i, 0]), float(cen[i, 1]), float(cen[i, 2]))
        _sv.set(sc * 1.15, sc * 0.38, sc * 1.15)
        _m.compose(_pv, _q, _sv)
        suckers.set_matrix_at(i, _m)
    suckers.instance_matrix_needs_update()


camera = tp.PerspectiveCamera(34, W / H, 0.05, 200)
TARGET = np.array([0.16, 0.62, 0.0])


def camera_at(t):
    """Slow orbit plus a dolly-in, looking down enough that no horizon shows."""
    u = min(t / FILM_T, 1.0)
    e = smooth(0.0, 1.0, u)
    if NB == 1:
        a = math.radians(-48 + 58 * e)
        r = 2.35 - 0.95 * e
        tgt = np.array([0.16 + 0.04 * e, 0.58 - 0.12 * e, 0.0])
        pitch = 20.0 - 4.0 * e
    else:
        a = math.radians(-35 + 60 * e)
        r = 3.35 - 0.85 * e
        tgt = np.array([0.08, 0.50 - 0.04 * e, 0.05])
        pitch = 23.0 - 3.0 * e
    yh = tgt[1] + r * math.tan(math.radians(pitch))
    camera.position.set(tgt[0] + r * math.sin(a), yh, tgt[2] + r * math.cos(a))
    camera.look_at(float(tgt[0]), float(tgt[1]), float(tgt[2]))


camera_at(0.0)


def push_geometry():
    sp_, sn_ = soup_pos.numpy(), soup_nrm.numpy()
    for b_, g_ in enumerate(arm_geoms):
        g_.update_attribute("position", sp_[b_ * CPA:(b_ + 1) * CPA])
        g_.update_attribute("normal", sn_[b_ * CPA:(b_ + 1) * CPA])
    bb = bp.numpy()
    for b_, ball in enumerate(balls):
        ball.position.set(float(bb[b_, 0]), float(bb[b_, 1]), float(bb[b_, 2]))
    push_suckers(x.numpy())


def caption(img, font):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    txt = "threepp + NVIDIA Warp  \u00b7  soft robotic tentacles, GPU XPBD  \u00b7  " \
          f"{NB} arms, {NV} particles, {NB * len(tets)} tets"
    x0, y0 = int(0.025 * W), int(H - 0.055 * H)
    dr.text((x0 + 1, y0 + 1), txt, font=font, fill=(0, 0, 0))
    dr.text((x0, y0), txt, font=font, fill=(205, 212, 218))
    return np.asarray(im)


if VIDEO > 0:
    os.makedirs(OUTDIR, exist_ok=True)
    from PIL import Image
    font = load_font(max(12, int(H * 0.018)))
    total = int(round(VIDEO * 60))
    tag = f"{W}x{H}_{LOOK}"
    STILL_AT = cli_arg("--still-at", 0.0, float)     # look check: one still, no film
    if STILL_AT > 0:
        ks = int(total * STILL_AT)
        for k in range(ks):
            step_frame()
        push_geometry()
        camera_at(ks / 60.0)
        for _ in range(40):
            renderer.render(scene, camera)
        img = caption(np.asarray(renderer.read_pixels()), font)
        fn = os.path.join(OUTDIR, f"look_{tag}_{ks:04d}.png")
        Image.fromarray(img).save(fn)
        print(f"wrote {fn}")
    mp4 = os.path.join(OUTDIR, f"soft_tentacle_{tag}.mp4")
    enc = Encoder(mp4, W, H, 60, crf=17, preset="medium") if STILL_AT <= 0 else None
    stills_at = [int(total * f) for f in (0.12, 0.45, 0.62, 0.9)]
    sheet_at = [int(total * (i + 0.5) / 9) for i in range(9)]
    sheet = []
    push_geometry()
    for _ in range(30):
        renderer.render(scene, camera)
    t0 = time.perf_counter()
    for k in range(total if enc else 0):
        step_frame()
        push_geometry()
        camera_at(k / 60.0)
        renderer.render(scene, camera)
        img = caption(np.asarray(renderer.read_pixels()), font)
        enc.send(img)
        if k in stills_at:
            Image.fromarray(img).save(os.path.join(OUTDIR, f"still_{tag}_{k:04d}.png"))
        if k in sheet_at:
            sheet.append(img[::2, ::2].copy())
        if k % 60 == 0:
            b = bp.numpy()[0]
            print(f"  frame {k}/{total} ({time.perf_counter() - t0:.0f}s) "
                  f"ball y={b[1]:.3f}", flush=True)
    if enc:
        enc.close()
        print(f"wrote {mp4} in {time.perf_counter() - t0:.0f}s")
    if len(sheet) == 9:
        rows = [np.concatenate(sheet[i * 3:(i + 1) * 3], 1) for i in range(3)]
        Image.fromarray(np.concatenate(rows, 0)).save(
            os.path.join(OUTDIR, f"contact_sheet_{tag}.png"))
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(*TARGET)
    canvas.on_window_resize(resize_handler(camera, renderer))

    def animate():
        step_frame()
        push_geometry()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)

"""A water balloon and a needle -- NVIDIA Warp simulation, threepp rendering.

The balloon is two coupled Warp simulations. The water is Position Based
Fluids (particles at D spacing, surfaced by GPU marching cubes and drawn as a
refracting water mesh). The rubber is an XPBD shell -- an icosphere
pre-stretched over the water, held by edge + weak bending constraints -- and
it is the WATER that keeps it inflated: there is no volume constraint. The two
meet through a thin-shell contact on a wp.Mesh of the live rubber: every fluid
particle is ray-cast from where it started the substep to where it wants to go
(no tunnelling through the thin film), then pushed R_SHELL off the closest
face, and the rubber takes its share of that push back through the face's
barycentric weights. The fluid density solve resists compression, the rubber
tries to shrink, and the balloon sits at the equilibrium in between, sagging
on the table like the real thing.

A needle comes in from the right. Faces it touches die, and from the hole the
tear runs through any neighbouring face that is still taut (mean edge stretch
above TEAR_RATIO) -- through a water-filled balloon that is every face, so the
rubber peels away around the whole ball, and for a moment the water is still
standing there in the balloon's shape. Then gravity.

The shot is filmed the way the real thing is filmed: the playback ramps into
slow motion (SLOWMO x) as the needle arrives, holds through the tear, and
ramps back out. The physics timestep is fixed throughout -- only how much sim
time a rendered frame covers changes.

    pip install warp-lang
    python warp_water_balloon.py                 # window; drag to orbit, Esc quits
    python warp_water_balloon.py --video 9       # 9 s of playback -> warp_water_balloon.mp4
    python warp_water_balloon.py --shot 1.0      # headless: sim to t=1.0 s, write a PNG
    python warp_water_balloon.py --slowmo 0.04   # slower slow motion
    python warp_water_balloon.py --max-substeps 2  # window: lighter sim, slower playback
    python warp_water_balloon.py --frames 300    # window: 300 frames, timing, then exit
    python warp_water_balloon.py --no-graph      # do not capture the substep CUDA graph

Vulkan only (the water is the ray-traced glass path). Needs a CUDA device.
"""
import math
import os
import sys
import tempfile
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (DensitySurface, Encoder, cli_arg, csr_from_pairs, edge_adjacency,
                         find_ffmpeg,
                         icosphere, load_font, parse_size, pbf_constants, resize_handler,
                         shell_pairs, standard_material)

SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 1.35, float)          # SIM seconds
VIDEO = cli_arg("--video", 0.0, float)              # PLAYBACK seconds
SLOWMO = cli_arg("--slowmo", 0.08, float)           # sim seconds per playback second
FRAMES = cli_arg("--frames", 0, int)               # window: run N frames, print timing, exit
HEADLESS = SHOT or VIDEO > 0.0
W, H = parse_size(cli_arg("--size", "1280x720", str))
FPS = 60

if HEADLESS:
    # Presenting to the hidden window is pure cost offline.
    os.environ.setdefault("THREEPP_VULKAN_SUPPRESS_PRESENT", "1")

wp.set_module_options({"fast_math": True})

# --- the balloon ------------------------------------------------------------

D = 0.003                    # fluid rest spacing (m): the resolution knob
H_SPH = 2.0 * D
MASS = 1.0
R_WATER = 0.065              # radius of the water sphere as filled
PRESTRETCH = 1.25            # rubber rest radius = (filled radius) / PRESTRETCH
SUBDIV = 5                   # icosphere subdivisions
R_SHELL = 1.35 * D           # fluid keeps this far off the rubber. Wider than the
                             # physical film on purpose: the marching-cubes skin
                             # sits a fraction of a cell outside the particle
                             # centres, and it must stay INSIDE the rubber until
                             # the rubber is gone
R_MEM0 = R_WATER + R_SHELL + 0.0005
R_REST = R_MEM0 / PRESTRETCH
TABLE_Y = 0.0
SPAWN_Y = R_MEM0 + 0.03      # centre height at t=0: a short drop, a wobble
WALL_EPS = 0.6 * D

MAX_SUB_WINDOW = cli_arg("--max-substeps", 4, int)
                             # Window only. The physics step below is fixed, so
                             # capping substeps per rendered frame does not
                             # coarsen anything -- it advances LESS sim time per
                             # frame, i.e. the playback slows down. That is the
                             # honest trade for an interactive window, where the
                             # CUDA sim and the Vulkan frame share the GPU. The
                             # film (--video/--shot) is never capped.
DT_SUB = 1.0 / 720.0         # physics substep. The fluid<->shell contact needs it
                             # this fine EVERYWHERE: coarser and the pressure solve
                             # under-converges (the resting balloon sags) or the
                             # water leaks out through the shell before the
                             # needle ever arrives.
ITERS = 3                    # coupled (fluid density + contact + rubber) iterations
MEM_ITERS = 2                # rubber Jacobi passes per coupled iteration
COUPLE_SHARE = 0.6           # fraction of a contact push the rubber takes back
V_MAX = 6.0
FLOOR_DRAG = 0.96            # per-substep tangential damping of the layer on the table
MAX_DP = 0.35 * D
JACOBI_RELAX = 0.4
XSPH_C = 0.05
GRAVITY = -9.81

STIFF_EDGE = 1.0
STIFF_BEND = 0.15
MEM_RELAX = 0.35
MEM_DAMPING = 0.02
MU_TABLE = 0.4

TEAR_RATIO = 1.04            # a face next to the hole tears while its edges are
                             # stretched past this; truly slack rubber stops the tear
TEAR_RINGS = 3               # face rings the tear may advance per substep
NEEDLE_R = 0.0015
NEEDLE_LEN = 0.085
NEEDLE_SPEED = 0.25          # m/s, real time
NEEDLE_T0 = 0.55             # sim s: the needle starts moving, once the balloon has settled
NEEDLE_X_START = R_MEM0 + 0.06
NEEDLE_X_STOP = 0.0
NEEDLE_Y = 0.62 * R_WATER    # aims below the resting balloon's centre, at its belly
PUNCT_EPS = 0.0008
SLOWMO_TRIGGER_X = R_MEM0 + 0.05    # tip x at which the playback starts slowing
SLOWMO_RAMP_IN = 0.2         # playback s
SLOWMO_HOLD = 0.30           # SIM s after the first face dies
SLOWMO_RAMP_OUT = 1.2        # playback s

# fluid domain (analytic clamp) and the surfacing grid around it
DOM = 0.40
GX0, GY0, GZ0 = -DOM, TABLE_Y - 0.015, -DOM
CELL = 1.2 * D
NGX = int(2.0 * DOM / CELL) + 1
NGY = int((0.30 - GY0) / CELL) + 1
NGZ = NGX
ISO = 0.4 * (CELL / D) ** 3  # a touch under half density: thin puddles survive surfacing
# Capacity, not the live count, is what the renderer pays per frame for a
# dynamic mesh here (vertex upload + BLAS work), so size it just above the real
# high-water mark; the note below shouts if it is ever hit.
MAX_TRIS = cli_arg("--max-tris", 60_000, int)

S_CORR_N = 4.0
S_CORR_DQ = 0.20 * H_SPH
PBF = pbf_constants(D, H_SPH, MASS, S_CORR_DQ, S_CORR_N)
POLY6, SPIKY = PBF["poly6"], PBF["spiky"]
RHO0, W_DQ = PBF["rho0"], PBF["w_dq"]
EPS_CFM, S_CORR_K = PBF["eps_cfm"], PBF["s_corr_k"]


# --- warp: fluid ------------------------------------------------------------

@wp.func
def w_poly6(r2: float) -> float:
    d = H_SPH * H_SPH - r2
    if d <= 0.0:
        return 0.0
    return POLY6 * d * d * d


@wp.func
def w_spiky_grad(rv: wp.vec3, r: float) -> wp.vec3:
    if r <= 1.0e-9 or r >= H_SPH:
        return wp.vec3(0.0, 0.0, 0.0)
    return rv * (SPIKY * (H_SPH - r) * (H_SPH - r) / r)


@wp.func
def seg_closest(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> wp.vec3:
    ab = b - a
    t = wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-12)
    t = wp.min(wp.max(t, 0.0), 1.0)
    return a + ab * t


@wp.func
def collide_fluid(p: wp.vec3, na: wp.vec3, nb: wp.vec3) -> wp.vec3:
    x = wp.min(wp.max(p[0], GX0 + 0.02), -GX0 - 0.02)
    z = wp.min(wp.max(p[2], GZ0 + 0.02), -GZ0 - 0.02)
    y = wp.max(p[1], TABLE_Y + WALL_EPS)
    q = wp.vec3(x, y, z)
    # the needle: a capsule
    c = seg_closest(q, na, nb)
    d = q - c
    l = wp.length(d)
    r = NEEDLE_R + WALL_EPS
    if l < r:
        if l > 1.0e-9:
            q = c + d * (r / l)
        else:
            q = c + wp.vec3(0.0, r, 0.0)
    return q


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3),
            v: wp.array(dtype=wp.vec3),
            xp: wp.array(dtype=wp.vec3),
            dtv: wp.array(dtype=float), needle: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dt = dtv[0]
    na = needle[0]
    nb = needle[1]
    vi = v[i] + wp.vec3(0.0, GRAVITY, 0.0) * dt
    sp = wp.length(vi)
    if sp > V_MAX:
        vi = vi * (V_MAX / sp)
    v[i] = vi
    xp[i] = collide_fluid(x[i] + vi * dt, na, nb)


@wp.kernel
def solve_lambda(xp: wp.array(dtype=wp.vec3),
                 grid: wp.uint64,
                 rho0: float,
                 eps_cfm: float,
                 lam: wp.array(dtype=float)):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = xp[i]
    rho = float(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)
    sum_grad2 = float(0.0)
    q = wp.hash_grid_query(grid, p, H_SPH)
    for j in q:
        rv = p - xp[j]
        r2 = wp.dot(rv, rv)
        if r2 < H_SPH * H_SPH:
            rho += MASS * w_poly6(r2)
            g = w_spiky_grad(rv, wp.sqrt(r2)) * (MASS / rho0)
            grad_i += g
            sum_grad2 += wp.dot(g, g)
    sum_grad2 += wp.dot(grad_i, grad_i)
    # resist compression only; a free-surface particle is under-dense by definition
    c = wp.max(rho / rho0 - 1.0, 0.0)
    lam[i] = -c / (sum_grad2 + eps_cfm)


@wp.kernel
def solve_delta(xp: wp.array(dtype=wp.vec3),
                lam: wp.array(dtype=float),
                grid: wp.uint64,
                rho0: float, w_dq: float, k_corr: float,
                needle: wp.array(dtype=wp.vec3),
                out: wp.array(dtype=wp.vec3)):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    na = needle[0]
    nb = needle[1]
    p = xp[i]
    li = lam[i]
    w_dq_inv = 1.0 / w_dq
    dp = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, p, H_SPH)
    for j in q:
        rv = p - xp[j]
        r2 = wp.dot(rv, rv)
        if r2 < H_SPH * H_SPH and r2 > 1.0e-12:
            qq = w_poly6(r2) * w_dq_inv
            q2 = qq * qq
            s_corr = -k_corr * (q2 * q2)
            dp += w_spiky_grad(rv, wp.sqrt(r2)) * (li + lam[j] + s_corr)
    d = dp * (MASS / rho0 * JACOBI_RELAX)
    dl = wp.length(d)
    if dl > MAX_DP:
        d = d * (MAX_DP / dl)
    out[i] = collide_fluid(p + d, na, nb)


@wp.kernel
def finalize(x: wp.array(dtype=wp.vec3),
             xp: wp.array(dtype=wp.vec3),
             v: wp.array(dtype=wp.vec3),
             dtv: wp.array(dtype=float)):
    i = wp.tid()
    dt = dtv[0]
    p = xp[i]
    vi = (p - x[i]) * (1.0 / dt)
    sp = wp.length(vi)
    if sp > V_MAX:
        vi = vi * (V_MAX / sp)
    # The table is wet and no-slip: drag the layer on it. Without this the
    # water runs out into a one-particle film that marching cubes cannot
    # surface, and the puddle visibly evaporates. A real puddle stays a puddle
    # (surface tension + contact angle); this is the cheap stand-in.
    if p[1] < TABLE_Y + 1.5 * D:
        vi = wp.vec3(vi[0] * FLOOR_DRAG, vi[1], vi[2] * FLOOR_DRAG)
    v[i] = vi
    x[i] = p


@wp.kernel
def xsph(x: wp.array(dtype=wp.vec3),
         v: wp.array(dtype=wp.vec3),
         grid: wp.uint64,
         rho0: float,
         v_out: wp.array(dtype=wp.vec3)):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = x[i]
    vi = v[i]
    dv = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, p, H_SPH)
    for j in q:
        rv = p - x[j]
        r2 = wp.dot(rv, rv)
        if r2 < H_SPH * H_SPH:
            dv += (v[j] - vi) * w_poly6(r2)
    v_out[i] = vi + dv * (XSPH_C * MASS / rho0)


# --- warp: the thin-shell contact (fluid <-> rubber) -------------------------

@wp.kernel
def shell_collide(xp: wp.array(dtype=wp.vec3),
                  x0: wp.array(dtype=wp.vec3),
                  mesh: wp.uint64,
                  midx: wp.array(dtype=wp.int32),
                  mdelta: wp.array(dtype=wp.vec3),
                  mcnt: wp.array(dtype=float),
                  share: float):
    """Keep every particle R_SHELL off the live rubber, on the side it came from.

    Two tests. A ray from the substep's start position to the current iterate
    catches a crossing outright (a particle can move further in one substep
    than the film is thick); then a closest-point query keeps the resting
    contact at R_SHELL. Both are two-sided -- the shreds after the pop have
    water on either side -- and both hand the rubber `share` of the push
    through the face's barycentric weights, averaged per vertex afterwards.
    """
    i = wp.tid()
    p = xp[i]
    s = x0[i]
    d = p - s
    L = wp.length(d)
    if L > 1.0e-7:
        dirn = d * (1.0 / L)
        rq = wp.mesh_query_ray(mesh, s, dirn, L)
        if rq.result:
            n = rq.normal
            if wp.dot(dirn, n) > 0.0:
                n = -n
            hit = s + dirn * rq.t
            pn = hit + n * R_SHELL
            delta = pn - p
            f = rq.face
            w0 = rq.u
            w1 = rq.v
            w2 = 1.0 - rq.u - rq.v
            push = delta * (-share)
            wp.atomic_add(mdelta, midx[f * 3 + 0], push * w0)
            wp.atomic_add(mdelta, midx[f * 3 + 1], push * w1)
            wp.atomic_add(mdelta, midx[f * 3 + 2], push * w2)
            wp.atomic_add(mcnt, midx[f * 3 + 0], w0)
            wp.atomic_add(mcnt, midx[f * 3 + 1], w1)
            wp.atomic_add(mcnt, midx[f * 3 + 2], w2)
            p = pn
    pq = wp.mesh_query_point_no_sign(mesh, p, R_SHELL)
    if pq.result:
        cp = wp.mesh_eval_position(mesh, pq.face, pq.u, pq.v)
        n = wp.mesh_eval_face_normal(mesh, pq.face)
        dv = p - cp
        if wp.dot(dv, n) < 0.0:
            n = -n
        dist = wp.length(dv)
        if dist < R_SHELL:
            pn = cp + n * R_SHELL
            delta = pn - p
            f = pq.face
            w0 = pq.u
            w1 = pq.v
            w2 = 1.0 - pq.u - pq.v
            push = delta * (-share)
            wp.atomic_add(mdelta, midx[f * 3 + 0], push * w0)
            wp.atomic_add(mdelta, midx[f * 3 + 1], push * w1)
            wp.atomic_add(mdelta, midx[f * 3 + 2], push * w2)
            wp.atomic_add(mcnt, midx[f * 3 + 0], w0)
            wp.atomic_add(mcnt, midx[f * 3 + 1], w1)
            wp.atomic_add(mcnt, midx[f * 3 + 2], w2)
            p = pn
    xp[i] = p


@wp.kernel
def mem_apply_push(pos: wp.array(dtype=wp.vec3),
                   mdelta: wp.array(dtype=wp.vec3),
                   mcnt: wp.array(dtype=float)):
    i = wp.tid()
    c = mcnt[i]
    if c > 0.0:
        pos[i] = pos[i] + mdelta[i] * (1.0 / wp.max(c, 1.0))


# --- warp: the rubber -------------------------------------------------------

@wp.kernel
def mem_integrate(x: wp.array(dtype=wp.vec3),
                  prev: wp.array(dtype=wp.vec3),
                  pred: wp.array(dtype=wp.vec3),
                  dtv: wp.array(dtype=float)):
    # dt comes from a device array: the substep is captured into a CUDA graph
    # and the playback may change dt between launches.
    i = wp.tid()
    dt = dtv[0]
    p = x[i]
    v = (p - prev[i]) * (1.0 - MEM_DAMPING)
    prev[i] = p
    pred[i] = p + v + wp.vec3(0.0, GRAVITY, 0.0) * dt * dt


@wp.kernel
def mem_solve(p_in: wp.array(dtype=wp.vec3),
              p_out: wp.array(dtype=wp.vec3),
              offsets: wp.array(dtype=int),
              indices: wp.array(dtype=int),
              rests: wp.array(dtype=float),
              stiffs: wp.array(dtype=float),
              pairid: wp.array(dtype=int),
              pair_alive: wp.array(dtype=int)):
    i = wp.tid()
    p = p_in[i]
    c = wp.vec3(0.0, 0.0, 0.0)
    for k in range(offsets[i], offsets[i + 1]):
        if pair_alive[pairid[k]] != 0:
            d = p_in[indices[k]] - p
            l = wp.length(d)
            if l > 1.0e-9:
                # half the correction: the neighbour computes the other half
                c += d * (0.5 * stiffs[k] * (l - rests[k]) / l)
    p_out[i] = p + c * MEM_RELAX


@wp.kernel
def mem_contacts(pos: wp.array(dtype=wp.vec3),
                 prev: wp.array(dtype=wp.vec3),
                 needle: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    na = needle[0]
    nb = needle[1]
    p = pos[i]
    if p[1] < TABLE_Y:
        p = wp.vec3(p[0], TABLE_Y, p[2])
        # friction: take back a fraction of this substep's tangential motion
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * MU_TABLE
    c = seg_closest(p, na, nb)
    d = p - c
    l = wp.length(d)
    if l < NEEDLE_R:
        if l > 1.0e-9:
            p = c + d * (NEEDLE_R / l)
        else:
            p = c + wp.vec3(0.0, NEEDLE_R, 0.0)
    pos[i] = p


@wp.kernel
def tear(pos: wp.array(dtype=wp.vec3),
         tris: wp.array(dtype=int),
         face_nbrs: wp.array(dtype=int),
         face_rest: wp.array(dtype=float),
         alive_in: wp.array(dtype=int),
         alive_out: wp.array(dtype=int),
         needle: wp.array(dtype=wp.vec3),
         n_dead: wp.array(dtype=int)):
    f = wp.tid()
    if alive_in[f] == 0:
        alive_out[f] = 0
        return
    na = needle[0]
    nb = needle[1]
    a = pos[tris[f * 3 + 0]]
    b = pos[tris[f * 3 + 1]]
    c = pos[tris[f * 3 + 2]]
    dead = int(0)
    cen = (a + b + c) * (1.0 / 3.0)
    if wp.length(cen - seg_closest(cen, na, nb)) < NEEDLE_R + PUNCT_EPS:
        dead = 1        # the needle is parked far away while inactive
    if dead == 0:
        if (alive_in[face_nbrs[f * 3 + 0]] == 0 or alive_in[face_nbrs[f * 3 + 1]] == 0
                or alive_in[face_nbrs[f * 3 + 2]] == 0):
            ratio = (wp.length(a - b) / face_rest[f * 3 + 0]
                     + wp.length(b - c) / face_rest[f * 3 + 1]
                     + wp.length(c - a) / face_rest[f * 3 + 2]) * (1.0 / 3.0)
            if ratio > TEAR_RATIO:
                dead = 1
    if dead != 0:
        wp.atomic_add(n_dead, 0, 1)
    alive_out[f] = 1 - dead


@wp.kernel
def pair_update(pair_faces: wp.array(dtype=int),
                pair_kind: wp.array(dtype=int),
                face_alive: wp.array(dtype=int),
                pair_alive: wp.array(dtype=int)):
    k = wp.tid()
    fa = face_alive[pair_faces[k * 2 + 0]]
    fb = face_alive[pair_faces[k * 2 + 1]]
    alive = int(0)
    if pair_kind[k] == 0:
        if fa != 0 or fb != 0:          # edge: lives while either owner does
            alive = 1
    else:
        if fa != 0 and fb != 0:         # bend: needs both owners
            alive = 1
    pair_alive[k] = alive


@wp.kernel
def mesh_indices_update(tris: wp.array(dtype=int),
                        face_alive: wp.array(dtype=int),
                        sentinel: int,
                        midx: wp.array(dtype=wp.int32)):
    f = wp.tid()
    for c in range(3):
        midx[f * 3 + c] = wp.where(face_alive[f] != 0, wp.int32(tris[f * 3 + c]),
                                   wp.int32(sentinel))


@wp.kernel
def accum_normals(pos: wp.array(dtype=wp.vec3),
                  tris: wp.array(dtype=int),
                  face_alive: wp.array(dtype=int),
                  nrm: wp.array(dtype=wp.vec3)):
    f = wp.tid()
    if face_alive[f] == 0:
        return
    ia, ib, ic = tris[f * 3], tris[f * 3 + 1], tris[f * 3 + 2]
    n = wp.cross(pos[ib] - pos[ia], pos[ic] - pos[ia])
    wp.atomic_add(nrm, ia, n)
    wp.atomic_add(nrm, ib, n)
    wp.atomic_add(nrm, ic, n)


@wp.kernel
def scatter(pos: wp.array(dtype=wp.vec3),
            nrm: wp.array(dtype=wp.vec3),
            tris: wp.array(dtype=int),
            face_alive: wp.array(dtype=int),
            out_pos: wp.array(dtype=wp.vec3),
            out_nrm: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    f = k / 3
    if face_alive[f] == 0:
        out_pos[k] = wp.vec3(0.0, -100.0, 0.0)     # zero area, off screen
        out_nrm[k] = wp.vec3(0.0, 1.0, 0.0)
        return
    i = tris[k]
    out_pos[k] = pos[i]
    n = nrm[i]
    out_nrm[k] = n / wp.max(wp.length(n), 1.0e-9)


# --- mesh construction (numpy, once) ----------------------------------------

verts0, faces = icosphere(SUBDIV)
n_verts = len(verts0)
n_faces = len(faces)
p_rest = verts0 * R_REST
p_mem0 = verts0 * R_MEM0 + np.float32([0.0, SPAWN_Y, 0.0])

# Constraints: mesh edges (stretch) and, per edge, the pair of opposite vertices
# of its two triangles (bending, weak). Every pair remembers the faces that own
# it so the tear can retire it: an edge dies with BOTH its faces, a bend pair
# with EITHER.
pairs = shell_pairs(faces, STIFF_EDGE, STIFF_BEND)       # (i, j, stiff, kind, face_a, face_b)
offsets_np, idx_np, rest_np, stiff_np, pairid_np = csr_from_pairs(
    n_verts, pairs, p_rest, extra_rows=1)                # +1 row for the sentinel

# per face: the three faces across its edges, and the rest length of each edge
edge_faces, _ = edge_adjacency(faces)
face_nbrs_np = np.zeros((n_faces, 3), np.int32)
face_rest_np = np.zeros((n_faces, 3), np.float32)
for fi, (fa, fb, fc) in enumerate(faces):
    for e, (i, j) in enumerate(((fa, fb), (fb, fc), (fc, fa))):
        owners = edge_faces[(min(i, j), max(i, j))]
        face_nbrs_np[fi, e] = owners[1] if owners[0] == fi else owners[0]
        face_rest_np[fi, e] = np.linalg.norm(p_rest[i] - p_rest[j])

# the water: a cubic lattice inside the filled sphere
_o = (np.arange(-int(R_WATER / D) - 1, int(R_WATER / D) + 2) + 0.5) * D
_gx, _gy, _gz = np.meshgrid(_o, _o, _o, indexing="ij")
_lat = np.stack([_gx.ravel(), _gy.ravel(), _gz.ravel()], 1)
_lat = _lat[(_lat ** 2).sum(1) < (R_WATER - 0.5 * D) ** 2]
rng = np.random.default_rng(7)
p_fluid0 = (_lat + np.float32([0.0, SPAWN_Y, 0.0])
            + rng.uniform(-0.05 * D, 0.05 * D, _lat.shape)).astype(np.float32)
N = len(p_fluid0)

wp.init()
device = wp.get_preferred_device()
if not device.is_cuda:
    raise SystemExit(f"warp_water_balloon needs a CUDA device "
                     f"(HashGrid + mesh queries at {1.0 / DT_SUB:.0f} Hz)")
print(f"water balloon: {N:,} fluid particles (D={D*1000:.0f} mm), rubber {n_verts:,} verts / "
      f"{n_faces:,} faces / {len(pairs):,} constraints, grid {NGX}x{NGY}x{NGZ} on {device}")

# fluid state
fx = wp.array(p_fluid0, dtype=wp.vec3, device=device)
fv = wp.zeros(N, dtype=wp.vec3, device=device)
fxp = wp.zeros(N, dtype=wp.vec3, device=device)
fxt = wp.zeros(N, dtype=wp.vec3, device=device)
fx0 = wp.zeros(N, dtype=wp.vec3, device=device)     # substep-start positions (CCD)
lam = wp.zeros(N, dtype=float, device=device)
fvt = wp.zeros(N, dtype=wp.vec3, device=device)
grid = wp.HashGrid(96, 64, 96, device)
grid.reserve(N)

# rubber state (+1 sentinel vertex the dead faces collapse onto)
SENTINEL = n_verts
_pm = np.vstack([p_mem0, np.float32([[0.0, -100.0, 0.0]])])
mx = wp.array(_pm, dtype=wp.vec3, device=device)
mprev = wp.array(_pm, dtype=wp.vec3, device=device)
ma = wp.array(_pm, dtype=wp.vec3, device=device)
mb = wp.array(_pm, dtype=wp.vec3, device=device)
mpts = wp.array(_pm, dtype=wp.vec3, device=device)   # what the contact mesh reads
mnrm = wp.zeros(n_verts + 1, dtype=wp.vec3, device=device)
mdelta = wp.zeros(n_verts + 1, dtype=wp.vec3, device=device)
mcnt = wp.zeros(n_verts + 1, dtype=float, device=device)
m_off = wp.array(offsets_np, dtype=int, device=device)
m_idx = wp.array(idx_np, dtype=int, device=device)
m_rest = wp.array(rest_np, dtype=float, device=device)
m_stiff = wp.array(stiff_np, dtype=float, device=device)
m_pairid = wp.array(pairid_np, dtype=int, device=device)
pair_faces = wp.array(np.array([[p[4], p[5]] for p in pairs], np.int32).ravel(),
                      dtype=int, device=device)
pair_kind = wp.array(np.array([p[3] for p in pairs], np.int32), dtype=int, device=device)
pair_alive = wp.array(np.ones(len(pairs), np.int32), dtype=int, device=device)
tris = wp.array(faces.reshape(-1), dtype=int, device=device)
face_nbrs = wp.array(face_nbrs_np.ravel(), dtype=int, device=device)
face_rest = wp.array(face_rest_np.ravel(), dtype=float, device=device)
face_alive = wp.array(np.ones(n_faces, np.int32), dtype=int, device=device)
face_alive2 = wp.array(np.ones(n_faces, np.int32), dtype=int, device=device)
n_dead = wp.zeros(1, dtype=int, device=device)
needle_dev = wp.array(np.full((2, 3), 9.0, np.float32), dtype=wp.vec3, device=device)
dt_dev = wp.array(np.float32([DT_SUB]), dtype=float, device=device)
midx = wp.array(faces.reshape(-1).astype(np.int32), dtype=wp.int32, device=device)
shell = wp.Mesh(points=mpts, indices=midx)
soup_pos = wp.zeros(n_faces * 3, dtype=wp.vec3, device=device)
soup_nrm = wp.zeros(n_faces * 3, dtype=wp.vec3, device=device)

# surfacing
surface = DensitySurface((GX0, GY0, GZ0), CELL, (NGX, NGY, NGZ), device,
                         blur=(0.25, 0.125))
CAP = MAX_TRIS * 3
stage_pos = wp.zeros(CAP, dtype=wp.vec3, device=device)
stage_nrm = wp.zeros(CAP, dtype=wp.vec3, device=device)
host_pos = wp.zeros(CAP, dtype=wp.vec3, device="cpu", pinned=True)
host_nrm = wp.zeros(CAP, dtype=wp.vec3, device="cpu", pinned=True)
hview_pos, hview_nrm = host_pos.numpy(), host_nrm.numpy()
# The rubber goes the same way: soup_pos.numpy() would allocate a fresh array
# per call and copy the whole soup twice a frame; a pinned mirror DMAs into
# memory that already exists and .numpy() on it is a view.
host_rub_pos = wp.zeros(n_faces * 3, dtype=wp.vec3, device="cpu", pinned=True)
host_rub_nrm = wp.zeros(n_faces * 3, dtype=wp.vec3, device="cpu", pinned=True)
hview_rub_pos, hview_rub_nrm = host_rub_pos.numpy(), host_rub_nrm.numpy()

# --- the clock --------------------------------------------------------------

sim_t = 0.0          # physics seconds
play_t = 0.0         # playback seconds (what the renderer and the video see)
faces_dead = 0
pop_sim_t = None     # when the first face died
slow_t0 = None       # playback time the slow-mo ramp began
slow_t1 = None       # playback time the ramp-out began


def needle_segment(t):
    """(handle, tip) of the needle at sim time t, or None before it moves."""
    if t < NEEDLE_T0:
        return None
    tip_x = max(NEEDLE_X_START - NEEDLE_SPEED * (t - NEEDLE_T0), NEEDLE_X_STOP)
    tip = (tip_x, NEEDLE_Y, 0.0)
    handle = (tip_x + NEEDLE_LEN, NEEDLE_Y, 0.0)
    return handle, tip


def playback_speed():
    """Sim seconds per playback second, from the slow-mo state machine."""
    global slow_t0, slow_t1
    seg = needle_segment(sim_t)
    if slow_t0 is None:
        if seg is not None and seg[1][0] < SLOWMO_TRIGGER_X:
            slow_t0 = play_t
        return 1.0
    if slow_t1 is None:
        e = min((play_t - slow_t0) / SLOWMO_RAMP_IN, 1.0)
        s = 1.0 + (SLOWMO - 1.0) * (e * e * (3.0 - 2.0 * e))
        held = (pop_sim_t is not None and sim_t > pop_sim_t + SLOWMO_HOLD) or \
               (pop_sim_t is None and play_t - slow_t0 > 6.0)
        if e >= 1.0 and held:
            slow_t1 = play_t
        return s
    e = min((play_t - slow_t1) / SLOWMO_RAMP_OUT, 1.0)
    return SLOWMO + (1.0 - SLOWMO) * (e * e * (3.0 - 2.0 * e))


def substep_body():
    """One physics substep. Device work ONLY -- no host reads, no Python
    branching on sim state -- so it can be captured into a CUDA graph and
    replayed with one launch per substep. The per-substep scalars (dt, the
    needle segment) come in through needle_dev / dt_dev."""
    # rubber: predict
    wp.launch(mem_integrate, dim=n_verts, device=device, inputs=[mx, mprev, ma, dt_dev])
    # fluid: predict
    wp.copy(fx0, fx)
    wp.launch(predict, dim=N, device=device, inputs=[fx, fv, fxp, dt_dev, needle_dev])
    cur, out = fxp, fxt
    pa, pb = ma, mb
    # Refit the contact BVH ONCE per substep, on the predicted rubber, rather
    # than once per coupled iteration. The rubber moves well under a particle
    # radius within a substep, so the query geometry being one iteration stale
    # changes nothing measurable, and the refits per frame stay low.
    wp.copy(mpts, pa)
    shell.refit()
    for k in range(ITERS):
        grid.build(points=cur, radius=H_SPH)
        wp.launch(solve_lambda, dim=N, device=device,
                  inputs=[cur, grid.id, RHO0, EPS_CFM, lam])
        wp.launch(solve_delta, dim=N, device=device,
                  inputs=[cur, lam, grid.id, RHO0, W_DQ, S_CORR_K, needle_dev, out])
        # Contact against the rubber (BVH refit once per substep, above), only
        # on the LAST density iteration: shell_collide is two BVH queries per
        # particle, the most expensive pass in the substep. The density solve
        # is what needs iterating; the shell only has to be enforced once per
        # substep, on the positions that are about to become the result.
        if k == ITERS - 1:
            mdelta.zero_()
            mcnt.zero_()
            wp.launch(shell_collide, dim=N, device=device,
                      inputs=[out, fx0, shell.id, midx, mdelta, mcnt, COUPLE_SHARE])
            wp.launch(mem_apply_push, dim=n_verts, device=device, inputs=[pa, mdelta, mcnt])
        # rubber constraints
        for _ in range(MEM_ITERS):
            wp.launch(mem_solve, dim=n_verts, device=device,
                      inputs=[pa, pb, m_off, m_idx, m_rest, m_stiff, m_pairid, pair_alive])
            pa, pb = pb, pa
        wp.launch(mem_contacts, dim=n_verts, device=device, inputs=[pa, mprev, needle_dev])
        cur, out = out, cur
    wp.launch(finalize, dim=N, device=device, inputs=[fx, cur, fv, dt_dev])
    grid.build(points=fx, radius=H_SPH)
    wp.launch(xsph, dim=N, device=device, inputs=[fx, fv, grid.id, RHO0, fvt])
    wp.copy(fv, fvt)
    wp.copy(mx, pa)
    # tearing: the needle cuts what it touches; the tear runs through taut rubber
    fa_in, fa_out = face_alive, face_alive2
    for _ in range(TEAR_RINGS):
        wp.launch(tear, dim=n_faces, device=device,
                  inputs=[mx, tris, face_nbrs, face_rest, fa_in, fa_out, needle_dev, n_dead])
        fa_in, fa_out = fa_out, fa_in
    if fa_in is not face_alive:
        wp.copy(face_alive, fa_in)
    wp.launch(pair_update, dim=len(pairs), device=device,
              inputs=[pair_faces, pair_kind, face_alive, pair_alive])
    wp.launch(mesh_indices_update, dim=n_faces, device=device,
              inputs=[tris, face_alive, SENTINEL, midx])


substep_graph = None


def capture_substep():
    """Bake substep_body() into a CUDA graph; the substep's launches become one."""
    global substep_graph
    if "--no-graph" in sys.argv:
        return
    try:
        wp.load_module(device=device)          # no JIT inside the capture
        with wp.ScopedCapture(device) as cap:
            substep_body()
        substep_graph = cap.graph
        print("  substep captured as a CUDA graph")
    except Exception as e:                     # noqa: BLE001
        substep_graph = None
        print(f"  note: CUDA graph capture failed ({e}); launching kernels one by one")


_needle_host = np.full((2, 3), 9.0, np.float32)


def substep(dt):
    global sim_t
    seg = needle_segment(sim_t)
    if seg is None:
        _needle_host[:] = 9.0                  # parked far away: every distance test fails
    else:
        _needle_host[0] = seg[0]
        _needle_host[1] = seg[1]
    needle_dev.assign(_needle_host)
    dt_dev.assign(np.float32([dt]))
    if substep_graph is not None:
        wp.capture_launch(substep_graph)
    else:
        substep_body()
    sim_t += dt


def advance_frame():
    """Advance the playback by one frame; the sim by speed/FPS seconds."""
    global play_t, faces_dead, pop_sim_t
    speed = playback_speed()
    span = speed / FPS
    n = max(1, int(math.ceil(span / DT_SUB - 1e-6)))
    if not HEADLESS and n > MAX_SUB_WINDOW:
        n = MAX_SUB_WINDOW           # advance less sim time; playback slows
        span = n * DT_SUB
        speed = span * FPS
    dt = span / n
    n_dead.zero_()
    for _ in range(n):
        substep(dt)
    nd = int(n_dead.numpy()[0])               # the one host read per frame
    if nd:
        faces_dead += nd
        if pop_sim_t is None:
            pop_sim_t = sim_t
            print(f"  pop at t = {sim_t:.3f} s (playback {play_t:.2f} s)")
    play_t += 1.0 / FPS
    return speed


_clamped = False


def build_surface():
    global _clamped
    got = surface.build(fx, N, ISO)
    if got > MAX_TRIS and not _clamped:
        _clamped = True
        print(f"  note: marching cubes produced {got:,} triangles, clamped to {MAX_TRIS:,}")
    return min(got, MAX_TRIS)


def refresh_geometry():
    # water
    ntris = build_surface()
    if ntris > 0:
        surface.expand(ntris, stage_pos, stage_nrm)
        rows = ntris * 3
        wp.copy(host_pos, stage_pos, count=rows)
        wp.copy(host_nrm, stage_nrm, count=rows)
    # rubber
    mnrm.zero_()
    wp.launch(accum_normals, dim=n_faces, device=device, inputs=[mx, tris, face_alive, mnrm])
    wp.launch(scatter, dim=n_faces * 3, device=device,
              inputs=[mx, mnrm, tris, face_alive, soup_pos, soup_nrm])
    wp.copy(host_rub_pos, soup_pos)
    wp.copy(host_rub_nrm, soup_nrm)
    # The pinned-host copies are asynchronous: the sync is what makes the
    # uploads below read finished bytes.
    wp.synchronize_device(device)
    if ntris > 0:
        water_geom.update_attribute("position", hview_pos[:rows])
        water_geom.update_attribute("normal", hview_nrm[:rows])
    water_geom.set_draw_range(0, 3 * ntris)
    rubber_geom.update_attribute("position", hview_rub_pos)
    rubber_geom.update_attribute("normal", hview_rub_nrm)
    seg = needle_segment(sim_t)
    if seg is not None:
        needle.visible = True
        needle.position.set(*seg[0])
    return ntris


# --- threepp scene ----------------------------------------------------------

if not getattr(tp, "HAS_VULKAN", True) or not hasattr(tp, "VulkanRenderer"):
    raise SystemExit("this demo needs the Vulkan renderer (the water is its glass path)")

canvas = tp.Canvas("threepp x warp - water balloon", width=W, height=H,
                   vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 1.0
renderer.shadow_map_enabled = True

scene = tp.Scene()


def make_sky(h=256, w=512):
    """A soft overcast-bright sky with a warm ground: an HDR equirect in numpy.

    No sun disc: the scene's DirectionalLight is the sun. The env is what the
    rubber's highlights and the water's refraction are made of.
    """
    v = (np.arange(h, dtype=np.float32) + 0.5) / h            # 0 = zenith
    u = (np.arange(w, dtype=np.float32) + 0.5) / w
    el = (v - 0.5) * math.pi                                    # elevation (row 0 = nadir as rendered)
    zen = np.float32([0.30, 0.48, 0.85]) * 1.6
    hor = np.float32([0.80, 0.87, 0.98]) * 1.15
    gnd = np.float32([0.32, 0.27, 0.22]) * 0.55
    t = np.clip(np.sin(np.maximum(el, 0.0)), 0.0, 1.0) ** 0.6
    sky = hor[None, :] * (1.0 - t[:, None]) + zen[None, :] * t[:, None]
    img = np.where((el > 0.0)[:, None], sky, gnd[None, :]).astype(np.float32)
    img = np.repeat(img[:, None, :], w, axis=1)
    # a broad bright patch where the sun is (the light itself is the scene's)
    az = u * 2.0 * math.pi
    sun_az, sun_el = math.radians(35.0), math.radians(48.0)
    cosang = (np.sin(el)[:, None] * math.sin(sun_el)
              + np.cos(el)[:, None] * math.cos(sun_el) * np.cos(az[None, :] - sun_az))
    glow = np.exp(-np.maximum(1.0 - cosang, 0.0) * 12.0).astype(np.float32)
    img += glow[:, :, None] * np.float32([2.0, 1.9, 1.6])
    return np.ascontiguousarray(img, dtype=np.float32)


env = tp.float_texture(make_sky())
scene.environment = env
scene.background = env

camera = tp.PerspectiveCamera(38, canvas.aspect(), 0.01, 50)
camera.position.set(0.27, 0.15, 0.40)
camera.look_at(0.0, 0.055, 0.0)

sun = tp.DirectionalLight(0xfff1dc, 2.8)
sun.position.set(1.8, 2.6, 1.2)
sun.cast_shadow = True
scene.add(sun)
scene.add(tp.HemisphereLight(0xcfe0ff, 0x5a4a3a, 0.35))

table = tp.Mesh(tp.BoxGeometry(4.0, 0.04, 2.0), standard_material(0xb08b63, 0.62))
table.position.set(0.0, TABLE_Y - 0.02, -0.3)
table.receive_shadow = True
scene.add(table)

# a studio wall behind: what the water refracts and the rubber reflects
wall = tp.Mesh(tp.BoxGeometry(6.0, 2.0, 0.04), standard_material(0x9fa6ae, 0.9))
wall.position.set(0.0, 0.96, -1.3)
wall.receive_shadow = True
scene.add(wall)

# the rubber: a non-indexed soup, dead faces collapse off screen
rubber_geom = tp.BufferGeometry()
rubber_geom.set_attribute("position", p_mem0[faces.reshape(-1)])
rubber_geom.set_attribute("normal", verts0[faces.reshape(-1)])
rubber = tp.Mesh(rubber_geom, standard_material(0xd4272e, 0.32, side=tp.Side.Double))
rubber.cast_shadow = True
rubber.frustum_culled = False
scene.add(rubber)

# the water: marching-cubes soup, refracting (the Vulkan glass path)
water_geom = tp.BufferGeometry()
water_geom.set_attribute("position", np.zeros((CAP, 3), np.float32))
water_geom.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (CAP, 1)))
water_geom.set_draw_range(0, 3)
WATER_OPAQUE = "--water-opaque" in sys.argv   # debug/perf: skip the glass retrace
water_mat = tp.MeshPhysicalMaterial()
water_mat.color = 0xcfeef5 if not WATER_OPAQUE else 0x9fd0e0
water_mat.roughness = 0.04
water_mat.metalness = 0.0
if not WATER_OPAQUE:
    water_mat.transmission = 1.0
water_mat.ior = 1.333
water_mat.side = tp.Side.Double
water_mat.transparent = False
water = tp.Mesh(water_geom, water_mat)
water.cast_shadow = True
water.frustum_culled = False
scene.add(water)

# the needle: a steel shaft and a cone tip, local +y -> world -x
needle = tp.Group()
steel = standard_material(tp.Color(0.85, 0.86, 0.88), 0.25, 1.0)
_tip_len = 0.008
shaft = tp.Mesh(tp.CylinderGeometry(NEEDLE_R, NEEDLE_R, NEEDLE_LEN - _tip_len, 18), steel)
shaft.position.set(0.0, 0.5 * (NEEDLE_LEN - _tip_len), 0.0)
shaft.cast_shadow = True
needle.add(shaft)
tipm = tp.Mesh(tp.ConeGeometry(NEEDLE_R, _tip_len, 18), steel)
tipm.position.set(0.0, NEEDLE_LEN - 0.5 * _tip_len, 0.0)
needle.add(tipm)
needle.rotation.z = math.pi / 2.0
needle.visible = False
scene.add(needle)


def _caption(img, text):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = load_font(max(14, img.shape[0] // 40))
    x, y = int(img.shape[1] * 0.025), img.shape[0] - int(img.shape[0] * 0.07)
    d.text((x + 1, y + 1), text, font=f, fill=(0, 0, 0))
    d.text((x, y), text, font=f, fill=(232, 236, 240))
    return np.asarray(im)


def film_frame(px, speed):
    """The rendered frame with a small caption burned in."""
    return _caption(px, f"t = {sim_t:6.3f} s   playback {speed:.2f}x")


# --- run --------------------------------------------------------------------

renderer.render(scene, camera)     # first frame: the mesh records now exist
substep_body()                     # first launch compiles + warms every kernel
capture_substep()

if SHOT:
    total = 0
    while sim_t < SHOT_TIME:
        speed = advance_frame()
        total += 1
        if "--trace" in sys.argv and total % 3 == 0:
            seg = needle_segment(sim_t)
            print(f"  frame {total:4d}  play {play_t:6.3f}  sim {sim_t:6.3f}  speed {speed:.3f}"
                  f"  tip_x {seg[1][0] if seg else None}  slow_t0 {slow_t0}  dead {faces_dead}")
        if HEADLESS and sim_t > SHOT_TIME - 0.25 * max(speed, SLOWMO):
            # converge the temporal pipeline on the last frames
            refresh_geometry()
            renderer.sim_time = play_t
            renderer.render(scene, camera)
    refresh_geometry()
    renderer.render(scene, camera)
    px = renderer.read_pixels()
    out = film_frame(px, speed)
    from PIL import Image
    Image.fromarray(out).save("warp_water_balloon.png")
    print(f"simulated {sim_t:.2f} s ({total} frames), rubber faces dead {faces_dead:,}, "
          f"wrote warp_water_balloon.png")
elif VIDEO:
    ff = find_ffmpeg()
    total = int(round(VIDEO * FPS))
    outdir = None
    proc = None
    if ff:
        proc = Encoder("warp_water_balloon.mp4", W, H, FPS, crf=18, an=False,
                       hide_banner=False, loglevel="error", ffmpeg=ff)
    else:
        outdir = tempfile.mkdtemp(prefix="warp_water_balloon_")
        from PIL import Image
    t0 = time.perf_counter()
    for k in range(total):
        speed = advance_frame()
        refresh_geometry()
        renderer.sim_time = play_t
        renderer.render(scene, camera)
        frame = film_frame(renderer.read_pixels(), speed)
        if proc is not None:
            proc.send(frame)
        else:
            Image.fromarray(frame).save(os.path.join(outdir, f"f{k:05d}.png"))
        if k % FPS == 0:
            print(f"  frame {k}/{total}  sim t={sim_t:.3f}s  speed {speed:.2f}x  "
                  f"dead faces {faces_dead:,}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    if proc is not None:
        proc.close()
        print(f"wrote warp_water_balloon.mp4 ({VIDEO:.0f}s playback, sim reached "
              f"{sim_t:.2f}s) in {time.perf_counter() - t0:.0f}s")
    else:
        print(f"ffmpeg not found -- frames left as PNGs in {outdir}")
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(0.0, 0.055, 0.0)
    canvas.on_window_resize(resize_handler(camera, renderer))
    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
    state = {"speed": 1.0, "tris": 0}

    def draw_hud():
        tp.imgui.set_next_window_pos(10, 10)
        tp.imgui.set_next_window_size(320, 0)
        tp.imgui.begin("Warp water balloon")
        tp.imgui.text(f"{N:,} water particles   {state['tris']:,} tris")
        tp.imgui.text(f"rubber {n_faces - faces_dead:,}/{n_faces:,} faces")
        tp.imgui.text(f"sim t {sim_t:7.3f} s   playback {state['speed']:.2f}x")
        tp.imgui.text(f"{tp.imgui.get_framerate():6.1f} fps")
        tp.imgui.separator()
        tp.imgui.text("drag = orbit, scroll = zoom")
        tp.imgui.end()

    def animate():
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
        state["speed"] = advance_frame()
        state["tris"] = refresh_geometry()
        controls.update()
        renderer.render(scene, camera)
        if ui is not None:
            ui.render(draw_hud)

    if FRAMES:
        t0 = time.perf_counter()
        acc = {"sim": 0.0, "geo": 0.0, "render": 0.0, "ui": 0.0, "n": 0}

        RENDER_ONLY = "--render-only" in sys.argv

        def timed():
            a = time.perf_counter()
            if not RENDER_ONLY:
                state["speed"] = advance_frame()
                wp.synchronize_device(device)
            b = time.perf_counter()
            if not RENDER_ONLY:
                state["tris"] = refresh_geometry()
            c = time.perf_counter()
            controls.update()
            renderer.render(scene, camera)
            d = time.perf_counter()
            if ui is not None:
                ui.render(draw_hud)
            acc["sim"] += b - a
            acc["geo"] += c - b
            acc["render"] += d - c
            acc["ui"] += time.perf_counter() - d
            acc["n"] += 1

        for _ in range(FRAMES):
            if not canvas.animate_once(timed):
                break
        wall = time.perf_counter() - t0
        n = max(acc["n"], 1)
        print(f"window {n} frames: {wall / n * 1000:.1f} ms/frame ({n / wall:.1f} fps) | "
              f"sim {acc['sim'] / n * 1000:.1f} ms  geometry {acc['geo'] / n * 1000:.1f} ms  "
              f"render {acc['render'] / n * 1000:.1f} ms  "
              f"ui {acc['ui'] / n * 1000:.1f} ms | "
              f"sim t {sim_t:.2f} s, playback {play_t:.2f} s, dead faces {faces_dead:,}")
        from PIL import Image
        Image.fromarray(renderer.read_pixels()).save("warp_water_balloon_window.png")
    else:
        canvas.animate(animate)

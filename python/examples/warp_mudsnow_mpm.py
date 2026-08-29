"""Granular mud and snow simulated with MLS-MPM in NVIDIA Warp, rendered by threepp.

One test yard, two pits, ONE material point method domain: a single background
grid and a per-particle material id, so the snow and the mud are solved by the
same three kernels and only their constitutive model differs.

    SNOW   a plow blade -- an angled, hard-yawed plate -- makes ONE pass down
           the wall side of the pit. Snow rides up the overhanging face, curls,
           fractures into chunks and spills off the trailing end into a windrow.
           Behind it: a lane scraped to the concrete, a windrow along its inner
           edge, half a bed still standing. Fixed corotated elasticity with a
           singular-value clamp (Stomakhin 2013): the clamp IS the fracture, and
           the hardening exponent is what packs the windrow instead of letting
           it slump.

    MUD    a heavy wheel rolls an S-curve through deep mud. It drops in (splash),
           then churns a trench that holds its walls and only partly slumps back.
           Hencky elasticity with a cohesive Drucker-Prager return map on the
           log singular values (Klar 2016 sand plus a cohesion offset): the
           cohesion is what makes the trench walls stand, the friction angle is
           what lets them creep back in.

Both pits surface through their own particles -> density grid -> marching cubes
(warp_common.DensitySurface) into two triangle-soup meshes. Positions AND the
smooth density-gradient normals go straight into the renderer's own vertex
buffers over CUDA/OpenGL interop -- nothing crosses host memory. The normals are
published winding-aligned (expand's sign=-1.0): wp.MarchingCubes winds opposite
the outward gradient, so the double-sided back-face flip would light outward
normals as pure black.

    pip install warp-lang
    python warp_mudsnow_mpm.py                 # window; drag to orbit, Esc quits
    python warp_mudsnow_mpm.py --vulkan        # ray-traced backend, same interop
    python warp_mudsnow_mpm.py --hdri          # overcast-dome environment light
    python warp_mudsnow_mpm.py --shot 2        # headless PNG at t=2s (mid-pass)
    python warp_mudsnow_mpm.py --shot 6        # ... at t=6s (both passes done)
    python warp_mudsnow_mpm.py --bench         # timed phase breakdown
    python warp_mudsnow_mpm.py --particles 120000   # ~208k is the default
    python warp_mudsnow_mpm.py --surface-every 2   # half the surfacing cost

Needs a CUDA device: the zero-copy surface path is CUDA/OpenGL interop on the
GL backend and VulkanRenderer.enable_vertex_interop + threepp.cuda_interop
(external-memory import) on --vulkan, with a host-copy fallback if the
Vulkan export cannot arm.
"""
import atexit
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (DensitySurface, bench_loop, cli_arg, orbit_loop, parse_size,
                         standard_material)

try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:
    VkInteropArray = None

BENCH = "--bench" in sys.argv
VULKAN = "--vulkan" in sys.argv
HDRI = "--hdri" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.0, float)
N_TARGET = cli_arg("--particles", 230_000, int)
WIDTH, HEIGHT = parse_size(cli_arg("--size", "1600x900", str))
# Surfacing is ~a fifth of the frame; --surface-every 2 halves it at the cost of
# a one-frame-stale mesh on odd frames.
SURFACE_EVERY = cli_arg("--surface-every", 1, int)

# --- the yard -----------------------------------------------------------------
# Long in x, two lanes in z with a curb between them. Everything below is in
# metres; the floor is y = 0.

XP0, XP1 = -1.52, 1.52       # pit interior along the yard's length
ZS0, ZS1 = -0.92, -0.14      # snow lane
ZM0, ZM1 = 0.14, 0.92        # mud lane
ZDIV = 0.14                  # centre curb half-width
CURB_H = 0.14                # curb / perimeter wall height
SNOW_D = 0.130               # fill depth
MUD_D = 0.140

# Particle spacing follows the requested count, and the grid follows the
# spacing: --particles is one knob that moves resolution AND cost together.
_VOL = (XP1 - XP0) * ((ZS1 - ZS0) * SNOW_D + (ZM1 - ZM0) * MUD_D)
PD = float((_VOL / max(N_TARGET, 1000)) ** (1.0 / 3.0))
H = 2.0 * PD                 # grid spacing: 2 particles per cell per axis
INV_H = 1.0 / H
V0 = PD ** 3                 # per-particle rest volume

# The grid pads the yard by two cells on every side, and GY0 is placed so node
# row j == 1 sits exactly on the floor.
GX0 = XP0 - 3.0 * H
GY0 = -H
GZ0 = ZS0 - 3.0 * H
NX = int((XP1 + 3.0 * H - GX0) / H) + 1
NY = int((0.62 - GY0) / H) + 1
NZ = int((ZM1 + 3.0 * H - GZ0) / H) + 1

# --- solver -------------------------------------------------------------------

DT = 8.0e-4
SUBSTEPS = int(math.ceil((1.0 / 60.0) / DT))
GRAV = -9.81
V_MAX = 9.0                  # grid velocity clamp; a blow-up is a NaN otherwise

# --- materials ----------------------------------------------------------------
# Stiffness is turned DOWN rather than substeps up: at 34 substeps of 0.5 ms the
# elastic wave speed sqrt(E/rho) has to stay under ~h/dt, and both of these sit
# comfortably inside it.

# mat[p] is 0 for snow and 1 for mud; the kernels branch on it directly.
E_SNOW, NU_SNOW, RHO_SNOW = 2.6e4, 0.20, 400.0
MU_SNOW = E_SNOW / (2.0 * (1.0 + NU_SNOW))
LA_SNOW = E_SNOW * NU_SNOW / ((1.0 + NU_SNOW) * (1.0 - 2.0 * NU_SNOW))
THETA_C = 2.2e-2             # compressive singular-value clamp -> packing
THETA_S = 7.5e-3             # tensile clamp -> fracture into chunks
XI_SNOW = 8.0                # hardening exponent
MASS_SNOW = RHO_SNOW * V0

E_MUD, NU_MUD, RHO_MUD = 1.8e4, 0.30, 1600.0
MU_MUD = E_MUD / (2.0 * (1.0 + NU_MUD))
LA_MUD = E_MUD * NU_MUD / ((1.0 + NU_MUD) * (1.0 - 2.0 * NU_MUD))
PHI_MUD = math.radians(28.0)                                   # friction angle
ALPHA_MUD = math.sqrt(2.0 / 3.0) * 2.0 * math.sin(PHI_MUD) / (3.0 - math.sin(PHI_MUD))
COH_MUD = 0.0075             # cohesion, in log-strain units: the trench walls
TEN_MUD = 0.010              # tensile strain the mud can carry before it parts
MUD_DAMP = 2.6               # 1/s velocity damping -- reads wet, not sandy
MUD_C = 0.72                 # APIC -> PIC blend for mud: viscosity, cheaply
MASS_MUD = RHO_MUD * V0

# --- colliders ----------------------------------------------------------------

PLOW_EX, PLOW_EY, PLOW_EZ = 0.018, 0.190, 0.345   # blade half-extents
PLOW_TILT = math.radians(24.0)    # a moldboard: the face lifts, so snow rides
                                  # UP it and rolls over instead of being shoved down
# A steep yaw is what turns "churn the whole bed" into "clear one lane": swung
# this far round, the blade's footprint across the lane is 0.51 m of the 0.78 m
# bed, and everything it lifts is cast off the trailing (+z) end into a windrow
# along the boundary instead of being carried the full width.
PLOW_YAW = math.radians(-60.0)
PLOW_SPEED = 0.72
PLOW_T0 = 0.12
PLOW_X0, PLOW_X1 = -1.90, 1.28   # parks inside the lane, wave still on the blade
MU_PLOW = 0.32

_ry = np.array([[math.cos(PLOW_YAW), 0.0, math.sin(PLOW_YAW)],
                [0.0, 1.0, 0.0],
                [-math.sin(PLOW_YAW), 0.0, math.cos(PLOW_YAW)]])
_rz = np.array([[math.cos(PLOW_TILT), -math.sin(PLOW_TILT), 0.0],
                [math.sin(PLOW_TILT), math.cos(PLOW_TILT), 0.0],
                [0.0, 0.0, 1.0]])
PLOW_R_NP = (_ry @ _rz).astype(np.float32)
# Ride height: the lowest corner of the rotated box, plus a scrape clearance.
_drop = (abs(PLOW_R_NP[1, 0]) * PLOW_EX + abs(PLOW_R_NP[1, 1]) * PLOW_EY
         + abs(PLOW_R_NP[1, 2]) * PLOW_EZ)
PLOW_CY = float(_drop) + 0.014
# Lane offset: the rotated box's half-width across the yard, so the blade's
# leading tip runs along the outer wall and the curb half of the bed is never
# touched. One straight pass, one cleared lane, half a bed still standing.
_half_z = (abs(PLOW_R_NP[2, 0]) * PLOW_EX + abs(PLOW_R_NP[2, 1]) * PLOW_EY
           + abs(PLOW_R_NP[2, 2]) * PLOW_EZ)
PLOW_CZ = ZS0 + float(_half_z)

BALL_R = 0.20
BALL_SPEED = 0.80
BALL_T0 = 0.02
BALL_X0 = -1.66
BALL_CZ = 0.5 * (ZM0 + ZM1)
BALL_AMP = 0.145             # S-curve amplitude across the lane
BALL_WAVE = 1.70             # metres per full S
BALL_Y = 0.185               # rolling height: the wheel reaches the pit floor
BALL_DROP = 0.34             # entry height above that
BALL_FALL = 5.0              # entry acceleration (m/s^2 of the scripted drop)
MU_BALL = 0.55

COL_EPS = 0.55 * H           # grid nodes this far outside a collider still react
WEPS = 0.5 * H

# --- warp kernels -------------------------------------------------------------

GRID_DIMS = (NX, NY, NZ)


@wp.func
def box_sdf(q: wp.vec3, e: wp.vec3) -> wp.vec4:
    """Distance and outward normal of an axis-aligned box, packed as (n, d)."""
    d = wp.vec3(wp.abs(q[0]) - e[0], wp.abs(q[1]) - e[1], wp.abs(q[2]) - e[2])
    sx = wp.where(q[0] >= 0.0, 1.0, -1.0)
    sy = wp.where(q[1] >= 0.0, 1.0, -1.0)
    sz = wp.where(q[2] >= 0.0, 1.0, -1.0)
    if d[0] > 0.0 or d[1] > 0.0 or d[2] > 0.0:
        m = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
        dist = wp.length(m)
        inv = 1.0 / wp.max(dist, 1.0e-9)
        return wp.vec4(m[0] * sx * inv, m[1] * sy * inv, m[2] * sz * inv, dist)
    # Inside: the nearest face wins, which is the least-negative slab distance.
    if d[0] >= d[1] and d[0] >= d[2]:
        return wp.vec4(sx, 0.0, 0.0, d[0])
    if d[1] >= d[2]:
        return wp.vec4(0.0, sy, 0.0, d[1])
    return wp.vec4(0.0, 0.0, sz, d[2])


@wp.func
def collide_node(v: wp.vec3, n: wp.vec3, vc: wp.vec3, mu: float) -> wp.vec3:
    """Project a grid velocity out of a moving collider, with Coulomb friction."""
    rel = v - vc
    vn = wp.dot(rel, n)
    if vn >= 0.0:
        return v                       # already separating: nothing to do
    vt = rel - n * vn
    vtl = wp.length(vt)
    if vtl > 1.0e-6:
        # Take back mu * |normal impulse| of tangential motion, never past zero.
        vt = vt * wp.max(0.0, 1.0 + mu * vn / vtl)
    else:
        vt = wp.vec3(0.0, 0.0, 0.0)
    return vc + vt


@wp.func
def mpm_stress(F: wp.mat33, Jp: float, mat: int) -> wp.mat33:
    """Kirchhoff stress tau of one particle, per material."""
    U = wp.mat33()
    V = wp.mat33()
    sig = wp.vec3()
    wp.svd3(F, U, sig, V)
    if mat == 0:
        # Snow: fixed corotated, stiffened by how much plastic compaction the
        # particle has accumulated. exp(xi * (1 - Jp)) is the whole reason a
        # berm stands up instead of flowing back into the lane.
        hd = wp.exp(XI_SNOW * (1.0 - Jp))
        mu = MU_SNOW * hd
        la = LA_SNOW * hd
        R = U * wp.transpose(V)
        J = sig[0] * sig[1] * sig[2]
        return ((F - R) * wp.transpose(F)) * (2.0 * mu) \
            + wp.identity(n=3, dtype=float) * (la * (J - 1.0) * J)
    # Mud: Hencky (log) strain, so the Drucker-Prager return map below acts on
    # the same singular values the stress is built from.
    e0 = wp.log(wp.max(sig[0], 1.0e-4))
    e1 = wp.log(wp.max(sig[1], 1.0e-4))
    e2 = wp.log(wp.max(sig[2], 1.0e-4))
    tr = e0 + e1 + e2
    t = wp.vec3(2.0 * MU_MUD * e0 + LA_MUD * tr,
                2.0 * MU_MUD * e1 + LA_MUD * tr,
                2.0 * MU_MUD * e2 + LA_MUD * tr)
    return U * wp.diag(t) * wp.transpose(U)


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        C: wp.array(dtype=wp.mat33),
        F: wp.array(dtype=wp.mat33),
        Jp: wp.array(dtype=float),
        mat: wp.array(dtype=wp.int32),
        gm: wp.array3d(dtype=float),
        gv: wp.array3d(dtype=wp.vec3),
        dt: float):
    """Scatter mass and APIC momentum, with the MLS-MPM force folded in."""
    p = wp.tid()
    xp = x[p]
    gx = (xp[0] - GX0) * INV_H
    gy = (xp[1] - GY0) * INV_H
    gz = (xp[2] - GZ0) * INV_H
    bi = int(wp.floor(gx - 0.5))
    bj = int(wp.floor(gy - 0.5))
    bk = int(wp.floor(gz - 0.5))
    if bi < 0 or bj < 0 or bk < 0 or bi > NX - 3 or bj > NY - 3 or bk > NZ - 3:
        return
    fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
    # Quadratic B-spline, the standard 3x3x3 stencil.
    wx = wp.vec3(0.5 * (1.5 - fx[0]) * (1.5 - fx[0]),
                 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0),
                 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5))
    wy = wp.vec3(0.5 * (1.5 - fx[1]) * (1.5 - fx[1]),
                 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0),
                 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5))
    wz = wp.vec3(0.5 * (1.5 - fx[2]) * (1.5 - fx[2]),
                 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0),
                 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5))

    m = wp.int32(mat[p])
    pm = wp.where(m == 0, MASS_SNOW, MASS_MUD)
    tau = mpm_stress(F[p], Jp[p], m)
    # MLS-MPM fuses the internal force into the affine scatter: one matrix does
    # the momentum AND the divergence of stress.
    affine = tau * (-dt * V0 * 4.0 * INV_H * INV_H) + C[p] * pm
    mv = v[p] * pm

    for a in range(3):
        for b in range(3):
            for c in range(3):
                w = wx[a] * wy[b] * wz[c]
                dpos = wp.vec3((float(a) - fx[0]) * H,
                               (float(b) - fx[1]) * H,
                               (float(c) - fx[2]) * H)
                wp.atomic_add(gm, bi + a, bj + b, bk + c, w * pm)
                wp.atomic_add(gv, bi + a, bj + b, bk + c,
                              (mv + affine * dpos) * w)


@wp.kernel
def grid_op(gm: wp.array3d(dtype=float),
            gv: wp.array3d(dtype=wp.vec3),
            dt: float,
            plow_c: wp.vec3, plow_r: wp.mat33, plow_v: wp.vec3,
            ball_c: wp.vec3, ball_v: wp.vec3, ball_w: wp.vec3):
    """Momentum -> velocity, gravity, then every boundary the yard has."""
    i, j, k = wp.tid()
    m = gm[i, j, k]
    if m <= 1.0e-11:
        gv[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        return
    v = gv[i, j, k] * (1.0 / m) + wp.vec3(0.0, GRAV * dt, 0.0)
    p = wp.vec3(GX0 + float(i) * H, GY0 + float(j) * H, GZ0 + float(k) * H)

    # The plow blade: an oriented box, so the query goes into its local frame.
    ql = wp.transpose(plow_r) * (p - plow_c)
    bx = box_sdf(ql, wp.vec3(PLOW_EX, PLOW_EY, PLOW_EZ))
    if bx[3] < COL_EPS:
        v = collide_node(v, plow_r * wp.vec3(bx[0], bx[1], bx[2]), plow_v, MU_PLOW)

    # The wheel: sphere plus its rolling surface velocity, which is what flings
    # mud off the back of it rather than merely pushing it aside.
    d = p - ball_c
    dl = wp.length(d)
    if dl < BALL_R + COL_EPS:
        n = d * (1.0 / wp.max(dl, 1.0e-9))
        v = collide_node(v, n, ball_v + wp.cross(ball_w, n * BALL_R), MU_BALL)

    # Centre curb between the lanes.
    if wp.abs(p[2]) < ZDIV + WEPS and p[1] < CURB_H:
        if p[2] >= 0.0:
            v = wp.vec3(v[0] * 0.5, v[1], wp.max(v[2], 0.0))
        else:
            v = wp.vec3(v[0] * 0.5, v[1], wp.min(v[2], 0.0))

    # Perimeter walls.
    if p[0] < XP0 + WEPS:
        v = wp.vec3(wp.max(v[0], 0.0), v[1], v[2])
    if p[0] > XP1 - WEPS:
        v = wp.vec3(wp.min(v[0], 0.0), v[1], v[2])
    if p[2] < ZS0 + WEPS:
        v = wp.vec3(v[0], v[1], wp.max(v[2], 0.0))
    if p[2] > ZM1 - WEPS:
        v = wp.vec3(v[0], v[1], wp.min(v[2], 0.0))

    # Floor: separating, with enough grip that a berm keeps its foot.
    if j <= 1:
        v = wp.vec3(v[0] * 0.45, wp.max(v[1], 0.0), v[2] * 0.45)

    sp = wp.length(v)
    if sp > V_MAX:
        v = v * (V_MAX / sp)
    gv[i, j, k] = v


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        C: wp.array(dtype=wp.mat33),
        F: wp.array(dtype=wp.mat33),
        Jp: wp.array(dtype=float),
        mat: wp.array(dtype=wp.int32),
        gv: wp.array3d(dtype=wp.vec3),
        dt: float):
    """Gather velocity and the affine field, advect, then return-map F."""
    p = wp.tid()
    xp = x[p]
    gx = (xp[0] - GX0) * INV_H
    gy = (xp[1] - GY0) * INV_H
    gz = (xp[2] - GZ0) * INV_H
    bi = int(wp.floor(gx - 0.5))
    bj = int(wp.floor(gy - 0.5))
    bk = int(wp.floor(gz - 0.5))
    if bi < 0 or bj < 0 or bk < 0 or bi > NX - 3 or bj > NY - 3 or bk > NZ - 3:
        v[p] = wp.vec3(0.0, 0.0, 0.0)
        return
    fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
    wx = wp.vec3(0.5 * (1.5 - fx[0]) * (1.5 - fx[0]),
                 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0),
                 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5))
    wy = wp.vec3(0.5 * (1.5 - fx[1]) * (1.5 - fx[1]),
                 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0),
                 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5))
    wz = wp.vec3(0.5 * (1.5 - fx[2]) * (1.5 - fx[2]),
                 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0),
                 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5))

    nv = wp.vec3(0.0, 0.0, 0.0)
    nc = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                w = wx[a] * wy[b] * wz[c]
                g = gv[bi + a, bj + b, bk + c]
                nv += g * w
                nc += wp.outer(g, wp.vec3(float(a) - fx[0],
                                          float(b) - fx[1],
                                          float(c) - fx[2])) * (4.0 * INV_H * w)

    m = wp.int32(mat[p])
    if m == 1:
        # Wet, not sandy: a little PIC in the APIC transfer plus plain drag.
        nc = nc * MUD_C
        nv = nv * wp.max(0.0, 1.0 - MUD_DAMP * dt)

    Fn = (wp.identity(n=3, dtype=float) + nc * dt) * F[p]
    U = wp.mat33()
    V = wp.mat33()
    sig = wp.vec3()
    wp.svd3(Fn, U, sig, V)
    # One guard for the whole solver: a singular value that has left this band
    # is a particle that already blew up, and clamping it is cheaper than
    # letting a NaN propagate through the grid to everything nearby.
    s0 = wp.min(wp.max(sig[0], 0.05), 4.0)
    s1 = wp.min(wp.max(sig[1], 0.05), 4.0)
    s2 = wp.min(wp.max(sig[2], 0.05), 4.0)

    jp = Jp[p]
    if m == 0:
        # Snow: clamp the stretch band. Compression past theta_c is packing (and
        # feeds the hardening); tension past theta_s is a crack, which is what
        # turns the curling wave into separate chunks.
        c0 = wp.min(wp.max(s0, 1.0 - THETA_C), 1.0 + THETA_S)
        c1 = wp.min(wp.max(s1, 1.0 - THETA_C), 1.0 + THETA_S)
        c2 = wp.min(wp.max(s2, 1.0 - THETA_C), 1.0 + THETA_S)
        jp = wp.min(wp.max(jp * (s0 / c0) * (s1 / c1) * (s2 / c2), 0.55), 1.6)
        s0, s1, s2 = c0, c1, c2
    else:
        # Mud: cohesive Drucker-Prager on the log singular values.
        e0 = wp.log(s0)
        e1 = wp.log(s1)
        e2 = wp.log(s2)
        tr = e0 + e1 + e2
        if tr > TEN_MUD:
            # Pulled apart past what the cohesion can hold: drop to the tip of
            # the yield cone, i.e. stress-free. This is where mud strings snap.
            k = wp.exp(TEN_MUD / 3.0)
            s0, s1, s2 = k, k, k
        else:
            h0 = e0 - tr / 3.0
            h1 = e1 - tr / 3.0
            h2 = e2 - tr / 3.0
            fn = wp.sqrt(h0 * h0 + h1 * h1 + h2 * h2)
            dg = fn - COH_MUD + ALPHA_MUD * (3.0 * LA_MUD + 2.0 * MU_MUD) \
                / (2.0 * MU_MUD) * tr
            if dg > 0.0 and fn > 1.0e-8:
                sc = dg / fn
                s0 = wp.exp(e0 - h0 * sc)
                s1 = wp.exp(e1 - h1 * sc)
                s2 = wp.exp(e2 - h2 * sc)
    Jp[p] = jp
    F[p] = U * wp.diag(wp.vec3(s0, s1, s2)) * wp.transpose(V)
    C[p] = nc
    v[p] = nv
    # Belt and braces: nothing leaves the yard, whatever the grid did.
    np_ = xp + nv * dt
    x[p] = wp.vec3(wp.min(wp.max(np_[0], XP0 + WEPS), XP1 - WEPS),
                   wp.min(wp.max(np_[1], 0.25 * H), 0.60),
                   wp.min(wp.max(np_[2], ZS0 + WEPS), ZM1 - WEPS))


# --- seeding ------------------------------------------------------------------


def fill(x0, x1, z0, z1, depth, pad):
    """A jittered lattice filling one pit, as an (n, 3) float32 array."""
    nx = max(1, int((x1 - x0 - 2 * pad) / PD))
    nz = max(1, int((z1 - z0 - 2 * pad) / PD))
    ny = max(1, int(depth / PD))
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.stack([x0 + pad + (ix.ravel() + 0.5) * PD,
                     0.5 * PD + (iy.ravel() + 0.5) * PD,
                     z0 + pad + (iz.ravel() + 0.5) * PD], axis=-1).astype(np.float32)


wp.init()
device = wp.get_preferred_device()

rng = np.random.default_rng(7)
snow_p = fill(XP0, XP1, ZS0, ZS1, SNOW_D, 0.6 * PD)
mud_p = fill(XP0, XP1, ZM0, ZM1, MUD_D, 0.6 * PD)
N_SNOW, N_MUD = len(snow_p), len(mud_p)
p0 = np.concatenate([snow_p, mud_p])
p0 = (p0 + rng.uniform(-0.22 * PD, 0.22 * PD, p0.shape)).astype(np.float32)
N = len(p0)
mat_np = np.concatenate([np.zeros(N_SNOW, np.int32), np.ones(N_MUD, np.int32)])

x = wp.array(p0, dtype=wp.vec3, device=device)
v = wp.zeros(N, dtype=wp.vec3, device=device)
Cm = wp.zeros(N, dtype=wp.mat33, device=device)
Fm = wp.array(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)), dtype=wp.mat33,
              device=device)
Jpm = wp.array(np.ones(N, np.float32), dtype=float, device=device)
matm = wp.array(mat_np, dtype=wp.int32, device=device)
gm = wp.zeros(GRID_DIMS, dtype=float, device=device)
gv = wp.zeros(GRID_DIMS, dtype=wp.vec3, device=device)

# Particles are contiguous by material, so each pit surfaces from a view.
snow_x = x[:N_SNOW]
mud_x = x[N_SNOW:]

print(f"mud/snow MLS-MPM: {N:,} particles ({N_SNOW:,} snow + {N_MUD:,} mud) "
      f"on {device}\n"
      f"  grid {NX}x{NY}x{NZ} @ h={H * 1000:.1f} mm, particle spacing "
      f"{PD * 1000:.1f} mm, dt={DT * 1e6:.0f} us x {SUBSTEPS} substeps/frame")

# --- surfacing ----------------------------------------------------------------

CELL = 1.15 * PD
MAX_TRIS = cli_arg("--max-tris", 260_000, int)
_stage = None   # shared host-fallback staging pair; publish() is sequential


class Pit:
    """One pit's particles -> density grid -> marching cubes -> a threepp mesh."""

    def __init__(self, points, z0, z1, iso_k, material, grain, grain_freq):
        self.grain = grain
        self.grain_freq = grain_freq
        origin = (XP0 - 4 * CELL, -2 * CELL, z0 - 4 * CELL)
        dims = (int((XP1 + 4 * CELL - origin[0]) / CELL) + 1,
                int((0.46 - origin[1]) / CELL) + 1,
                int((z1 + 4 * CELL - origin[2]) / CELL) + 1)
        self.dims = dims
        self.surface = DensitySurface(origin, CELL, dims, device)
        self.points = points
        self.n = points.shape[0]
        # A cell fully inside the material collects (CELL/PD)^3 particles, so
        # the iso level tracks the resolution rather than being a magic number.
        self.iso = iso_k * (CELL / PD) ** 3
        self.ntris = 0
        self.reg = None       # GL: (position, normal) RegisteredGLBuffer pair
        self.vk = None        # Vulkan: (position, normal) VkInteropArray pair
        self.vk_ntris = 0     # triangle count the in-render callback expands
        cap = MAX_TRIS * 3
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", np.zeros((cap, 3), np.float32))
        self.geometry.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (cap, 1)))
        self.geometry.set_draw_range(0, 3)
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.cast_shadow = True
        self.mesh.receive_shadow = True
        self.mesh.frustum_culled = False     # the CPU-side bounds never see GPU writes

    def build(self):
        self.ntris = min(self.surface.build(self.points, self.n, self.iso), MAX_TRIS)

    def publish(self, renderer):
        """Both vertex attributes straight into the renderer's own buffers.

        The expand kernel writes positions and normals through the renderer's
        buffers (GL: interop-mapped VBOs here; Vulkan: the imported exports,
        from the in-render callback), nothing via the host. sign=-1.0 is the
        winding: marching cubes emits these triangles wound the other way
        round, and a double-sided material flips the shading normal on the
        face the camera actually sees, so the attribute has to arrive
        pre-flipped or every lit surface reads inside-out (pure black under
        this scene's lights).
        """
        if self.vk is not None:
            # Zero copy: nothing to push. _vk_on_frame() expands from inside
            # the renderer's frame; the drawRange published here is what its
            # raster draw, BLAS build and interop copies clamp to.
            self.vk_ntris = self.ntris
            self.geometry.set_draw_range(0, 3 * self.ntris)
            return
        if VULKAN:
            self._publish_host()
            return
        if self.reg is None:
            pid = renderer.gl_buffer_id(self.geometry, "position")
            nid = renderer.gl_buffer_id(self.geometry, "normal")
            if pid is None or nid is None:
                return
            flags = wp.RegisteredGLBuffer.WRITE_DISCARD
            self.reg = (wp.RegisteredGLBuffer(int(pid), device, flags),
                        wp.RegisteredGLBuffer(int(nid), device, flags))
        if self.ntris > 0:
            dp = self.reg[0].map(dtype=wp.vec3, shape=(MAX_TRIS * 3,))
            dn = self.reg[1].map(dtype=wp.vec3, shape=(MAX_TRIS * 3,))
            self.surface.expand(self.ntris, dp, dn, sign=-1.0,
                                grain=self.grain, grain_freq=self.grain_freq)
            self.reg[0].unmap()
            self.reg[1].unmap()
        self.geometry.set_draw_range(0, 3 * self.ntris)

    def _publish_host(self):
        """Fallback when the Vulkan export cannot arm: one device->host hop.

        Plain synchronous staging -- the .numpy() copy is the sync -- because
        this only runs when the zero-copy route is unavailable and simple
        beats pipelined for a fallback.
        """
        global _stage
        if self.ntris > 0:
            if _stage is None:
                _stage = (wp.zeros(MAX_TRIS * 3, dtype=wp.vec3, device=device),
                          wp.zeros(MAX_TRIS * 3, dtype=wp.vec3, device=device))
            self.surface.expand(self.ntris, _stage[0], _stage[1], sign=-1.0,
                                grain=self.grain, grain_freq=self.grain_freq)
            rows = 3 * self.ntris
            self.geometry.update_attribute("position", _stage[0][:rows].numpy())
            self.geometry.update_attribute("normal", _stage[1][:rows].numpy())
        self.geometry.set_draw_range(0, 3 * self.ntris)

    def _vk_on_frame(self):
        """Runs inside render(), post-fence and pre-record: write the imported
        arrays over exactly the surface build() left behind, then synchronize.
        The synchronize is MANDATORY -- host ordering is the only thing
        sequencing the CUDA write against the Vulkan frame that reads it."""
        if self.vk_ntris > 0:
            self.surface.expand(self.vk_ntris, self.vk[0].array, self.vk[1].array,
                                sign=-1.0, grain=self.grain,
                                grain_freq=self.grain_freq)
        wp.synchronize_device(device)

    def arm_vulkan(self, renderer, name):
        """Point the expand kernel at the renderer's vertex buffers. True if live.

        Must run AFTER the first render(): the record the export comes from is
        created on the frame the mesh is first drawn. Any failure leaves the
        pit on the host route.
        """
        if VkInteropArray is None or not hasattr(renderer, "enable_vertex_interop"):
            print(f"  note: no CUDA->Vulkan vertex interop in this build -- "
                  f"{name} takes the host route")
            return False
        try:
            # The soup re-triangulates every frame -- one changed cell shifts
            # every later vertex slot -- so per-vertex motion history is noise:
            # declare it and the mesh reprojects world-static instead of
            # flickering under the temporal passes.
            h = renderer.enable_vertex_interop(self.mesh, self._vk_on_frame,
                                               stable_correspondence=False)
        except TypeError:   # build predating stable_correspondence
            h = renderer.enable_vertex_interop(self.mesh, self._vk_on_frame)
        if h is None:
            print(f"  note: vulkan vertex interop did not arm for {name} -- "
                  f"host route (the renderer prints the reason on stderr)")
            return False
        (pos_handle, pos_bytes), (nrm_handle, nrm_bytes) = h
        cap = MAX_TRIS * 3
        try:
            self.vk = (VkInteropArray(pos_handle, pos_bytes, wp.vec3, cap, device),
                       VkInteropArray(nrm_handle, nrm_bytes, wp.vec3, cap, device))
        except Exception as e:
            print(f"  note: CUDA import of the vulkan export failed for {name} "
                  f"({e}) -- host route")
            renderer.disable_vertex_interop(self.mesh)
            self.vk = None
            return False
        # Degenerate the whole capacity once: the exports are fresh VRAM, and
        # everything downstream clamps to the drawRange -- but a consumer that
        # ever forgot the clamp should read a harmless off-screen point.
        self.surface.expand(0, self.vk[0].array, self.vk[1].array, dim=MAX_TRIS)
        wp.synchronize_device(device)
        # Ordered teardown: the CUDA mappings must go before the renderer frees
        # the Vulkan memory they point at; interpreter shutdown alone would
        # collect them in an arbitrary order.
        atexit.register(self._release_vulkan, renderer)
        return True

    def _release_vulkan(self, renderer):
        if self.vk is None:
            return
        pair, self.vk = self.vk, None
        for a in pair:
            a.close()
        renderer.disable_vertex_interop(self.mesh)


# --- collider kinematics ------------------------------------------------------


def plow_state(t):
    """(centre, velocity) of the blade at time t."""
    s = PLOW_X0 + PLOW_SPEED * max(0.0, t - PLOW_T0)
    moving = PLOW_T0 < t <= PLOW_T0 + (PLOW_X1 - PLOW_X0) / PLOW_SPEED
    return ((min(s, PLOW_X1), PLOW_CY, PLOW_CZ),
            (PLOW_SPEED if moving else 0.0, 0.0, 0.0))


def ball_state(t):
    """(centre, linear velocity, angular velocity) of the wheel at time t."""
    tt = max(0.0, t - BALL_T0)
    bx = min(BALL_X0 + BALL_SPEED * tt, 1.30)   # parks at the end of its own trench
    k = 2.0 * math.pi / BALL_WAVE
    bz = BALL_CZ + BALL_AMP * math.sin(k * (bx - BALL_X0))
    vz = BALL_AMP * k * math.cos(k * (bx - BALL_X0)) * BALL_SPEED
    lift = BALL_DROP - BALL_FALL * tt * tt
    if lift > 0.0:
        by, vy = BALL_Y + lift, -2.0 * BALL_FALL * tt
    else:
        by, vy = BALL_Y, 0.0
    vx = BALL_SPEED if bx < 1.30 else 0.0
    # Rolling without slipping: omega = up x v / R, so the contact point stands
    # still and the top of the wheel throws mud forward.
    return ((bx, by, bz), (vx, vy, vz), (vz / BALL_R, 0.0, -vx / BALL_R))


# --- simulation loop ----------------------------------------------------------

sim_time = 0.0
PLOW_R = wp.mat33(*PLOW_R_NP.reshape(-1).tolist())


def sim_step():
    """Advance one 60 fps frame."""
    global sim_time
    for _ in range(SUBSTEPS):
        pc, pv = plow_state(sim_time)
        bc, bv, bw = ball_state(sim_time)
        gm.zero_()
        gv.zero_()
        wp.launch(p2g, dim=N, device=device,
                  inputs=[x, v, Cm, Fm, Jpm, matm, gm, gv, DT])
        wp.launch(grid_op, dim=GRID_DIMS, device=device,
                  inputs=[gm, gv, DT, wp.vec3(*pc), PLOW_R, wp.vec3(*pv),
                          wp.vec3(*bc), wp.vec3(*bv), wp.vec3(*bw)])
        wp.launch(g2p, dim=N, device=device,
                  inputs=[x, v, Cm, Fm, Jpm, matm, gv, DT])
        sim_time += DT


def sim_step_timed():
    """The same three phases, each synced, for --bench."""
    global sim_time
    acc = [0.0, 0.0, 0.0]
    for _ in range(SUBSTEPS):
        pc, pv = plow_state(sim_time)
        bc, bv, bw = ball_state(sim_time)
        t0 = time.perf_counter()
        gm.zero_()
        gv.zero_()
        wp.launch(p2g, dim=N, device=device,
                  inputs=[x, v, Cm, Fm, Jpm, matm, gm, gv, DT])
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        wp.launch(grid_op, dim=GRID_DIMS, device=device,
                  inputs=[gm, gv, DT, wp.vec3(*pc), PLOW_R, wp.vec3(*pv),
                          wp.vec3(*bc), wp.vec3(*bv), wp.vec3(*bw)])
        wp.synchronize_device(device)
        t2 = time.perf_counter()
        wp.launch(g2p, dim=N, device=device,
                  inputs=[x, v, Cm, Fm, Jpm, matm, gv, DT])
        wp.synchronize_device(device)
        t3 = time.perf_counter()
        acc[0] += t1 - t0
        acc[1] += t2 - t1
        acc[2] += t3 - t2
        sim_time += DT
    return acc


# --- threepp scene ------------------------------------------------------------

if VULKAN and not tp.vulkan_available():
    print("vulkan not available on this machine; falling back to OpenGL")
    VULKAN = False

canvas = tp.Canvas("threepp x warp - mud & snow MPM", width=WIDTH, height=HEIGHT,
                   antialiasing=4, headless=SHOT or BENCH)
if VULKAN:
    renderer = tp.VulkanRenderer(canvas)
else:
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 1.15

scene = tp.Scene()
scene.background = 0x141920
# The backdrop is the background colour and the floor fading into it: no
# horizon line to fight the yard for attention, and a gradient behind the slab
# instead of a flat void. The yard is inside 6 m, so nothing in it is touched.
scene.set_fog(0x141920, 4.5, 18.0)


def make_overcast_hdr(path, W=1024, HH=512):
    """A sunless winter overcast dome, written as a Radiance .hdr.

    Deliberately carries NO sun disk -- the scene's DirectionalLight stays the
    one sun -- so what the environment adds is the thing an overcast sky IS: a
    huge soft area light. Glossy mud then picks up wide sheet highlights
    instead of point glints, and snow gets sky-coloured fill in its cavities.
    A broad bright smear sits at the key light's azimuth so the environment's
    speculars agree with the sun's direction.
    """
    j = np.arange(HH).reshape(HH, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / HH) * math.pi
    phi = (i / W) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (HH, W))
    sin_t = np.sin(theta)
    d = np.stack([sin_t * np.cos(phi), y, sin_t * np.sin(phi)], axis=-1)
    up = np.clip(y, 0.0, 1.0)[..., None]
    t = up ** 0.6
    sky = np.array([0.52, 0.50, 0.47]) * (1.0 - t) + np.array([0.30, 0.34, 0.42]) * t
    sun_dir = np.float32([3.2, 2.15, 1.6])
    sun_dir /= np.linalg.norm(sun_dir)
    ang = np.arccos(np.clip((d * sun_dir).sum(-1), -1.0, 1.0))
    sky = sky + np.exp(-(ang / math.radians(35.0)) ** 2)[..., None] \
        * np.array([0.65, 0.55, 0.42])
    # Below the horizon: dark cold ground, so upward-facing glossy facets do
    # not reflect a sky that is not there.
    sky = np.where((y < 0)[..., None], np.array([0.055, 0.052, 0.048]), sky)
    rgb = np.maximum(sky, 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    mant, exp = np.frexp(np.where(mask, m, 1.0))
    scale = np.where(mask, mant * 256.0 / np.where(mask, m, 1.0), 0.0)
    rgbe = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    for c in range(3):
        rgbe[..., c] = np.clip(rgb[..., c] * scale, 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    # An RGBE row starting (2, 2, <128) means "adaptive RLE" to a .hdr reader;
    # nudge the one pixel that could fake that signature (same trick as
    # warp_fluid's make_sky_hdr).
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (HH, W))
        f.write(rgbe.tobytes())
    return path


if HDRI:
    import tempfile
    scene.environment = tp.RGBELoader().load(make_overcast_hdr(
        os.path.join(tempfile.gettempdir(), "threepp_mudsnow_sky.hdr")))

# Three-quarters down the yard's length from the snow side, low enough that the
# windrow and the trench walls are seen in profile rather than from above, and
# far enough back that the whole slab sits inside the frame with a margin.
camera = tp.PerspectiveCamera(46, canvas.aspect(), 0.05, 60)
camera.position.set(-2.24, 1.30, -1.82)
camera.look_at(0.02, -0.05, 0.0)

scene.add(tp.HemisphereLight(0xc4cfdc, 0x39352e, 0.80))
# A low warm key from over the near-left corner, raking down the yard: the
# lanes' own front faces are lit, and every chunk, windrow and trench wall
# throws a shadow as long as it is tall across the concrete behind it. That
# shadow relief is the whole difference between "rubble" and "carved".
sun = tp.DirectionalLight(0xffdca8, 3.4)
sun.position.set(3.2, 2.15, 1.6)
sun.cast_shadow = True
sun.set_shadow_frustum(-3.4, 3.4, 3.4, -3.4)
sun.set_shadow_bias(-0.0009)
scene.add(sun)

# Both surfaces are marching-cubes triangle soups: double-sided, because the
# winding is the library's business, and smooth shaded off the density-gradient
# normal Pit.publish writes device-side. Flat shading is the wrong read for either
# material -- it turns fractured snow into styrofoam and wet mud into faceted glints.
snow_mat = standard_material(0xf2f6fa, 0.95, side=tp.Side.Double, flat_shading=False)
# Physical, for specular_intensity: mud is porous, so its specular should be
# WEAK rather than merely blurred — turning F0 down eats the raking-angle sun
# glance that roughness alone only kills at cost of reading bone dry.
mud_mat = tp.MeshPhysicalMaterial()
mud_mat.color = 0x54371f
mud_mat.roughness = 0.5
mud_mat.specular_intensity = 0.3
mud_mat.side = tp.Side.Double
mud_mat.flat_shading = False

# Grain: micro-relief bent into the expand normals (see DensitySurface.expand).
# Snow is fine and crystalline -- small strong grain; mud is gloopy -- larger,
# softer lumps, so its wet highlight breaks into patches instead of a sheen.
snow_pit = Pit(snow_x, ZS0, ZS1, 0.42, snow_mat, grain=0.45, grain_freq=55.0)
mud_pit = Pit(mud_x, ZM0, ZM1, 0.50, mud_mat, grain=0.35, grain_freq=20.0)
scene.add(snow_pit.mesh)
scene.add(mud_pit.mesh)

# The yard: a slab, a centre curb and a perimeter of low walls. Concrete, so the
# two materials read against something neutral.
concrete = standard_material(0x8e8b85, 0.9)
slab = tp.Mesh(tp.BoxGeometry(XP1 - XP0 + 0.24, 0.10, ZM1 - ZS0 + 0.24), concrete)
slab.position.set(0.5 * (XP0 + XP1), -0.05, 0.5 * (ZS0 + ZM1))
slab.receive_shadow = True
scene.add(slab)

curb = tp.Mesh(tp.BoxGeometry(XP1 - XP0 + 0.24, CURB_H, 2 * ZDIV), concrete)
curb.position.set(0.5 * (XP0 + XP1), 0.5 * CURB_H, 0.0)
curb.cast_shadow = True
curb.receive_shadow = True
scene.add(curb)

WT = 0.09
for cx, cz, sx, sz in ((0.5 * (XP0 + XP1), ZS0 - 0.5 * WT, XP1 - XP0 + 0.24, WT),
                       (0.5 * (XP0 + XP1), ZM1 + 0.5 * WT, XP1 - XP0 + 0.24, WT),
                       (XP0 - 0.5 * WT, 0.5 * (ZS0 + ZM1), WT, ZM1 - ZS0 + 0.24),
                       (XP1 + 0.5 * WT, 0.5 * (ZS0 + ZM1), WT, ZM1 - ZS0 + 0.24)):
    w = tp.Mesh(tp.BoxGeometry(sx, CURB_H, sz), concrete)
    w.position.set(cx, 0.5 * CURB_H, cz)
    w.cast_shadow = True
    w.receive_shadow = True
    scene.add(w)

ground = tp.Mesh(tp.PlaneGeometry(40, 40), standard_material(0x1e2329, 0.95))
ground.rotate_x(-math.pi / 2)
ground.position.y = -0.101
ground.receive_shadow = True
scene.add(ground)

# The plow: the blade the SDF collides against, and nothing else. A frame behind
# it sits exactly between this camera and the curling wave, so the blade carries
# the read on its own.
plow = tp.Group()
blade = tp.Mesh(tp.BoxGeometry(2 * PLOW_EX, 2 * PLOW_EY, 2 * PLOW_EZ),
                standard_material(0xd8dde3, 0.42, 0.10))
# Euler order is XYZ, so rotation (0, yaw, tilt) IS Ry(yaw) @ Rz(tilt) -- the
# same matrix the SDF query transposes.
blade.rotation.y = PLOW_YAW
blade.rotation.z = PLOW_TILT
blade.cast_shadow = True
blade.receive_shadow = True
plow.add(blade)
scene.add(plow)

# The wheel: a heading group (yawed to the S-curve tangent) holding a spin group,
# so the tread visibly rolls instead of sliding.
wheel = tp.Group()
spin = tp.Group()
tyre = tp.Mesh(tp.SphereGeometry(BALL_R * 0.93, 28, 20),
               standard_material(0x3a3a3e, 0.7))
tyre.cast_shadow = True
spin.add(tyre)
tread = tp.Mesh(tp.TorusGeometry(BALL_R * 0.88, BALL_R * 0.30, 14, 26),
                standard_material(0x202024, 0.55))
tread.cast_shadow = True
spin.add(tread)
hub = tp.Mesh(tp.CylinderGeometry(BALL_R * 0.34, BALL_R * 0.34, BALL_R * 0.9, 20),
              standard_material(0x9aa0a6, 0.35, 0.7))
hub.rotate_x(math.pi / 2)
spin.add(hub)
wheel.add(spin)
scene.add(wheel)


def place_colliders():
    """Move the rendered colliders onto this frame's kinematic state."""
    pc, _ = plow_state(sim_time)
    plow.position.set(*pc)
    bc, bv, _ = ball_state(sim_time)
    wheel.position.set(*bc)
    wheel.rotation.y = math.atan2(-bv[2], bv[0])
    spin.rotate_z(-math.hypot(bv[0], bv[2]) / BALL_R * (1.0 / 60.0))


frame_no = 0


def frame():
    """One simulated frame, its two surfaces, and the collider poses."""
    global frame_no
    sim_step()
    if frame_no % SURFACE_EVERY == 0:
        snow_pit.build()
        mud_pit.build()
        snow_pit.publish(renderer)
        mud_pit.publish(renderer)
    place_colliders()
    frame_no += 1


def render():
    renderer.render(scene, camera)


def save_shot(path):
    if VULKAN:
        renderer.save_frame(scene, camera, path)   # renders + reads back
    else:
        render()
        renderer.save_frame(path)


# Neither backend's vertex buffers exist until the renderer has drawn each mesh
# once: gl_buffer_id returns None before the first render, and the record the
# Vulkan export comes from is built on the frame the mesh is first seen.
render()
if VULKAN:
    armed = snow_pit.arm_vulkan(renderer, "snow") + mud_pit.arm_vulkan(renderer, "mud")
    print("vulkan surface route: " + {2: "zero-copy CUDA -> Vulkan",
                                      1: "mixed (see notes above)",
                                      0: "host copy"}[armed])
snow_pit.publish(renderer)
mud_pit.publish(renderer)

if BENCH:
    for _ in range(20):
        frame()
        render()

    def timed():
        p, g, gg = sim_step_timed()
        t0 = time.perf_counter()
        snow_pit.build()
        mud_pit.build()
        snow_pit.publish(renderer)
        mud_pit.publish(renderer)
        wp.synchronize_device(device)
        place_colliders()
        return p, g, gg, time.perf_counter() - t0

    bench_loop(timed, render, ("p2g", "grid", "g2p", "surface"),
               warmup=0, timed=40,
               label=f"{N:,} particles, {SUBSTEPS} substeps "
                     f"[{'vulkan' if VULKAN else 'opengl'}]")
    # The phase columns cost three device syncs per substep. This one does not.
    for _ in range(10):
        frame()
        render()
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(60):
        frame()
        render()
    wp.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1000.0 / 60.0
    print(f"  unsynced frame: {ms:.2f} ms/frame ({1000.0 / ms:.0f} fps)  "
          f"{snow_pit.ntris + mud_pit.ntris:,} tris")
    save_shot("warp_mudsnow_mpm.png")
elif SHOT:
    total = int(round(SHOT_TIME * 60))
    # The Vulkan pipeline is TEMPORAL: probe GI, denoisers and the upscaler
    # converge over frames, so a single render after the sim loop would capture
    # frame ONE of all of them. Render the last stretch so the shot is the
    # CONVERGED image; GL needs no history, one render before the save is it.
    warm = min(total, 90) if VULKAN else 0
    for i in range(total):
        frame()
        if i >= total - warm:
            render()
    save_shot("warp_mudsnow_mpm.png")
    print(f"simulated {SHOT_TIME:.1f} s, {snow_pit.ntris:,} snow + "
          f"{mud_pit.ntris:,} mud triangles, wrote warp_mudsnow_mpm.png")
else:
    orbit_loop(canvas, renderer, scene, camera, frame, target=(-0.1, 0.05, 0.0))

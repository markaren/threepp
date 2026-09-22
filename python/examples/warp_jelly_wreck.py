"""A wall of jelly blocks smashed by a chrome wrecking ball: Warp + threepp.

Every block is a real deformable body: a 5x5x5 particle lattice braced by
axis, face-diagonal and body-diagonal distance constraints, solved with
graph-coloured Gauss-Seidel XPBD (no two constraints in a colour share a
particle, so each colour is one race-free launch). Blocks touch each other
through their particles: a wp.HashGrid over every particle in the scene,
rebuilt every substep, pushes apart particles of different blocks and takes
back part of their relative sliding (friction). The wrecking ball is a heavy
pendulum on a chain that simply shoves particles out of its way.

What threepp draws is not the lattice: each block is a smooth rounded cube
(1200 triangles) EMBEDDED in its lattice with fixed trilinear weights, so the
coarse physics drives a fine glossy surface -- the classic cage-deformation
trick. Rendered on the Vulkan hybrid path tracer, the jelly is transmissive
and refracts the blocks behind it.

    python warp_jelly_wreck.py                 # window; drag to orbit, Esc quits
    python warp_jelly_wreck.py --video 10      # headless film -> C:/dev/_softbody_films/jelly_wreck
    python warp_jelly_wreck.py --video 10 --size 960x540   # quick preview
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
                         orbit_loop, parse_size, scatter_soup, standard_material)

VIDEO = cli_arg("--video", 0.0, float)
W, H = parse_size(cli_arg("--size", "1920x1080", str))
OUT_DIR = cli_arg("--out-dir", r"C:\dev\_softbody_films\jelly_wreck", str)
TAG = cli_arg("--tag", "jelly_wreck", str)
OPAQUE = "--opaque" in sys.argv

# --- tunables ---------------------------------------------------------------

BLOCK = 0.40                 # block side (m)
R_P = 0.035                  # particle contact radius
NC = 4                       # lattice cells per block side
NL = NC + 1
HL = 0.5 * BLOCK - R_P       # lattice half extent: particles + radius = the block
SPACING = 2.0 * HL / NC
NP = NL ** 3                 # particles per block

SUB_DT = 1.0 / 1440.0        # one substep; a normal-speed frame is 24 of them
SUBS_PER_FRAME = 24
STIFF = cli_arg("--stiff", 0.03, float)      # jelly: GS stiffness per substep
DAMP = cli_arg("--damp", 0.00012, float)     # per substep (air + internal loss)
DAMP_SETTLE = 0.02
MIRROR = cli_arg("--mirror", 0.5, float)   # centre-strut stiffness factor
KD = cli_arg("--kd", 0.004, float)       # relative-velocity damping along struts
MAX_PUSH = wp.constant(0.006)            # cap on a particle-particle correction per substep
MU_PP = cli_arg("--mu", 0.6, float)   # block-block Coulomb friction
MU_FLOOR = 0.12
MU_BALL = 0.05
GRAV = wp.constant(wp.vec3(0.0, -9.81, 0.0))
QRAD = 2.0 * R_P + 0.02      # hash query radius: contact + one substep of slack

# Wrecking ball: a pendulum swinging in the y-z plane through the wall.
R_B = cli_arg("--ball-r", 0.50, float)
PIVOT = np.array([0.0, 5.75, 0.0])
L_PEND = 4.55
PHI0 = math.radians(-cli_arg("--swing", 52.0, float))
PEND_DAMP = 0.02

# Wall: brick bond, 7 / 6 blocks per row, 6 rows, 2 deep.
ROWS = 6
GAP = 0.004

PALETTE = [(1.00, 0.16, 0.24),   # raspberry
           (0.42, 0.92, 0.14),   # lime
           (1.00, 0.52, 0.10),   # orange
           (0.22, 0.45, 1.00),   # blueberry
           (0.74, 0.22, 1.00),   # grape
           (1.00, 0.86, 0.16)]   # lemon

# --- lattice template (one block, numpy) ------------------------------------


def lid(i, j, k):
    return (i * NL + j) * NL + k


lat = np.array([(-HL + i * SPACING, -HL + j * SPACING, -HL + k * SPACING)
                for i in range(NL) for j in range(NL) for k in range(NL)], np.float32)
dirs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)]
tcons = []
for i in range(NL):
    for j in range(NL):
        for k in range(NL):
            for dx, dy, dz in dirs:
                a, b, c = i + dx, j + dy, k + dz
                if 0 <= a < NL and 0 <= b < NL and 0 <= c < NL:
                    tcons.append((lid(i, j, k), lid(a, b, c)))
n_local = len(tcons)
# Long struts through the centre (each particle to its mirror image): a soft
# memory of the whole block's shape, so a hard hit cannot leave it crumpled.
for n_ in range(NP // 2):
    tcons.append((n_, NP - 1 - n_))
tfac = np.ones(len(tcons), np.float32)
tfac[n_local:] = MIRROR
used = [0] * NP
tcol = np.empty(len(tcons), np.int32)
for n, (i, j) in enumerate(tcons):
    m = used[i] | used[j]
    c = 0
    while (m >> c) & 1:
        c += 1
    tcol[n] = c
    used[i] |= 1 << c
    used[j] |= 1 << c
n_colors = int(tcol.max()) + 1
tci = np.array([c[0] for c in tcons], np.int32)
tcj = np.array([c[1] for c in tcons], np.int32)

# --- render template: a rounded cube embedded in the lattice ----------------

M = cli_arg("--facegrid", 10, int)
RHO = 0.055                  # edge rounding radius
h = 0.5 * BLOCK
pts, tris_l = [], []
g = np.sin(0.5 * math.pi * np.linspace(-1.0, 1.0, M + 1))   # denser near edges
for ax in range(3):
    for s in (-1.0, 1.0):
        b_ax, c_ax = [a for a in range(3) if a != ax]
        base = len(pts)
        for u in range(M + 1):
            for v in range(M + 1):
                p = np.zeros(3)
                p[ax] = s
                p[b_ax] = g[u]
                p[c_ax] = g[v]
                pts.append(p * h)
        for u in range(M):
            for v in range(M):
                a0 = base + u * (M + 1) + v
                tris_l.append((a0, a0 + M + 1, a0 + 1))
                tris_l.append((a0 + 1, a0 + M + 1, a0 + M + 2))
pts = np.array(pts)
inner = np.clip(pts, -(h - RHO), h - RHO)
dv = pts - inner
dl = np.linalg.norm(dv, axis=1, keepdims=True)
pts = inner + RHO * dv / np.maximum(dl, 1e-9)
key = np.round(pts * 1e5).astype(np.int64)
_, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
inv = inv.reshape(-1)
rverts = pts[first].astype(np.float32)
rtris = inv[np.array(tris_l)]
# outward winding
ta, tb, tc = rverts[rtris[:, 0]], rverts[rtris[:, 1]], rverts[rtris[:, 2]]
flip = np.einsum("ij,ij->i", np.cross(tb - ta, tc - ta), ta + tb + tc) < 0.0
rtris[flip] = rtris[flip][:, ::-1]
NR = len(rverts)
# trilinear embedding (extrapolates into the particle-radius shell)
uu = (rverts + HL) / SPACING
cell = np.clip(np.floor(uu), 0, NC - 1).astype(np.int32)
frac = (uu - cell).astype(np.float32)
emb_t = np.zeros((NR, 8), np.int32)
for cidx in range(8):
    dx, dy, dz = cidx & 1, (cidx >> 1) & 1, (cidx >> 2) & 1
    emb_t[:, cidx] = lid(cell[:, 0] + dx, cell[:, 1] + dy, cell[:, 2] + dz)

# --- the wall ---------------------------------------------------------------

rng = np.random.default_rng(7)
blocks = []                  # (centre, yaw, colour)
pitch = BLOCK + GAP
for r in range(ROWS):
    n_row = 7 if r % 2 == 0 else 6
    y = R_P + HL + r * (BLOCK + 0.002)
    for i in range(n_row):
        x = (i - 0.5 * (n_row - 1)) * pitch
        for z in (-0.5 * pitch, 0.5 * pitch):
            blocks.append(((x, y, z), float(rng.uniform(-0.02, 0.02)),
                           int(rng.integers(len(PALETTE)))))
blocks.sort(key=lambda b: b[2])
NB = len(blocks)
N = NB * NP

x0 = np.zeros((N, 3), np.float32)
body_np = np.repeat(np.arange(NB, dtype=np.int32), NP)
for b, (c, yaw, _) in enumerate(blocks):
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    x0[b * NP:(b + 1) * NP] = lat @ rot.T + np.array(c, np.float32)

# global constraints ordered by colour, then block
ci_l, cj_l, cf_l, cstart, ccount = [], [], [], [], []
off = (np.arange(NB, dtype=np.int32) * NP)[:, None]
pos = 0
for c in range(n_colors):
    sel = tcol == c
    gi = (tci[sel][None, :] + off).ravel()
    gj = (tcj[sel][None, :] + off).ravel()
    ci_l.append(gi)
    cf_l.append(np.tile(tfac[sel], NB))
    cj_l.append(gj)
    cstart.append(pos)
    ccount.append(len(gi))
    pos += len(gi)
ci_np = np.concatenate(ci_l)
cj_np = np.concatenate(cj_l)
cf_np = np.concatenate(cf_l).astype(np.float32)
rest_np = np.linalg.norm(x0[ci_np] - x0[cj_np], axis=1).astype(np.float32)
NCON = len(ci_np)

emb_np = (emb_t[None, :, :] + (np.arange(NB) * NP)[:, None, None]).reshape(-1, 8)
embw_np = np.tile(frac, (NB, 1))
rtris_g = (rtris[None, :, :] + (np.arange(NB) * NR)[:, None, None]).reshape(-1)
NRV = NB * NR
NTRI = NB * len(rtris)

# --- warp -------------------------------------------------------------------


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
            damp: wp.array(dtype=float), dt: float):
    i = wp.tid()
    p = x[i]
    v = (p - prev[i]) * (1.0 - damp[0])
    prev[i] = p
    x[i] = p + v + GRAV * dt * dt


@wp.kernel
def solve_color(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
                ci: wp.array(dtype=int),
                cj: wp.array(dtype=int), rest: wp.array(dtype=float),
                fac: wp.array(dtype=float),
                stiff: wp.array(dtype=float), kd: float, start: int):
    k = start + wp.tid()
    i = ci[k]
    j = cj[k]
    pi = x[i]
    pj = x[j]
    d = pj - pi
    l = wp.length(d)
    if l > 1.0e-9:
        n = d / l
        # elastic pull + damping of the strut's stretch rate (jelly, not rubber)
        vr = wp.dot((pj - prev[j]) - (pi - prev[i]), n)
        corr = n * (0.5 * fac[k] * (stiff[0] * (l - rest[k]) + kd * vr))
        x[i] = pi + corr
        x[j] = pj - corr


@wp.kernel
def contact_pp(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
               prev: wp.array(dtype=wp.vec3), body: wp.array(dtype=int),
               delta: wp.array(dtype=wp.vec3), dist: float, qrad: float, mu: float):
    i = wp.tid()
    p = x[i]
    b = body[i]
    acc = wp.vec3(0.0, 0.0, 0.0)
    cnt = int(0)
    q = wp.hash_grid_query(grid, p, qrad)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if body[j] != b:
            pj = x[j]
            e = p - pj
            l = wp.length(e)
            if l < dist and l > 1.0e-7:
                n = e / l
                pen = dist - l
                c = n * (0.5 * pen)
                # Coulomb friction: cancel relative slip up to mu * penetration
                rv = (p - prev[i]) - (pj - prev[j])
                rt = rv - n * wp.dot(rv, n)
                tl = wp.length(rt)
                if tl > 1.0e-9:
                    c = c - rt * (0.5 * wp.min(tl, mu * pen) / tl)
                acc = acc + c
                cnt += 1
    if cnt > 0:
        dd = acc / float(cnt)
        dl = wp.length(dd)
        if dl > MAX_PUSH:
            dd = dd * (MAX_PUSH / dl)
        delta[i] = dd
    else:
        delta[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_bounds(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
                 delta: wp.array(dtype=wp.vec3), ball: wp.array(dtype=wp.vec3),
                 rb: float, rp: float, mu_b: float, mu_f: float):
    i = wp.tid()
    p = x[i] + delta[i]
    c = ball[0]
    e = p - c
    l = wp.length(e)
    rr = rb + rp
    if l < rr:
        n = e / wp.max(l, 1.0e-6)
        p = c + n * rr
        rv = (p - prev[i]) - ball[1]
        rt = rv - n * wp.dot(rv, n)
        p = p - rt * mu_b
    if p[1] < rp:
        p = wp.vec3(p[0], rp, p[2])
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * mu_f
    x[i] = p


@wp.kernel
def embed(x: wp.array(dtype=wp.vec3), emb: wp.array2d(dtype=int),
          w: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    f = w[k]
    acc = wp.vec3(0.0, 0.0, 0.0)
    for c in range(8):
        wx = wp.where((c & 1) == 1, f[0], 1.0 - f[0])
        wy = wp.where(((c >> 1) & 1) == 1, f[1], 1.0 - f[1])
        wz = wp.where(((c >> 2) & 1) == 1, f[2], 1.0 - f[2])
        acc = acc + x[emb[k, c]] * (wx * wy * wz)
    out[k] = acc


wp.init()
device = wp.get_preferred_device()
print(f"jelly wreck: {NB} blocks, {N:,} particles, {NCON:,} constraints in {n_colors} colours, "
      f"{NTRI:,} render triangles on {device}", flush=True)

x = wp.array(x0, dtype=wp.vec3, device=device)
prev = wp.array(x0, dtype=wp.vec3, device=device)
delta = wp.zeros(N, dtype=wp.vec3, device=device)
body = wp.array(body_np, dtype=int, device=device)
ci = wp.array(ci_np, dtype=int, device=device)
cj = wp.array(cj_np, dtype=int, device=device)
rest = wp.array(rest_np, dtype=float, device=device)
cfac = wp.array(cf_np, dtype=float, device=device)
stiff = wp.array(np.float32([STIFF]), dtype=float, device=device)
damp = wp.array(np.float32([DAMP_SETTLE]), dtype=float, device=device)
ball = wp.array(np.zeros((2, 3), np.float32), dtype=wp.vec3, device=device)
emb = wp.array(emb_np, dtype=int, device=device)
embw = wp.array(embw_np, dtype=wp.vec3, device=device)
rpos = wp.zeros(NRV, dtype=wp.vec3, device=device)
rnrm = wp.zeros(NRV, dtype=wp.vec3, device=device)
rtri = wp.array(rtris_g.astype(np.int32), dtype=int, device=device)
soup_pos = wp.zeros(NTRI * 3, dtype=wp.vec3, device=device)
soup_nrm = wp.zeros(NTRI * 3, dtype=wp.vec3, device=device)
grid = wp.HashGrid(128, 128, 128, device=device)


def substep_launches():
    wp.launch(predict, dim=N, device=device, inputs=[x, prev, damp, SUB_DT])
    for c in range(n_colors):
        wp.launch(solve_color, dim=ccount[c], device=device,
                  inputs=[x, prev, ci, cj, rest, cfac, stiff, KD, cstart[c]])
    wp.launch(contact_pp, dim=N, device=device,
              inputs=[grid.id, x, prev, body, delta, 2.0 * R_P, QRAD, MU_PP])
    wp.launch(apply_bounds, dim=N, device=device,
              inputs=[x, prev, delta, ball, R_B, R_P, MU_BALL, MU_FLOOR])


grid.build(x, QRAD)
graph = None
if device.is_cuda:
    with wp.ScopedCapture(device) as cap:
        substep_launches()
    graph = cap.graph


# --- pendulum (CPU, heavy: the jelly does not slow it) ----------------------

class Pendulum:
    def __init__(self):
        self.phi = PHI0
        self.om = 0.0
        self.active = False

    def pos(self, phi=None):
        phi = self.phi if phi is None else phi
        return PIVOT + L_PEND * np.array([0.0, -math.cos(phi), math.sin(phi)])

    def step(self, dt):
        if not self.active:
            return np.zeros(3)
        p0 = self.pos()
        self.om += dt * (-(9.81 / L_PEND) * math.sin(self.phi) - PEND_DAMP * self.om)
        self.phi += dt * self.om
        return self.pos() - p0


pend = Pendulum()
sim_t = 0.0
ball_host = np.zeros((2, 3), np.float32)


def substeps(n):
    global sim_t
    for _ in range(n):
        d = pend.step(SUB_DT)
        ball_host[0] = pend.pos()
        ball_host[1] = d
        ball.assign(ball_host)
        grid.build(x, QRAD)
        if graph is not None:
            wp.capture_launch(graph)
        else:
            substep_launches()
        if pend.active:
            sim_t += SUB_DT


def refresh_geometry():
    wp.launch(embed, dim=NRV, device=device, inputs=[x, emb, embw, rpos])
    rnrm.zero_()
    wp.launch(accum_normals, dim=NTRI, device=device, inputs=[rpos, rtri, rnrm])
    wp.launch(scatter_soup, dim=NTRI * 3, device=device,
              inputs=[rpos, rnrm, rtri, soup_pos, soup_nrm])
    soup.publish()
    # ball + chain follow the pendulum
    bp = pend.pos()
    ball_mesh.position.set(*bp)
    ball_mesh.rotation.x = pend.phi
    for k, link in enumerate(links):
        d = (k + 0.5) * LINK_PITCH
        lp = PIVOT + d * np.array([0.0, -math.cos(pend.phi), math.sin(pend.phi)])
        link.position.set(*lp)
        link.rotation.set(-pend.phi, 0.5 * math.pi * (k % 2), 0.0)


# --- scene ------------------------------------------------------------------

headless = VIDEO > 0
canvas = tp.Canvas("threepp x warp - jelly wreck", width=W, height=H,
                   antialiasing=4, vsync=False, headless=headless)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = cli_arg("--exposure", 1.1, float)

scene = tp.Scene()
SUN_DIR = np.array([0.55, 0.78, 0.30])
SUN_DIR /= np.linalg.norm(SUN_DIR)


def studio_env(w=1024, hgt=512):
    """Procedural studio: dark cyclorama, three big softboxes."""
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
    col += box((-0.379, 0.53, -0.758), 0.9, 0.2, 5.0, (0.85, 0.9, 1.0))  # backlight: just above frame, pools on the floor
    col += box((-0.5, 0.35, 0.8), 0.35, 0.25, 2.2, (1.0, 1.0, 1.0))    # fill
    col += box((0.0, 1.0, 0.0), 0.5, 0.5, 2.5, (1.0, 1.0, 1.0))        # top
    out = np.ones((hgt, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


env = studio_env()
scene.environment = env
scene.background = tp.Color(0.016, 0.017, 0.020)

sun = tp.DirectionalLight(0xfff2e0, 2.2)
sun.position.set(*(SUN_DIR * 12.0))
sun.cast_shadow = True
scene.add(sun)

floor_mat = standard_material(tp.Color(0.030, 0.032, 0.036), 0.16, 0.0)
floor = tp.Mesh(tp.PlaneGeometry(600, 600), floor_mat)
floor.rotation.x = -math.pi / 2
floor.receive_shadow = True
scene.add(floor)

# jelly: one mesh per colour; blocks are sorted by colour so each is a range
groups = []
tri_per_block = len(rtris)
init_soup = rverts[rtris.reshape(-1)]
for ccol in range(len(PALETTE)):
    bs = [b for b in range(NB) if blocks[b][2] == ccol]
    if not bs:
        continue
    a, b = bs[0] * tri_per_block * 3, (bs[-1] + 1) * tri_per_block * 3
    geom = tp.BufferGeometry()
    geom.set_attribute("position", np.zeros((b - a, 3), np.float32))
    geom.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (b - a, 1)))
    if OPAQUE:
        mat = tp.MeshPhysicalMaterial()
        mat.color = tp.Color(*PALETTE[ccol])
        mat.roughness = 0.18
        mat.clearcoat = 1.0
        mat.clearcoat_roughness = 0.05
    else:
        mat = tp.MeshPhysicalMaterial()
        mat.color = tp.Color(*PALETTE[ccol])
        mat.roughness = cli_arg("--jelly-rough", 0.06, float)
        mat.metalness = 0.0
        mat.transmission = cli_arg("--transmission", 1.0, float)
        mat.ior = 1.35
        mat.transparent = False
    mesh = tp.Mesh(geom, mat)
    mesh.cast_shadow = True
    mesh.receive_shadow = True
    mesh.frustum_culled = False
    scene.add(mesh)
    groups.append((mesh, a, b - a))
# The soup goes straight from Warp into the renderer's vertex buffers (host copy
# only as a fallback, or with --no-interop).
soup = SoupInterop(renderer, device, soup_pos, soup_nrm, groups,
                   interop="--no-interop" not in sys.argv)

chrome = standard_material(tp.Color(0.96, 0.96, 0.97), 0.04, 1.0)
ball_mesh = tp.Mesh(tp.SphereGeometry(R_B, 96, 64), chrome)
ball_mesh.cast_shadow = True
scene.add(ball_mesh)
LINK_PITCH = 0.105
steel = standard_material(tp.Color(0.75, 0.76, 0.78), 0.18, 1.0)
link_geom = tp.TorusGeometry(0.058, 0.016, 12, 32)
links = []
for k in range(int((L_PEND - R_B + 0.02) / LINK_PITCH)):
    lm = tp.Mesh(link_geom, steel)
    lm.scale.set(0.62, 1.0, 1.0)
    lm.cast_shadow = True
    scene.add(lm)
    links.append(lm)

camera = tp.PerspectiveCamera(38, W / H, 0.05, 200)


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


FILM = max(VIDEO, 1e-6)


def camera_at(tf):
    u = smooth(tf / FILM)
    az = math.radians(52.0 - 30.0 * u)
    r = 8.2 - 1.6 * u
    hgt = 1.9 + 0.6 * u
    tgt = np.array([0.0, 1.05 - 0.40 * u, 0.2 + 0.5 * u])
    camera.position.set(tgt[0] + r * math.sin(az), hgt, tgt[2] + r * math.cos(az))
    camera.look_at(*tgt)


# --- settle the wall before the ball is released ----------------------------

t0 = time.perf_counter()
substeps(int(0.8 / SUB_DT))
damp.assign(np.float32([DAMP]))
substeps(int(0.4 / SUB_DT))
wp.synchronize_device(device)
xs = x.numpy()
print(f"settled in {time.perf_counter() - t0:.1f}s; wall top y={xs[:, 1].max():.3f}, "
      f"nan={np.isnan(xs).any()}", flush=True)
pend.active = True

# time of first contact (ball surface reaches the wall's back face)
_p = Pendulum()
_p.active = True
T_HIT, _t = None, 0.0
while T_HIT is None and _t < 3.0:
    _p.step(SUB_DT)
    _t += SUB_DT
    if _p.pos()[2] + R_B > -pitch - R_P:
        T_HIT = _t
print(f"ball reaches the wall at t={T_HIT:.3f}s", flush=True)
SLOW = 0.125


def speed(ts):
    """Film speed as a function of sim time: 1x, ease to 1/8x through the hit."""
    a0, a1 = T_HIT - 0.14, T_HIT - 0.03
    b0, b1 = T_HIT + 0.30, T_HIT + 0.46
    if ts < a0 or ts > b1:
        return 1.0
    if ts < a1:
        return 1.0 + (SLOW - 1.0) * smooth((ts - a0) / (a1 - a0))
    if ts < b0:
        return SLOW
    return SLOW + (1.0 - SLOW) * smooth((ts - b0) / (b1 - b0))


if not headless:
    pend.active = True

    def step_frame():
        substeps(SUBS_PER_FRAME)
        refresh_geometry()

    refresh_geometry()
    camera_at(0.0)
    orbit_loop(canvas, renderer, scene, camera, step_frame, target=(0.0, 1.0, 0.0))
    sys.exit(0)

# --- film -------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
from PIL import Image, ImageDraw  # noqa: E402

font = load_font(max(14, int(H * 0.021)))
font_small = load_font(max(12, int(H * 0.018)))
CAPTION = (f"threepp + NVIDIA Warp  \u00b7  {NB} jelly blocks, {N:,} particles, "
           f"{NCON // 1000}k constraints  \u00b7  GPU XPBD, 1440 substeps/s")


def overlay(rgb, spd):
    img = Image.fromarray(rgb)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    m = int(H * 0.03)
    sh = max(1, H // 540)
    pos = (m, H - m - int(H * 0.021))
    dr.text((pos[0] + sh, pos[1] + sh), CAPTION, font=font, fill=(0, 0, 0, 150))
    dr.text(pos, CAPTION, font=font, fill=(245, 245, 248, 215))
    a = int(210 * max(0.0, min(1.0, (1.0 - spd) / (1.0 - SLOW))))
    if a > 4:
        txt = "slow motion  1/8\u00d7"
        tw = dr.textlength(txt, font=font_small)
        dr.text((W - m - tw + sh, m + sh), txt, font=font_small, fill=(0, 0, 0, int(a * 0.6)))
        dr.text((W - m - tw, m), txt, font=font_small, fill=(245, 245, 248, a))
    return np.asarray(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


total = int(round(VIDEO * 60))
STILLS = [int(total * float(f)) for f in cli_arg("--stills", "0.09,0.30,0.52,0.92", str).split(",")]
SHEET = [int(round(i * (total - 1) / 8)) for i in range(9)]
sheet_imgs = {}
mp4 = os.path.join(OUT_DIR, f"{TAG}.mp4")
enc = Encoder(mp4, W, H, 60, crf=17, preset="slow")
refresh_geometry()
camera_at(0.0)
for _ in range(30):
    renderer.render(scene, camera)
acc = 0.0
tstart = time.perf_counter()
for k in range(total):
    spd = speed(sim_t)
    acc += spd * SUBS_PER_FRAME
    n = int(acc)
    acc -= n
    substeps(n)
    refresh_geometry()
    camera_at(k / 60.0)
    renderer.render(scene, camera)
    rgb = overlay(renderer.read_pixels(), spd)
    enc.send(rgb)
    if k in STILLS:
        Image.fromarray(rgb).save(os.path.join(OUT_DIR, f"{TAG}_still_{STILLS.index(k)}_f{k:04d}.png"))
    if k in SHEET:
        sheet_imgs[k] = Image.fromarray(rgb).resize((640, 360), Image.LANCZOS)
    if k % 60 == 0:
        xs = x.numpy()
        print(f"  frame {k}/{total} sim_t={sim_t:.3f} speed={spd:.3f} "
              f"ymin={xs[:, 1].min():.3f} ymax={xs[:, 1].max():.3f} nan={np.isnan(xs).any()} "
              f"({time.perf_counter() - tstart:.0f}s)", flush=True)
rc = enc.close()
sheet = Image.new("RGB", (640 * 3, 360 * 3))
for n_, k in enumerate(SHEET):
    if k in sheet_imgs:
        sheet.paste(sheet_imgs[k], ((n_ % 3) * 640, (n_ // 3) * 360))
sheet.save(os.path.join(OUT_DIR, f"{TAG}_contact_sheet.png"))
print(f"rendered {total} frames in {time.perf_counter() - tstart:.0f}s -> {mp4} (rc={rc})", flush=True)

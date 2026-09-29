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
    python warp_jelly_wreck.py --video 10 --no-sensors     # the plain film, no sensor mosaic
    python warp_jelly_wreck.py --video 10 --no-sensors --clean   # ... and no caption or badge (for a film that titles it)

The film is a 3x3 sensor mosaic by default: the RGB view in the top-left 2x2,
and depth, instance ids, optical flow, event camera and lidar tiles, all from
the same simulated instant. --size is the mosaic size; the RGB view renders
at 2/3 of it and the depth and id tiles are that same render's AOVs.
"""
import colorsys
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from threepp.utils import fetch_file
from demo_common import read_radiance_hdr
from warp_common import (Encoder, SoupInterop, accum_normals, cli_arg, load_font,
                         orbit_loop, parse_size, scatter_soup, standard_material)

VIDEO = cli_arg("--video", 0.0, float)
W, H = parse_size(cli_arg("--size", "1920x1080", str))
OUT_DIR = cli_arg("--out-dir", r"C:\dev\_softbody_films\jelly_wreck", str)
TAG = cli_arg("--tag", "jelly_wreck", str)
OPAQUE = "--opaque" in sys.argv
SENSORS = VIDEO > 0 and "--no-sensors" not in sys.argv
CLEAN = "--clean" in sys.argv          # the picture alone: no caption, no slow-motion badge
TW, TH = W // 3, H // 3                  # sensor tile
RW, RH = (2 * TW, 2 * TH) if SENSORS else (W, H)   # render size (the RGB view)

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
STIFF = cli_arg("--stiff", 0.008, float)      # jelly: GS stiffness per substep
DAMP = cli_arg("--damp", 0.00012, float)     # per substep (air + internal loss)
DAMP_SETTLE = 0.02
MIRROR = cli_arg("--mirror", 0.30, float)   # centre-strut stiffness factor
KD = cli_arg("--kd", 0.0007, float)      # relative-velocity damping along struts (low: wobble rings on)
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

# The set: --look studio | stage | sunset (see "looks" below). HDRIs are Poly
# Haven (CC0), downloaded once into ~/.cache/threepp/hdri.
LOOK = cli_arg("--look", "stage", str)
HDRI_RES = cli_arg("--hdri-res", "2k", str)
HDRI_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "threepp", "hdri")
CYC_EL = (15.0, 30.0)                  # deg: painted-out walls below, the real studio above
CYC_R = (11.0, 5.0, 18.0)              # cyc: floor radius, cove radius, wall height (m)
CYC_ARC = (200.0, 140.0)               # deg about +y (atan2(z, x)): sweep centre, half-width

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
    rig.rotation.x = -pend.phi      # ball + chain follow the pendulum as one transform


# --- scene ------------------------------------------------------------------

headless = VIDEO > 0
canvas = tp.Canvas("threepp x warp - jelly wreck", width=RW, height=RH,
                   antialiasing=4, vsync=False, headless=headless)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = cli_arg("--exposure", {"studio": 1.1, "stage": 1.0, "sunset": 0.55}.get(LOOK, 1.0), float)

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
    return col


def fetch_hdri(name):
    """A Poly Haven HDRI (CC0) as an (H, W, 3) float array, row 0 = nadir (the
    float_texture convention). Fetched once into ~/.cache/threepp/hdri; None
    when offline, and the look falls back to the procedural studio."""
    try:
        path = fetch_file(f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{HDRI_RES}/"
                          f"{name}_{HDRI_RES}.hdr", HDRI_CACHE)
    except Exception as e:
        print(f"HDRI {name} unavailable ({e}); using the procedural studio", flush=True)
        return None
    return read_radiance_hdr(path)[::-1]


def paint_walls(rgb, tone, gain):
    """The Vulkan renderer draws scene.environment as the sky, so what the camera
    sees above the floor IS the HDRI. Paint everything below CYC_EL[0] of
    elevation to one flat tone, blending back to the real studio (dark ceiling,
    the key octabox) by CYC_EL[1]: the camera never looks that high, the chrome
    and the jelly reflect it."""
    hgt = rgb.shape[0]
    elev = np.degrees(((np.arange(hgt) + 0.5) / hgt - 0.5) * math.pi)
    k = np.clip((elev - CYC_EL[0]) / (CYC_EL[1] - CYC_EL[0]), 0.0, 1.0)
    k = (k * k * (3.0 - 2.0 * k))[:, None, None]
    return np.float32(tone) * (1.0 - k) + rgb * gain * k


def add_strip(rgb, az_deg, el_deg, half_az, half_el, inten, tint=(1.0, 1.0, 1.0)):
    """Add a soft-edged rectangular emitter to an equirect (row 0 = nadir) at
    azimuth az_deg (atan2(z, x)) and elevation el_deg; half sizes in degrees."""
    hgt, w = rgb.shape[:2]
    el = ((np.arange(hgt, dtype=np.float32) + 0.5) / hgt - 0.5) * 180.0
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 360.0
    da = (az - az_deg + 180.0) % 360.0 - 180.0
    fa = np.clip((half_az - np.abs(da)) / 2.0, 0.0, 1.0)[None, :]
    fe = np.clip((half_el - np.abs(el - el_deg)) / 1.5, 0.0, 1.0)[:, None]
    return rgb + (fa * fe)[..., None] * np.float32(tint) * np.float32(inten)


def aim_sun(rgb, az_deg):
    """Spin an equirect about +y so its brightest pixel sits at azimuth az_deg
    (atan2(z, x), the renderer's u = 0.5 + atan2(z, x) / 2pi). Returns the
    spun image and the sun's elevation in degrees."""
    hgt, w = rgb.shape[:2]
    j, i = np.unravel_index(rgb.mean(-1).argmax(), rgb.shape[:2])
    az0 = ((i + 0.5) / w - 0.5) * 360.0
    el = ((j + 0.5) / hgt - 0.5) * 180.0
    return np.roll(rgb, int(round((az_deg - az0) / 360.0 * w)), axis=1), el


def dir_from(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])


def spot(color, intensity, pos, target, angle, penumbra=0.5):
    s = tp.SpotLight(tp.Color(*color), intensity, 0.0, angle, penumbra, 2.0)
    s.position.set(*pos)
    tgt = tp.Group()
    tgt.position.set(*target)
    scene.add(tgt)
    s.set_target(tgt)
    s.cast_shadow = True
    scene.add(s)
    return s


# --- looks --------------------------------------------------------------------
#   studio  a light seamless cyc under a photographed studio: clean, flat
#   stage   a black set in haze: one hard key spot from above, two coloured rim
#           spots from behind, so the beams are visible and the glass glows
#   sunset  a real sky with the sun low BEHIND the wall: the jelly is backlit
#           and every block throws a long coloured shadow toward the camera
WALL_C = (0.0, 1.1, 0.0)
if LOOK == "studio":
    env_rgb = fetch_hdri("studio_small_09")
    env_rgb = studio_env() if env_rgb is None else paint_walls(env_rgb, (0.30, 0.30, 0.31), 0.45)
    sun = tp.DirectionalLight(0xfff2e0, 2.6)
    sun.position.set(*(SUN_DIR * 12.0))
    sun.cast_shadow = True
    scene.add(sun)
    set_mat = standard_material(tp.Color(0.50, 0.50, 0.51), 0.55, 0.0)
    set_geo = "cyc"
elif LOOK == "stage":
    env_rgb = fetch_hdri("studio_small_09")
    env_rgb = studio_env() * 0.1 if env_rgb is None else paint_walls(env_rgb, (0.003, 0.003, 0.004), 0.05)
    # Strip softboxes the camera never sees (above 35 deg, behind and beside the
    # camera, which sits at azimuth 38-68 deg): the chrome and the glass faces
    # reflect them as clean bright bars.
    sb = cli_arg("--softbox", 2.2, float)
    env_rgb = add_strip(env_rgb, 50.0, 40.0, 34.0, 3.5, 2.0 * sb)                    # front, behind camera
    env_rgb = add_strip(env_rgb, 50.0, 72.0, 60.0, 5.0, 0.7 * sb)                # overhead
    env_rgb = add_strip(env_rgb, 140.0, 40.0, 10.0, 12.0, 0.5 * sb, (0.8, 0.9, 1.0))   # camera-left
    env_rgb = add_strip(env_rgb, -40.0, 40.0, 10.0, 12.0, 0.5 * sb, (1.0, 0.9, 0.85))  # camera-right
    spot((1.0, 0.93, 0.84), cli_arg("--key", 420.0, float), (2.5, 9.5, 3.5), WALL_C, 0.40, 0.55)
    # Rims from the SIDES, a little behind: from straight behind, the ball hanging
    # in the beam throws a shadow shaft through the haze into the lens.
    spot((0.25, 0.75, 1.0), cli_arg("--rim", 300.0, float), (-7.5, 6.0, -1.5), WALL_C, 0.35, 0.6)
    spot((1.0, 0.35, 0.65), cli_arg("--rim", 300.0, float), (7.5, 6.5, -1.0), WALL_C, 0.35, 0.6)
    # The background light: glass only glows when something bright is BEHIND
    # it, so a soft spot throws a pool onto the cove behind the set, on the
    # camera's line through the wall. The pool is the glow; the sweep around it
    # stays black because nothing else lights it.
    # Aimed low so the pool runs down the cove onto the floor behind the wall:
    # the whole wall, bottom rows included, is backed by light.
    spot((0.95, 0.97, 1.0), cli_arg("--bg", 4200.0, float), (3.0, 11.0, 5.0),
         (-9.5, 0.6, -7.5), 0.40, 0.9)
    scene.set_fog_exp2(tp.Color(1.0, 1.0, 1.0), cli_arg("--haze", 0.019, float))
    renderer.fog_anisotropy = 0.45
    renderer.bloom_intensity = cli_arg("--bloom", 0.12, float)
    renderer.bloom_threshold = 2.0
    set_mat = (standard_material(tp.Color(0.02, 0.02, 0.022), 0.08, 0.0),   # floor: glossy black mirror
               standard_material(tp.Color(0.30, 0.30, 0.31), 0.9, 0.0))     # sweep: matte, lit only by the bg spot
    set_geo = "cyc"
else:   # sunset
    SUN_AZ = cli_arg("--sun-az", 222.0, float)     # opposite the camera: backlit
    env_rgb = fetch_hdri("kloppenheim_06_puresky")
    if env_rgb is None:
        env_rgb, sun_el = studio_env(), 12.0
    else:
        env_rgb, sun_el = aim_sun(env_rgb, SUN_AZ)
    sun = tp.DirectionalLight(tp.Color(1.0, 0.72, 0.46), cli_arg("--sun", 4.0, float))
    sun.position.set(*(dir_from(SUN_AZ, max(sun_el, cli_arg("--sun-el", 9.0, float))) * 30.0))
    sun.cast_shadow = True
    scene.add(sun)
    scene.set_fog_exp2(tp.Color(1.0, 0.9, 0.8), cli_arg("--haze", 0.002, float))
    renderer.fog_anisotropy = 0.7
    set_mat = standard_material(tp.Color(0.20, 0.17, 0.14), 0.4, 0.0)
    set_geo = "plane"
scene.environment = tp.float_texture(np.ascontiguousarray(env_rgb, np.float32))


def lathe(prof, th0, th1, seg):
    """Revolve a (r, y, n_r, n_y) profile about +y over [th0, th1] radians."""
    prof = np.float32(prof)
    th = np.linspace(th0, th1, seg + 1, dtype=np.float32)
    c, s_ = np.cos(th)[:, None], np.sin(th)[:, None]
    pos = np.stack([c * prof[:, 0], np.tile(prof[:, 1], (seg + 1, 1)), s_ * prof[:, 0]], -1)
    nrm = np.stack([c * prof[:, 2], np.tile(prof[:, 3], (seg + 1, 1)), s_ * prof[:, 2]], -1)
    n_p = len(prof)
    i, j = np.meshgrid(np.arange(seg), np.arange(n_p - 1), indexing="ij")
    a0 = i * n_p + j
    a1, b0 = a0 + 1, a0 + n_p
    b1 = b0 + 1
    idx = np.stack([np.stack([a0, b0, a1], -1), np.stack([a1, b0, b1], -1)], -2)   # faces inward/up
    geo = tp.BufferGeometry()
    geo.set_attribute("position", pos.reshape(-1, 3).astype(np.float32))
    geo.set_attribute("normal", nrm.reshape(-1, 3).astype(np.float32))
    geo.set_index(idx.reshape(-1).astype(np.uint32))
    return geo


def cyclorama(r_floor=CYC_R[0], r_cove=CYC_R[1], height=CYC_R[2]):
    """A photo-studio sweep: a floor disc, and behind the set a quarter-circle
    cove of radius r_cove rising into a vertical wall `height` tall.

    There is no horizon because there is no edge: what is behind the set is
    the same lit surface curving up out of the floor. The sweep is an ARC
    centred opposite the camera, not a ring -- a wall on the sun's side would
    shadow the whole set -- and it is open at the top, so the studio HDRI still
    lights the set and the chrome sees its ceiling and octabox."""
    disc = [(0.0, 0.0, 0.0, 1.0)] + [(r, 0.0, 0.0, 1.0) for r in np.linspace(1.0, r_floor, 12)]
    sweep = [(r_floor, 0.0, 0.0, 1.0)]
    for a in np.linspace(0.0, 0.5 * math.pi, 24)[1:]:       # cove centre (r_floor, r_cove)
        sweep.append((r_floor + r_cove * math.sin(a), r_cove * (1.0 - math.cos(a)),
                      -math.sin(a), math.cos(a)))
    sweep += [(r_floor + r_cove, y, -1.0, 0.0) for y in np.linspace(r_cove, height, 8)[1:]]
    mid, half = math.radians(CYC_ARC[0]), math.radians(CYC_ARC[1])
    return lathe(disc, 0.0, 2.0 * math.pi, 160), lathe(sweep, mid - half, mid + half, 120)


if set_geo == "cyc":
    set_geos = cyclorama()
else:
    set_geos = [tp.PlaneGeometry(600, 600)]
    set_geos[0].rotate_x(-math.pi / 2)
if not isinstance(set_mat, tuple):
    set_mat = (set_mat, set_mat)
set_meshes = []
for g, m in zip(set_geos, set_mat):
    floor = tp.Mesh(g, m)
    floor.receive_shadow = True
    scene.add(floor)
    set_meshes.append(floor)

# jelly: one mesh per colour; blocks are sorted by colour so each is a range.
# Sensor mode draws one mesh per block (same materials) so every block gets
# its own instance id.
groups = []
tri_per_block = len(rtris)
init_soup = rverts[rtris.reshape(-1)]


def jelly_material(ccol):
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
    return mat


jelly_mats = [jelly_material(ccol) for ccol in range(len(PALETTE))]
if SENSORS:
    ranges = [(b, b + 1, blocks[b][2]) for b in range(NB)]
else:
    ranges = []
    for ccol in range(len(PALETTE)):
        bs = [b for b in range(NB) if blocks[b][2] == ccol]
        if bs:
            ranges.append((bs[0], bs[-1] + 1, ccol))
for b0, b1, ccol in ranges:
    a, b = b0 * tri_per_block * 3, b1 * tri_per_block * 3
    geom = tp.BufferGeometry()
    geom.set_attribute("position", np.zeros((b - a, 3), np.float32))
    geom.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (b - a, 1)))
    mesh = tp.Mesh(geom, jelly_mats[ccol])
    mesh.cast_shadow = True
    mesh.receive_shadow = True
    mesh.frustum_culled = False
    scene.add(mesh)
    groups.append((mesh, a, b - a))
# The soup goes straight from Warp into the renderer's vertex buffers (host copy
# only as a fallback, or with --no-interop).
soup = SoupInterop(renderer, device, soup_pos, soup_nrm, groups,
                   interop="--no-interop" not in sys.argv)

chrome = standard_material(tp.Color(0.96, 0.96, 0.97), 0.03, 1.0)
LINK_PITCH = 0.105
steel = standard_material(tp.Color(0.75, 0.76, 0.78), 0.18, 1.0)


def chain_geometry(n, pitch, R=0.058, r=0.016, seg_t=32, seg_r=12):
    """n alternating torus links hanging down -y from the origin, merged into ONE
    geometry: the whole chain and the ball ride a single pivot transform, so they
    cannot be drawn at different instants."""
    u = np.linspace(0.0, 2.0 * math.pi, seg_t + 1)[:, None]
    v = np.linspace(0.0, 2.0 * math.pi, seg_r + 1)[None, :]
    ring = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u),
                     np.broadcast_to(r * np.sin(v), (seg_t + 1, seg_r + 1))], -1).reshape(-1, 3)
    nrm = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u),
                    np.broadcast_to(np.sin(v), (seg_t + 1, seg_r + 1))], -1).reshape(-1, 3)
    ring = ring * (0.62, 1.0, 1.0)                        # oval link
    nrm = nrm / (0.62, 1.0, 1.0)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    i, j = np.meshgrid(np.arange(seg_t), np.arange(seg_r), indexing="ij")
    a0 = i * (seg_r + 1) + j
    b0, d0 = a0 + seg_r + 1, a0 + 1
    quad = np.stack([np.stack([a0, b0, d0], -1), np.stack([b0, b0 + 1, d0], -1)], -2).reshape(-1)
    pos_l, nrm_l, idx_l = [], [], []
    for k in range(n):
        ang = 0.5 * math.pi * (k % 2)
        c_, s_ = math.cos(ang), math.sin(ang)
        rot = np.array([[c_, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c_]])
        pos_l.append(ring @ rot.T + (0.0, -(k + 0.5) * pitch, 0.0))
        nrm_l.append(nrm @ rot.T)
        idx_l.append(quad + k * len(ring))
    geo = tp.BufferGeometry()
    geo.set_attribute("position", np.concatenate(pos_l).astype(np.float32))
    geo.set_attribute("normal", np.concatenate(nrm_l).astype(np.float32))
    geo.set_index(np.concatenate(idx_l).astype(np.uint32))
    return geo


# The pendulum rig: pivot group at PIVOT, rotated by -phi about x, carrying the
# ball at (0, -L_PEND, 0) and the chain above it.
rig = tp.Group()
rig.position.set(*PIVOT)
scene.add(rig)
ball_mesh = tp.Mesh(tp.SphereGeometry(R_B, 96, 64), chrome)
ball_mesh.position.set(0.0, -L_PEND, 0.0)
ball_mesh.cast_shadow = True
rig.add(ball_mesh)
chain = tp.Mesh(chain_geometry(int((L_PEND - R_B + 0.02) / LINK_PITCH), LINK_PITCH), steel)
chain.cast_shadow = True
rig.add(chain)

camera = tp.PerspectiveCamera(38, RW / RH, 0.05, 200)

# Instance ids: block b is b + 1 (per block in sensor mode, else per colour
# mesh), then the rig and the set.
ID_BALL, ID_CHAIN, ID_FLOOR, ID_PROP = 100, 101, 102, 104
if SENSORS:
    for n_, (m_, _, _) in enumerate(groups):
        renderer.set_instance_id(m_, n_ + 1)
    renderer.set_instance_id(ball_mesh, ID_BALL)
    renderer.set_instance_id(chain, ID_CHAIN)
    for n_, m_ in enumerate(set_meshes):
        renderer.set_instance_id(m_, ID_FLOOR + n_)


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


FILM = max(VIDEO, 1e-6)


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
# The camera sits on the far side from the ball: first contact is hidden, and at
# 1/8x it is a graze. What reads as THE hit is the ball's centre reaching the
# wall's near face, where it bursts toward the lens -- the shake keys on that.
BURST_Z = cli_arg("--burst-z", 0.4, float)
T_BURST, _t = None, 0.0
_p = Pendulum()
_p.active = True
while T_BURST is None and _t < 3.0:
    _p.step(SUB_DT)
    _t += SUB_DT
    if _p.pos()[2] >= BURST_Z:
        T_BURST = _t
SLOW = cli_arg("--slow", 0.125, float)


def speed(ts):
    """Film speed as a function of sim time: 1x, ease to SLOW through the hit."""
    a0, a1 = T_HIT - 0.14, T_HIT - 0.03
    b0, b1 = T_HIT + 0.34, T_HIT + 0.52
    if ts < a0 or ts > b1:
        return 1.0
    if ts < a1:
        return 1.0 + (SLOW - 1.0) * smooth((ts - a0) / (a1 - a0))
    if ts < b0:
        return SLOW
    return SLOW + (1.0 - SLOW) * smooth((ts - b0) / (b1 - b0))


def film_clock():
    """Film times of first contact, of the burst and of the return to 1x (replays
    the film loop's substep count)."""
    acc, ts, tf_hit, tf_burst = 0.0, 0.0, None, None
    for k in range(3600):
        spd = speed(ts)
        acc += spd * SUBS_PER_FRAME
        n = int(acc)
        acc -= n
        ts += n * SUB_DT
        if tf_hit is None and ts >= T_HIT:
            tf_hit = k / 60.0
        if tf_burst is None and ts >= T_BURST:
            tf_burst = k / 60.0
        if tf_burst is not None and spd >= 1.0:
            return tf_hit, tf_burst, k / 60.0
    return tf_hit, tf_burst, 3600 / 60.0


TF_HIT, TF_BURST, TF_FAST = film_clock()
print(f"film: contact at {TF_HIT:.2f}s, burst at {TF_BURST:.2f}s, back to 1x at {TF_FAST:.2f}s", flush=True)

# Camera keys: (film time, azimuth from +z deg, radius, height, target x, y, z).
CAM_KEYS = [
    (0.0,           50.0, 8.6, 2.40, 0.0, 1.35, -0.40),   # establishing 3/4, ball swinging in behind
    (TF_HIT - 0.2,  40.0, 6.4, 1.50, 0.0, 1.15, -0.10),   # push in
    (TF_HIT + 1.1,  30.0, 5.5, 0.60, 0.0, 1.15, 0.20),    # low hero angle through the burst
    (TF_FAST - 0.7, 22.0, 5.0, 0.50, 0.0, 1.05, 0.40),    # blocks flying at the lens
    (TF_FAST + 0.6, 44.0, 8.2, 2.40, 0.0, 0.85, 1.00),    # pull back as the ball swings out
    (FILM - 2.4,    56.0, 8.0, 3.00, 0.0, 0.60, 1.20),    # orbit reveal of the pile
    (FILM,          64.0, 7.6, 3.40, 0.0, 0.45, 1.30),
]
for _i in range(1, len(CAM_KEYS)):   # short films: keep key times increasing
    if CAM_KEYS[_i][0] < CAM_KEYS[_i - 1][0] + 0.25:
        CAM_KEYS[_i] = (CAM_KEYS[_i - 1][0] + 0.25,) + CAM_KEYS[_i][1:]
_KT = np.array([k[0] for k in CAM_KEYS])
_KV = np.array([k[1:] for k in CAM_KEYS])
SHAKE = cli_arg("--shake", 0.05, float)     # m, camera shake amplitude at first contact
FRAME_TOP = 0.80                             # ball top kept below this fraction of the half-height
cam_stats = {"ball_ndc_max": -9.0}
cam_pose = {}                                # last camera_at: position and target


def cam_keys_at(tf):
    """Cubic Hermite through CAM_KEYS (Catmull-Rom tangents, zero at the ends)."""
    tf = min(max(tf, _KT[0]), _KT[-1])
    i = min(int(np.searchsorted(_KT, tf, side="right")) - 1, len(_KT) - 2)
    t0, t1 = _KT[i], _KT[i + 1]

    def tangent(j):
        if j == 0 or j == len(_KT) - 1:
            return np.zeros(_KV.shape[1])
        return (_KV[j + 1] - _KV[j - 1]) / (_KT[j + 1] - _KT[j - 1])

    h = t1 - t0
    u = (tf - t0) / h
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * _KV[i] + h10 * h * tangent(i) + h01 * _KV[i + 1] + h11 * h * tangent(i + 1)


def camera_at(tf):
    az, r, hgt, tx, ty, tz = cam_keys_at(tf)
    tgt = np.array([tx, ty, tz])
    pos = np.array([tx + r * math.sin(math.radians(az)), hgt, tz + r * math.cos(math.radians(az))])
    # impact shake: damped, deterministic, keyed on the film time of the burst
    dt = tf - TF_BURST
    if dt > 0.0:
        env = SHAKE * math.exp(-dt / 0.22) * min(1.0, dt / 0.03)
        off = env * np.array([math.sin(2 * math.pi * 11.0 * dt),
                              0.8 * math.sin(2 * math.pi * 13.7 * dt + 1.3),
                              0.5 * math.sin(2 * math.pi * 9.1 * dt + 2.1)])
        pos += off
        tgt += 0.4 * off
    # tilt up when the ball's top would leave the frame (soft knee)
    f = tgt - pos
    d = np.linalg.norm(f)
    f /= d
    rt = np.cross(f, (0.0, 1.0, 0.0))
    rt /= np.linalg.norm(rt)
    up = np.cross(rt, f)
    v = pend.pos() + np.array([0.0, 1.1 * R_B, 0.0]) - pos
    half = math.radians(0.5 * camera.fov)
    if v @ f > 0.1:
        ang = math.atan2(v @ up, v @ f)
        x_ = ang - FRAME_TOP * half
        k_ = math.radians(3.0)
        e = 0.0 if x_ <= -k_ else ((x_ + k_) ** 2 / (4 * k_) if x_ < k_ else x_)
        if e > 0.0:
            f = f * math.cos(e) + up * math.sin(e)
            tgt = pos + d * f
            ang -= e
        cam_stats["ball_ndc_max"] = max(cam_stats["ball_ndc_max"], math.tan(ang) / math.tan(half))
    cam_pose["pos"], cam_pose["tgt"] = pos.copy(), tgt.copy()
    camera.position.set(*pos)
    camera.look_at(*tgt)


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
    if CLEAN:                                    # --clean: the picture alone, for a film that titles it
        return rgb
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
        txt = f"slow motion  1/{round(1.0 / SLOW)}\u00d7"
        tw = dr.textlength(txt, font=font_small)
        dr.text((W - m - tw + sh, m + sh), txt, font=font_small, fill=(0, 0, 0, int(a * 0.6)))
        dr.text((W - m - tw, m), txt, font=font_small, fill=(245, 245, 248, a))
    return np.asarray(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


# --- sensor mosaic ----------------------------------------------------------
# 3x3 grid of TW x TH tiles: RGB in the top-left 2x2; depth, instance ids
# (right column), optical flow, event camera, lidar (bottom row). Depth and
# ids are AOVs of the RGB render itself (read_aovs_typed: one render),
# coloured at render resolution and box-filtered 2:1. Flow is computed from
# the sim state and the depth: render() drives 3 GPU frames of the same state
# (flush_frames), so the renderer's own motion AOV reads ~0.

DEPTH_RANGE = (0.5, 14.0)                    # m, near = bright
FLOW_FULL = 28.0                             # px per 1/60 s (at 1280 wide) that saturates
GUTTER = max(2, H // 270)
GUTTER_RGB = (7, 7, 9)
font_title = load_font(max(14, int(H * 0.024)))
font_tile = load_font(max(11, int(H * 0.0145)))
TITLE = "threepp  \u00d7  NVIDIA Warp"
SUBTITLE = (f"{NB} soft bodies, GPU XPBD  |  RGB, depth, instance ids, optical flow, "
            f"event camera, lidar  \u2014  one simulated instant")
CAPTION_S = (f"{NB} jelly blocks, {N:,} particles, {NCON // 1000}k constraints  \u00b7  "
             f"1440 substeps/s  \u00b7  Vulkan ray tracing")


def _poly_lut(coef):
    t = np.linspace(0.0, 1.0, 256)[:, None]
    acc = np.zeros((256, 3))
    for c in coef[::-1]:
        acc = acc * t + np.asarray(c)
    return (np.clip(acc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# inferno, polynomial fit of the matplotlib map (Matt Zucker)
INFERNO = _poly_lut([(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184),
                     (0.1065134194856116, 0.5639564367884091, 3.932712388889277),
                     (11.60249308247187, -3.972853965665698, -15.9423941062914),
                     (-41.70399613139459, 17.43639888205313, 44.35414519872813),
                     (77.162935699427, -33.40235894210092, -81.80730925738993),
                     (-71.31942824499214, 32.62606426397723, 73.20951985803202),
                     (25.13112622477341, -12.24266895238567, -23.07032500287172)])


def _id_lut():
    lut = np.zeros((256, 3), np.uint8)
    for i in range(1, NB + 1):
        r, g_, b = colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, (0.55, 0.78, 0.95)[i % 3],
                                       (1.0, 0.80)[(i // 3) % 2])
        lut[i] = (int(r * 255), int(g_ * 255), int(b * 255))
    lut[ID_BALL] = (170, 174, 184)
    lut[ID_CHAIN] = (112, 116, 126)
    lut[ID_FLOOR] = (34, 36, 42)
    lut[ID_FLOOR + 1] = (22, 23, 28)
    lut[ID_PROP] = (78, 80, 90)
    return lut


ID_LUT = _id_lut()


def _flow_wheel():
    """The Middlebury flow colour wheel (Baker et al. 2011), (55, 3) in [0, 1]."""
    segs = [(15, (1, 0, 0), (1, 1, 0)), (6, (1, 1, 0), (0, 1, 0)), (4, (0, 1, 0), (0, 1, 1)),
            (11, (0, 1, 1), (0, 0, 1)), (13, (0, 0, 1), (1, 0, 1)), (6, (1, 0, 1), (1, 0, 0))]
    rows = []
    for n_, c0, c1 in segs:
        f = np.arange(n_)[:, None] / n_
        rows.append(np.asarray(c0) * (1.0 - f) + np.asarray(c1) * f)
    return np.concatenate(rows).astype(np.float32)


WHEEL = _flow_wheel()


def down2(img):
    """2:1 box filter of an (h, w, 3) uint8 image."""
    h_, w_ = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    v = img[:h_, :w_].reshape(h_ // 2, 2, w_ // 2, 2, 3).astype(np.uint16)
    return (v.sum((1, 3)) // 4).astype(np.uint8)


def depth_tile(d, ids):
    t = np.clip((DEPTH_RANGE[1] - d) / (DEPTH_RANGE[1] - DEPTH_RANGE[0]), 0.0, 1.0) ** 1.6
    img = INFERNO[(t * 255.0).astype(np.uint8)]
    img[ids == 0] = 0
    return down2(img)


def ids_tile(ids):
    img = ID_LUT[np.minimum(ids, 255)]
    edge = np.zeros(ids.shape, bool)             # dark outline where the id changes
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[1:, :] |= ids[1:, :] != ids[:-1, :]
    img[edge] = img[edge] // 4
    return down2(img)


def _basis(pos, tgt):
    f = tgt - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    return r, np.cross(r, f), f


def body_motion(xa, xb, phi_a, phi_b):
    """Per instance id, the map from a surface point now to the same material
    point one film frame ago: P_prev = R[id] @ P + t[id]. Blocks: the best
    rigid fit (Kabsch) of their particle lattice; ball and chain: the pendulum
    angle; the set: identity."""
    R = np.tile(np.eye(3, dtype=np.float32), (256, 1, 1))
    t = np.zeros((256, 3), np.float32)
    A = xa.reshape(NB, NP, 3).astype(np.float64)
    B = xb.reshape(NB, NP, 3).astype(np.float64)
    ca, cb = A.mean(1), B.mean(1)
    U, _, Vt = np.linalg.svd(np.einsum("bni,bnj->bij", A - ca[:, None], B - cb[:, None]))
    D = np.tile(np.eye(3), (NB, 1, 1))
    D[:, 2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rinv = U @ D @ Vt                            # transpose of the fitted rotation
    R[1:NB + 1] = Rinv
    t[1:NB + 1] = ca - np.einsum("bij,bj->bi", Rinv, cb)
    c_, s_ = math.cos(phi_b - phi_a), math.sin(phi_b - phi_a)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c_, -s_], [0.0, s_, c_]])
    for i in (ID_BALL, ID_CHAIN):
        R[i] = rx
        t[i] = PIVOT - rx @ PIVOT
    return R, t


def flow_tile(d, ids, motion, spd, pose, prev_pose):
    """Ground-truth optical flow: each pixel's surface point (view depth) is
    carried back one film frame by its body's motion and projected into the
    previous camera. The bodies' share is divided by the film speed, so slow
    motion shows 1x motion; the camera's share stays as filmed. Middlebury
    wheel: hue = direction, value = magnitude."""
    h_, w_ = d.shape
    fpx = 0.5 * h_ / math.tan(math.radians(0.5 * camera.fov))
    xs = ((np.arange(w_, dtype=np.float32) + 0.5 - 0.5 * w_) / fpx)[::2]
    ys = (-(np.arange(h_, dtype=np.float32) + 0.5 - 0.5 * h_) / fpx)[::2]
    d, ids = d[::2, ::2], np.minimum(ids[::2, ::2], 255)
    r, u, f = (v.astype(np.float32) for v in _basis(*pose))
    r0, u0, f0 = (v.astype(np.float32) for v in _basis(*prev_pose))
    P = pose[0].astype(np.float32) + d[..., None] * (xs[None, :, None] * r + ys[:, None, None] * u + f)
    R, t = motion
    q = P - prev_pose[0].astype(np.float32)
    mov = (ids >= 1) & (ids <= NB) | (ids == ID_BALL) | (ids == ID_CHAIN)
    Pm, im = P[mov], ids[mov]
    q[mov] += (np.einsum("nij,nj->ni", R[im], Pm) + t[im] - Pm) / max(spd, 1e-3)
    pz = np.maximum(q @ f0, 1e-3)
    s = fpx * 1280.0 / RW                        # px at 1280 wide
    mu = s * (xs[None, :] - (q @ r0) / pz)       # current - previous, +x right
    mv = s * ((q @ u0) / pz - ys[:, None])       # +y down
    mag = np.sqrt(mu * mu + mv * mv)
    a = np.arctan2(-mv, -mu) / math.pi           # [-1, 1]
    fk = (a + 1.0) * 0.5 * (len(WHEEL) - 1)
    k0 = fk.astype(np.int32)
    k1 = (k0 + 1) % len(WHEEL)
    fr = (fk - k0)[..., None]
    col = WHEEL[k0] * (1.0 - fr) + WHEEL[k1] * fr
    val = np.clip(mag / FLOW_FULL, 0.0, 1.0) ** 0.8
    img = (col * val[..., None] * 255.0).astype(np.uint8)
    img[ids == 0] = 0
    return img


def placeholder():
    img = np.empty((TH, TW, 3), np.uint8)
    img[:] = (14, 14, 17)
    return img


try:
    from sensor_panels import EventPanel, LidarPanel
except Exception as e:                           # noqa: BLE001 - the tiles fall back
    print(f"sensor panels unavailable ({e}); placeholder tiles", flush=True)
    EventPanel = LidarPanel = None


class Panel:
    """A sensor_panels panel behind a guard: a failure prints once and the tile
    goes dark."""

    def __init__(self, name, cls, *args):
        self.name, self.p = name, None
        if cls is not None:
            try:
                self.p = cls(*args)
            except Exception as e:               # noqa: BLE001
                print(f"{name} panel failed to start ({e})", flush=True)

    def frame(self, *args):
        if self.p is not None:
            try:
                img = np.asarray(self.p.frame(*args))[..., :3].astype(np.uint8)
                if img.shape[:2] != (TH, TW):
                    img = np.asarray(Image.fromarray(img).resize((TW, TH), Image.BILINEAR))
                return img
            except Exception as e:               # noqa: BLE001
                print(f"{self.name} panel failed ({e})", flush=True)
                self.p = None
        return placeholder()

    def caption(self, default):
        return getattr(self.p, "caption", default) if self.p is not None else default


def tile_captions(dr, labels):
    """labels: [(x, y, head, detail)], (x, y) = tile top-left in the mosaic."""
    m = max(6, int(H * 0.009))
    sh = max(1, H // 540)
    for x0, y0, head, detail in labels:
        x_, y_ = x0 + m, y0 + m
        dr.text((x_ + sh, y_ + sh), head, font=font_tile, fill=(0, 0, 0, 170))
        dr.text((x_, y_), head, font=font_tile, fill=(240, 240, 244, 235))
        if detail:
            xd = x_ + dr.textlength(head + "   ", font=font_tile)
            dr.text((xd + sh, y_ + sh), detail, font=font_tile, fill=(0, 0, 0, 150))
            dr.text((xd, y_), detail, font=font_tile, fill=(190, 190, 198, 200))


def mosaic(rgb, tiles, spd, labels):
    out = np.empty((H, W, 3), np.uint8)
    out[:] = GUTTER_RGB
    out[:2 * TH, :2 * TW] = rgb[:2 * TH, :2 * TW]
    for (col, row), img in tiles.items():
        out[row * TH:(row + 1) * TH, col * TW:(col + 1) * TW] = img
    g0, g1 = GUTTER // 2, GUTTER - GUTTER // 2
    out[:, 2 * TW - g0:2 * TW + g1] = GUTTER_RGB
    out[2 * TH - g0:2 * TH + g1, :] = GUTTER_RGB
    out[TH - g0:TH + g1, 2 * TW:] = GUTTER_RGB
    out[2 * TH:, TW - g0:TW + g1] = GUTTER_RGB
    img = Image.fromarray(out)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    m = int(H * 0.022)
    sh = max(1, H // 540)
    # RGB tile: title top-left, caption bottom-left, slow-motion badge top-right
    dr.text((m + sh, m + sh), TITLE, font=font_title, fill=(0, 0, 0, 160))
    dr.text((m, m), TITLE, font=font_title, fill=(248, 248, 250, 235))
    ys = m + int(H * 0.034)
    dr.text((m + sh, ys + sh), SUBTITLE, font=font_tile, fill=(0, 0, 0, 160))
    dr.text((m, ys), SUBTITLE, font=font_tile, fill=(215, 215, 222, 210))
    yc = 2 * TH - m - int(H * 0.018)
    dr.text((m + sh, yc + sh), CAPTION_S, font=font_tile, fill=(0, 0, 0, 150))
    dr.text((m, yc), CAPTION_S, font=font_tile, fill=(235, 235, 240, 200))
    a = int(210 * max(0.0, min(1.0, (1.0 - spd) / (1.0 - SLOW))))
    if a > 4:
        txt = f"slow motion  1/{round(1.0 / SLOW)}\u00d7"
        tw = dr.textlength(txt, font=font_small)
        xr = 2 * TW - GUTTER - m - tw
        dr.text((xr + sh, m + sh), txt, font=font_small, fill=(0, 0, 0, int(a * 0.6)))
        dr.text((xr, m), txt, font=font_small, fill=(245, 245, 248, a))
    tile_captions(dr, labels)
    return np.asarray(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


total = int(round(VIDEO * 60))
STILLS = [int(total * float(f)) for f in cli_arg("--stills", "0.09,0.30,0.52,0.92", str).split(",")]
SHEET = [int(round(i * (total - 1) / 8)) for i in range(9)]
sheet_imgs = {}
mp4 = os.path.join(OUT_DIR, f"{TAG}.mp4")
# Share-safe H.264: explicit High@4.2, a keyframe every second, moov up front.
enc = Encoder(mp4, W, H, 60, crf=16, preset="slow", faststart=True,
              extra=["-profile:v", "high", "-level:v", "4.2", "-g", "60", "-bf", "2"])
# --master: also a lossless copy (x264 qp 0, 4:4:4) as ground truth for the pixels.
master = (Encoder(os.path.join(OUT_DIR, f"{TAG}_master.mkv"), W, H, 60, crf=0, preset="veryfast",
                  pix_fmt="yuv444p") if "--master" in sys.argv else None)
if SENSORS:
    ev_panel = Panel("event camera", EventPanel, renderer, TW, TH)
    lidar_panel = Panel("lidar", LidarPanel, renderer, scene, TW, TH)
    # meshes the panels added (the lidar post): one muted id, clear of the blocks
    known = {m_.id for m_, _, _ in groups} | {m_.id for m_ in set_meshes} | {ball_mesh.id, chain.id}
    scene.traverse(lambda o: renderer.set_instance_id(o, ID_PROP)
                   if isinstance(o, tp.Mesh) and o.id not in known else None)
    t_sens = {"render+read": 0.0, "gbuffer tiles": 0.0, "event": 0.0, "lidar": 0.0, "compose": 0.0}
refresh_geometry()
camera_at(0.0)
for _ in range(30):
    renderer.render(scene, camera)
prev_pose = (cam_pose["pos"], cam_pose["tgt"])
x_prev, phi_prev = (x.numpy(), pend.phi) if SENSORS else (None, None)
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
    if SENSORS:
        pose = (cam_pose["pos"], cam_pose["tgt"])
        t_a = time.perf_counter()
        aov = renderer.read_aovs_typed(scene, camera, ["rgb", "depth", "instance_ids"])
        t_b = time.perf_counter()
        d_, ids_ = aov["depth"], aov["instance_ids"]
        x_now = x.numpy()
        tiles = {(2, 0): depth_tile(d_, ids_), (2, 1): ids_tile(ids_),
                 (0, 2): flow_tile(d_, ids_, body_motion(x_prev, x_now, phi_prev, pend.phi),
                                   spd, pose, prev_pose)}
        x_prev, phi_prev = x_now, pend.phi
        t_c = time.perf_counter()
        tiles[(1, 2)] = ev_panel.frame()
        t_d = time.perf_counter()
        tiles[(2, 2)] = lidar_panel.frame(scene, camera, sim_t)
        t_e = time.perf_counter()
        labels = [(2 * TW, 0, "DEPTH", f"metric {DEPTH_RANGE[0]:g}\u2013{DEPTH_RANGE[1]:g} m"),
                  (2 * TW, TH, "INSTANCE IDS", f"{NB} bodies"),
                  (0, 2 * TH, "OPTICAL FLOW", "shown at 1\u00d7 speed" if spd < 0.999 else ""),
                  (TW, 2 * TH, "EVENT CAMERA", ev_panel.caption("DVS")),
                  (2 * TW, 2 * TH, "LIDAR", lidar_panel.caption("32 beams"))]
        rgb = mosaic(aov["rgb"], tiles, spd, labels)
        t_f = time.perf_counter()
        for key_, dt_ in zip(t_sens, (t_b - t_a, t_c - t_b, t_d - t_c, t_e - t_d, t_f - t_e)):
            t_sens[key_] += dt_
        prev_pose = pose
    else:
        renderer.render(scene, camera)
        rgb = overlay(renderer.read_pixels(), spd)
    enc.send(rgb)
    if master is not None:
        master.send(rgb)
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
if master is not None:
    master.close()
sheet = Image.new("RGB", (640 * 3, 360 * 3))
for n_, k in enumerate(SHEET):
    if k in sheet_imgs:
        sheet.paste(sheet_imgs[k], ((n_ % 3) * 640, (n_ // 3) * 360))
sheet.save(os.path.join(OUT_DIR, f"{TAG}_contact_sheet.png"))
print(f"ball top max ndc y={cam_stats['ball_ndc_max']:.2f}", flush=True)
if SENSORS and total:
    print("sensor mosaic ms/frame: " + ", ".join(f"{k_} {1000.0 * v_ / total:.1f}"
                                                  for k_, v_ in t_sens.items()), flush=True)
print(f"rendered {total} frames in {time.perf_counter() - tstart:.0f}s -> {mp4} (rc={rc})", flush=True)

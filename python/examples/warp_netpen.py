"""Net-pen inspection ROV: Warp-cloth net with a tear, collar, BlueROV2 on a patrol with tether, bubbles,
camera + sonar insets, and a Warp school of procedural salmon.

    python warp_netpen.py                          # window; drag to orbit, Esc quits
    python warp_netpen.py --shot p3_hud --out x.png --seconds 8 --size 1600x900
    python warp_netpen.py --film [out.mp4]         # 82 s film + contact sheet + poster (--film-test: stills per cut)
Cameras: p1_net_wide p1_collar_below p1_rov_hero p1_tear p2_school p2_fish_close p2_leak p3_hud p3_sonar_tear
         barge barge_stern.
`--fish N` (400); `--profile` prints per-stage ms over 300 live frames and exits; `--no-interop` uploads the school
through the host. Vulkan only; Warp on CUDA if present.
"""
import atexit
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import cli_arg, orbit_loop, parse_size, sky_env, standard_material
try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:
    VkInteropArray = None

SHOT = cli_arg("--shot", "", str)
FILM = "--film" in sys.argv
FILM_BENCH = "--film-bench" in sys.argv               # interleaved A/B of the offline frame path over one short cut
FILM_TEST = "--film-test" in sys.argv                # 3 stills per cut instead of the mp4
FILM = FILM or FILM_TEST                             # --film-test on its own used to fall through to the window
HEADLESS = bool(SHOT) or FILM or FILM_BENCH
SECONDS = cli_arg("--seconds", 6.0, float)
W, H = parse_size(cli_arg("--size", "1600x900", str))
CAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "aaa_caps", "netpen")
OUT = cli_arg("--out", os.path.join(CAP_DIR, f"{SHOT}.png"), str)
FILM_OUT = cli_arg("--film", os.path.join(CAP_DIR, "netpen_film.mp4"), str)
PROFILE = "--profile" in sys.argv
INTEROP = "--no-interop" not in sys.argv          # CUDA->Vulkan zero copy for the school (host upload otherwise)
NET_INTEROP = "--net-interop" in sys.argv         # opt-in repro: four interop meshes = VK_ERROR_DEVICE_LOST on dev 2026-09-03

# ---- the real site ----------------------------------------------------------
# --terrain [packdir] puts the pen where it actually is: Norddalsfjorden,
# 62.28428673 N 7.27893000 E, offshore of a 500 m fjord wall. The pack is
# FETCHED CENTRED ON THAT POINT, so the pack origin IS the site and the pen
# stays exactly where it has always been -- at the world origin, WATER_Y 0.
# Nothing about the pen, the barge or the ROV moves; the world grows around it.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TERRAIN = "--terrain" in sys.argv
TERRAIN_DIR = cli_arg("--terrain", os.path.join(ROOT, "geodata", "norddal"), str)
if TERRAIN and not os.path.isfile(os.path.join(TERRAIN_DIR, "region.json")):
    print(f"no region pack at {TERRAIN_DIR}\nfetch it with:\n"
          f"  python scripts/geodata/fetch_norway_terrain.py "
          f"--center 62.28428672700424,7.27893000198078 --size 4000 --res 1 "
          f"--canopy --preview --name {os.path.basename(TERRAIN_DIR)}")
    sys.exit(2)

# ---- world ------------------------------------------------------------------
WATER_Y = 0.0
PEN_R, PEN_D = 7.0, 8.0
SUN_DIR = np.array([0.62, 0.55, 0.30]); SUN_DIR /= np.linalg.norm(SUN_DIR)
SUN_H = SUN_DIR[[0, 2]] / np.linalg.norm(SUN_DIR[[0, 2]])
TEAR_TH = math.atan2(-SUN_H[1], -SUN_H[0])          # the wall the sun lights from inside
TEAR_Y, TEAR_R = -3.0, 0.65
MESH_M, TILE_MESHES = 0.045, 4
TILE_M = MESH_M * TILE_MESHES
MURK_SIGMA = 0.22                                     # fjord: 4-6 m visibility
MURK_COLOR = tp.Color(0.020, 0.070, 0.055)            # green-grey, low luminance

# ---- net cloth: one periodic grid, wall rows then bottom rings ---------------
NU, NV, NR = 288, 72, 16
NVT = NV + NR
R_IN = 0.9
NET_TOP = WATER_Y - 0.15
DT, SUBSTEPS, ITERATIONS = 1.0 / 480.0, 8, 8
OFFS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2)]

wp.init()
device = wp.get_preferred_device()


def net_rest():
    th = 2.0 * math.pi * np.arange(NU) / NU
    rows = []
    for iv in range(NVT):
        if iv < NV:
            r, y = PEN_R, NET_TOP - iv * PEN_D / (NV - 1)
        else:
            r, y = PEN_R - (iv - NV + 1) * (PEN_R - R_IN) / NR, NET_TOP - PEN_D
        rows.append(np.stack([r * np.cos(th), np.full(NU, y), r * np.sin(th)], 1))
    return np.asarray(rows, np.float32).reshape(-1, 3)


S0 = PEN_R * (TEAR_TH % (2.0 * math.pi))            # arc length of the tear meridian
D0 = NET_TOP - TEAR_Y                                # row length down to the tear centre
TONGUES = [(1.55, 0.22, 0.20), (2.85, 0.24, 0.15), (-0.45, 0.20, 0.13)]   # (rim angle, length, half-width)


def wall_coords(p):
    """(arc offset from the tear meridian, height above the tear centre) for wall point(s)."""
    p = np.asarray(p, np.float64)
    dth = (np.arctan2(p[..., 2], p[..., 0]) - TEAR_TH + np.pi) % (2.0 * np.pi) - np.pi
    return PEN_R * dth, p[..., 1] - TEAR_Y


def tear_radius(phi):
    return TEAR_R * (1.0 + 0.30 * np.sin(3.0 * phi + 1.1) + 0.18 * np.sin(7.0 * phi + 0.4) + 0.10 * np.sin(13.0 * phi + 2.0))


def tongue_frame(s, d, k):
    """(along, across) of wall point(s) in tongue k's frame: base on the rim, axis pointing into the hole."""
    phi, _, _ = TONGUES[k]
    c, sn = math.cos(phi), math.sin(phi)
    bs, bd = (tear_radius(phi) + 0.08) * c, (tear_radius(phi) + 0.08) * sn
    return -(s - bs) * c - (d - bd) * sn, -(s - bs) * sn + (d - bd) * c


def tear_classes(rest):
    """Per vertex: -1 intact net, -2 invisible hole interior, k = flap tongue k; plus its hinge flag."""
    s, d = wall_coords(rest)
    r, phi = np.hypot(s, d), np.arctan2(d, s)
    cls = np.full(len(rest), -1, np.int32)
    hinge = np.zeros(len(rest), bool)
    cls[r < tear_radius(phi) - 0.08] = -2
    for k, (_, L, w) in enumerate(TONGUES):
        t, q = tongue_frame(s, d, k)
        on = (cls == -2) & (t > -0.1) & (t < L + 0.12) & (np.abs(q) < w + 0.10)
        cls[on] = k
        hinge[on & (t < 0.12)] = True
    return cls, hinge


def net_mask(cls, hinge):
    """Spring bits: a tongue keeps springs inside itself and from its hinge row to the rim, nothing else."""
    mask = np.zeros(NU * NVT, np.int32)
    iu, iv = np.arange(NU * NVT) % NU, np.arange(NU * NVT) // NU
    for bit, (du, dv) in enumerate(OFFS):
        jv = iv + dv
        j = np.clip(jv, 0, NVT - 1) * NU + (iu + du) % NU
        cj, hj = cls[j], hinge[j]
        keep = (cls == cj) | ((cls == -1) & hj) | ((cj == -1) & hinge) | ((cls < 0) & (cj < 0))
        mask[(jv >= 0) & (jv < NVT) & ~keep] |= 1 << bit
    return mask


@wp.func
def spring_m(p: wp.vec3, pos: wp.array(dtype=wp.vec3), rest: wp.array(dtype=wp.vec3),
             i: int, iu: int, iv: int, nu: int, nvt: int, stiffness: float,
             mask: int, bit: int) -> wp.vec3:
    if (mask & (1 << bit)) != 0 or iv < 0 or iv >= nvt:
        return wp.vec3(0.0, 0.0, 0.0)
    j = iv * nu + (iu + nu) % nu
    d = pos[j] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    r = wp.length(rest[j] - rest[i])
    return d * (0.5 * stiffness * (l - r) / l)


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
              inv_mass: wp.array(dtype=float), load: wp.array(dtype=float),
              vec_uni: wp.array(dtype=wp.vec3), sc_uni: wp.array(dtype=float), dt: float):
    i = wp.tid()
    p = pos[i]
    step = p - prev[i]
    prev[i] = p
    if inv_mass[i] == 0.0:
        pred[i] = p
        return
    t = sc_uni[6]
    # Current: a world flow plus a slow spatial swell so the wall breathes, not slides.
    cur = vec_uni[0] * (1.0 + 0.30 * wp.sin(0.45 * p[1] + 0.22 * p[0] + 0.9 * t)
                        + 0.12 * wp.sin(0.9 * p[2] - 0.6 * t))
    v_rel = cur - step / dt
    n = nrm[i]
    vn = wp.dot(n, v_rel)
    acc = wp.vec3(0.0, -load[i], 0.0) + n * (sc_uni[0] * vn * wp.abs(vn)) + (v_rel - n * vn) * sc_uni[1]
    pred[i] = p + step * (1.0 - sc_uni[2]) + acc * dt * dt


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3),
          rest: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
          mask: wp.array(dtype=int), sc_uni: wp.array(dtype=float), nu: int, nvt: int):
    i = wp.tid()
    p = p_in[i]
    if inv_mass[i] == 0.0:
        p_out[i] = p
        return
    iu = i % nu
    iv = i // nu
    m = mask[i]
    st = sc_uni[3]
    sh = sc_uni[4]
    bd = sc_uni[5]
    c = wp.vec3(0.0, 0.0, 0.0)
    c += spring_m(p, p_in, rest, i, iu - 1, iv, nu, nvt, st, m, 0)
    c += spring_m(p, p_in, rest, i, iu + 1, iv, nu, nvt, st, m, 1)
    c += spring_m(p, p_in, rest, i, iu, iv - 1, nu, nvt, st, m, 2)
    c += spring_m(p, p_in, rest, i, iu, iv + 1, nu, nvt, st, m, 3)
    c += spring_m(p, p_in, rest, i, iu - 1, iv - 1, nu, nvt, sh, m, 4)
    c += spring_m(p, p_in, rest, i, iu + 1, iv - 1, nu, nvt, sh, m, 5)
    c += spring_m(p, p_in, rest, i, iu - 1, iv + 1, nu, nvt, sh, m, 6)
    c += spring_m(p, p_in, rest, i, iu + 1, iv + 1, nu, nvt, sh, m, 7)
    c += spring_m(p, p_in, rest, i, iu - 2, iv, nu, nvt, bd, m, 8)
    c += spring_m(p, p_in, rest, i, iu + 2, iv, nu, nvt, bd, m, 9)
    c += spring_m(p, p_in, rest, i, iu, iv - 2, nu, nvt, bd, m, 10)
    c += spring_m(p, p_in, rest, i, iu, iv + 2, nu, nvt, bd, m, 11)
    p_out[i] = p + c * 0.3


@wp.kernel
def compute_normals(pos: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3), nu: int, nvt: int):
    i = wp.tid()
    iu = i % nu
    iv = i // nu
    xm = pos[iv * nu + (iu - 1 + nu) % nu]
    xp = pos[iv * nu + (iu + 1) % nu]
    ym = pos[wp.max(iv - 1, 0) * nu + iu]
    yp = pos[wp.min(iv + 1, nvt - 1) * nu + iu]
    n = wp.cross(yp - ym, xp - xm)
    nrm[i] = n / wp.max(wp.length(n), 1.0e-9)


class Net:
    def __init__(self):
        self.n = NU * NVT
        rest = net_rest()
        rng = np.random.default_rng(7)
        p0 = np.ascontiguousarray(rest + rng.uniform(-3e-3, 3e-3, rest.shape), np.float32)
        self.cls, hinge = tear_classes(rest)
        mask = net_mask(self.cls, hinge)
        iv = np.arange(self.n) // NU
        inv_mass = np.ones(self.n, np.float32)
        inv_mass[iv == 0] = 0.0                       # hung from the collar
        load = np.full(self.n, 0.10, np.float32)      # twine: barely negative buoyancy
        load[iv == NV - 1] = 2.4                      # sinker tube
        load[self.cls >= 0] = 0.55                    # torn flaps droop
        self.pos = wp.array(p0, dtype=wp.vec3, device=device)
        self.prev = wp.array(p0, dtype=wp.vec3, device=device)
        self.pred = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.scratch = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.nrm = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.rest = wp.array(rest, dtype=wp.vec3, device=device)
        self.inv_mass = wp.array(inv_mass, dtype=float, device=device)
        self.load = wp.array(load, dtype=float, device=device)
        self.mask = wp.array(mask, dtype=int, device=device)
        self.vec_uni = wp.zeros(1, dtype=wp.vec3, device=device)
        self.sc_uni = wp.zeros(8, dtype=float, device=device)
        self.rest_host = rest

    def push_uniforms(self, current, t):
        self.vec_uni.assign(np.asarray([current], np.float32))
        self.sc_uni.assign(np.asarray([55.0, 2.5, 0.06, 1.0, 0.5, 0.35, t, 0.0], np.float32))

    def launches(self):
        for _ in range(SUBSTEPS):
            wp.launch(compute_normals, dim=self.n, device=device, inputs=[self.pos, self.nrm, NU, NVT])
            wp.launch(integrate, dim=self.n, device=device,
                      inputs=[self.pos, self.prev, self.pred, self.nrm, self.inv_mass, self.load,
                              self.vec_uni, self.sc_uni, DT])
            a, b = self.pred, self.scratch
            for _ in range(ITERATIONS):
                wp.launch(solve, dim=self.n, device=device,
                          inputs=[a, b, self.rest, self.inv_mass, self.mask, self.sc_uni, NU, NVT])
                a, b = b, a
            wp.copy(self.pos, a)
        wp.launch(compute_normals, dim=self.n, device=device, inputs=[self.pos, self.nrm, NU, NVT])


ROPE_N, ROPE_L = 40, 5.4                             # short enough to go taut at the deep end of the patrol
ROPE_SEG = ROPE_L / (ROPE_N - 1)


@wp.func
def rope_spring(p: wp.vec3, pos: wp.array(dtype=wp.vec3), j: int, n: int, r: float, k: float) -> wp.vec3:
    if j < 0 or j >= n:
        return wp.vec3(0.0, 0.0, 0.0)
    d = pos[j] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    return d * (0.5 * k * (l - r) / l)


@wp.kernel
def rope_integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), pred: wp.array(dtype=wp.vec3),
                   pins: wp.array(dtype=wp.vec3), n: int, dt: float):
    i = wp.tid()
    p = pos[i]
    step = p - prev[i]
    prev[i] = p
    if i == 0:
        pred[i] = pins[0]
        return
    if i == n - 1:
        pred[i] = pins[1]
        return
    q = p + step * 0.92 + wp.vec3(0.0, 0.30, 0.0) * dt * dt        # slightly buoyant, heavily damped
    pred[i] = wp.vec3(q[0], wp.min(q[1], -0.06), q[2])


@wp.kernel
def rope_solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3), n: int, seg_a: wp.array(dtype=float)):
    i = wp.tid()
    p = p_in[i]
    seg = seg_a[0]
    if i == 0 or i == n - 1:
        p_out[i] = p
        return
    c = rope_spring(p, p_in, i - 1, n, seg, 1.0) + rope_spring(p, p_in, i + 1, n, seg, 1.0)
    c += rope_spring(p, p_in, i - 2, n, 1.96 * seg, 0.25) + rope_spring(p, p_in, i + 2, n, 1.96 * seg, 0.25)
    p_out[i] = p + c * 0.8


class Rope:
    def __init__(self):
        z = np.zeros((ROPE_N, 3), np.float32)
        self.pos, self.prev = wp.array(z, dtype=wp.vec3, device=device), wp.array(z, dtype=wp.vec3, device=device)
        self.pred, self.scratch = wp.zeros(ROPE_N, dtype=wp.vec3, device=device), wp.zeros(ROPE_N, dtype=wp.vec3, device=device)
        self.pins = wp.zeros(2, dtype=wp.vec3, device=device)
        self.seg = wp.array(np.float32([ROPE_SEG]), dtype=float, device=device)   # rest length per segment, paid out per frame

    def reset(self, a, b):
        p = np.linspace(a, b, ROPE_N).astype(np.float32)
        p[:, 1] = np.minimum(p[:, 1] + 0.6 * np.sin(np.pi * np.linspace(0, 1, ROPE_N)), -0.06)
        self.pos.assign(p)
        self.prev.assign(p)
        self.pins.assign(np.asarray([a, b], np.float32))

    def pay_out(self, dist):
        """Rest length 1.06 x the straight run (capped at the rope): a gentle catenary, never loops."""
        self.seg.assign(np.float32([min(1.06 * dist, ROPE_L) / (ROPE_N - 1)]))

    def launches(self):
        for _ in range(SUBSTEPS):
            wp.launch(rope_integrate, dim=ROPE_N, device=device, inputs=[self.pos, self.prev, self.pred, self.pins, ROPE_N, DT])
            a, b = self.pred, self.scratch
            for _ in range(ITERATIONS):
                wp.launch(rope_solve, dim=ROPE_N, device=device, inputs=[a, b, ROPE_N, self.seg])
                a, b = b, a
            wp.copy(self.pos, a)


net = Net()
rope = Rope()
graph = None
if device.is_cuda:
    with wp.ScopedCapture(device) as cap:
        net.launches()
        rope.launches()
    graph = cap.graph


def net_step(current, t):
    net.push_uniforms(current, t)
    if graph is not None:
        wp.capture_launch(graph)
    else:
        net.launches()
        rope.launches()


# ---- textures ----------------------------------------------------------------
def vnoise(Y, X, cy, cx, rng):
    """Value noise at (Y, X) given in cell units, periodic on a (cy, cx) lattice."""
    g = rng.random((cy, cx)).astype(np.float32)
    y0, x0 = np.floor(Y).astype(int), np.floor(X).astype(int)
    ty, tx = Y - y0, X - x0
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    y0, x0 = y0 % cy, x0 % cx
    y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
    return (g[y0, x0] * (1 - tx) * (1 - ty) + g[y0, x1] * tx * (1 - ty)
            + g[y1, x0] * (1 - tx) * ty + g[y1, x1] * tx * ty)


def fbm(Y, X, cy, cx, rng, octaves=5):
    out, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        out = out + amp * vnoise(Y * (1 << k), X * (1 << k), cy << k, cx << k, rng)
        tot += amp
        amp *= 0.5
    return (out / tot).astype(np.float32)


def grid(h, w, cy, cx):
    """(Y, X) broadcastable cell coordinates covering a (h, w) texture with (cy, cx) cells."""
    return ((np.arange(h) + 0.5) / h * cy)[:, None], ((np.arange(w) + 0.5) / w * cx)[None, :]


def net_maps(U, V, ppm, keep=None, twine=0.11):
    """Knotted square mesh at tile coords (U, V) in meshes -> (albedo+alpha, normal map) uint8 arrays."""
    rng = np.random.default_rng(3)
    U, V = U % TILE_MESHES, V % TILE_MESHES
    hw = 0.5 * twine
    wob_u = 0.035 * np.sin(2 * np.pi * V + 0.7) + 0.02 * np.sin(4 * np.pi * V * 1.3)
    wob_v = 0.035 * np.sin(2 * np.pi * U + 2.1) + 0.02 * np.sin(4 * np.pi * U * 0.7)
    du = np.abs((U + wob_u) % 1.0 - 0.5)
    dv = np.abs((V + wob_v) % 1.0 - 0.5)
    du, dv = 0.5 - du, 0.5 - dv
    hv = np.sqrt(np.clip(1 - (du / hw) ** 2, 0, 1))
    hh = np.sqrt(np.clip(1 - (dv / hw) ** 2, 0, 1))
    dk = np.hypot(du, dv)
    hk = 1.25 * np.sqrt(np.clip(1 - (dk / (1.7 * hw)) ** 2, 0, 1))
    hgt = np.maximum(np.maximum(hv, hh), hk)
    fibre = 0.82 + 0.36 * fbm(V / TILE_MESHES * 16, U / TILE_MESHES * 16, 16, 16, rng)
    twist = 0.5 + 0.5 * np.sin(2 * np.pi * (V * 18 + U * 18)) * (hv > hh) + 0.5 * np.sin(2 * np.pi * (U * 18 - V * 18)) * (hv <= hh)
    base = np.float32([0.34, 0.37, 0.28])
    col = base[None, None, :] * (fibre * (0.85 + 0.3 * twist))[..., None]
    col[hk > 0.98] *= 0.7
    alpha = hgt > 0.0 if keep is None else (hgt > 0.0) & keep
    rgba = np.zeros(hgt.shape + (4,), np.uint8)
    rgba[..., :3] = np.clip(col * 255, 0, 255)
    rgba[..., 3] = np.where(alpha, 255, 0)
    gy, gx = np.gradient(hgt * 0.5 * hw * ppm * 0.012)
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nmap = np.full(hgt.shape + (4,), 255, np.uint8)
    nmap[..., :3] = np.clip((nrm * 0.5 + 0.5) * 255, 0, 255)
    return rgba, nmap


def net_tile(px=256):
    u = (np.arange(px) + 0.5) / px * TILE_MESHES
    U, V = np.meshgrid(u, u)
    rgba, nmap = net_maps(U, V, px / TILE_MESHES)
    return tp.data_texture(rgba, srgb=True), tp.data_texture(nmap, srgb=False)


def tear_keep(U, V, s, d, rng):
    """Texel survives the tear: outside the noisy rim, on a flap tongue, or a broken twine end poking in."""
    r, phi = np.hypot(s, d), np.arctan2(d, s)
    n = fbm(d / 0.05, s / 0.05, 64, 64, rng, 3)
    keep = r > tear_radius(phi) * (0.93 + 0.14 * n)
    for k, (_, L, w) in enumerate(TONGUES):
        t, q = tongue_frame(s, d, k)
        keep |= (t > -0.05) & (t < L * (0.85 + 0.3 * n)) & (np.abs(q) < np.minimum(w * (1 - 0.3 * t / L) * (0.7 + 0.6 * n), w + 0.03))
    hw = 0.5 * 0.11
    for _ in range(9):
        ph = rng.uniform(-np.pi, np.pi)
        rs, rd = tear_radius(ph) * math.cos(ph), tear_radius(ph) * math.sin(ph)
        uk, vk = (rs + S0) / MESH_M, (D0 - rd) / MESH_M
        ln = rng.uniform(0.5, 1.5)
        if rng.random() < 0.5:                        # along a vertical twine, toward the hole centre
            sg = 1.0 if rd > 0 else -1.0
            keep |= (np.abs(U - np.round(uk)) < 1.6 * hw) & ((V - vk) * sg > -0.3) & ((V - vk) * sg < ln)
        else:
            sg = -1.0 if rs > 0 else 1.0
            keep |= (np.abs(V - np.round(vk)) < 1.6 * hw) & ((U - uk) * sg > -0.3) & ((U - uk) * sg < ln)
    return keep


def tear_patch(px=2048):
    """Net texture over the tear window at texel resolution: the ragged rim lives here, not in the grid."""
    Y, X = grid(px, px, 1, 1)
    U = PATCH_U[0] + X * (PATCH_U[1] - PATCH_U[0])
    V = PATCH_V[0] + Y * (PATCH_V[1] - PATCH_V[0])
    U, V = np.broadcast_arrays(U, V)
    s, d = U * MESH_M - S0, D0 - V * MESH_M
    keep = tear_keep(U, V, s, d, np.random.default_rng(21))
    rgba, nmap = net_maps(U, V, px / (PATCH_U[1] - PATCH_U[0]), keep)
    return tp.data_texture(rgba, srgb=True), tp.data_texture(nmap, srgb=False)


def fouling_maps():
    """Biofouling: an unwrapped wall texture of lacy olive clumps plus a per-particle tint weight."""
    rng = np.random.default_rng(11)
    h, w = 2048, 4096
    f = fbm(*grid(h, w, 18, 72), 18, 72, rng)
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    th = 2 * np.pi * u
    sunlit = 0.5 + 0.5 * np.cos(th - TEAR_TH)
    depth_w = np.clip(1.15 - 1.6 * v * NVT / NV, 0.0, 1.0)      # algae wants light
    weight = (0.35 + 0.65 * sunlit)[None, :] * depth_w[:, None]
    weight[v > (NV - 1) / (NVT - 1)] = 0.05
    thr = 0.70 - 0.24 * weight
    edge = np.clip((f - thr) / 0.10, 0, 1)
    fine = fbm(*grid(h, w, 192, 768), 192, 768, np.random.default_rng(5), 4)
    lace = fbm(*grid(h, w, 384, 1536), 384, 1536, np.random.default_rng(6), 3)
    alpha = (edge * (0.4 + 1.2 * (fine - 0.35)) > 0.35) & (lace > 0.28 + 0.35 * (1 - edge))   # frayed, holey
    C = 2 * np.pi * PEN_R
    s = ((u * C - S0 + 0.5 * C) % C - 0.5 * C)[None, :]
    d = (D0 - v * (NVT - 1) * PEN_D / (NV - 1))[:, None]
    alpha &= np.hypot(s, d) > tear_radius(np.arctan2(d, s)) + 0.03
    tint = fbm(*grid(h, w, 12, 48), 12, 48, np.random.default_rng(9))[..., None]
    col = np.float32([0.42, 0.45, 0.19]) * (1 - tint) + np.float32([0.36, 0.30, 0.13]) * tint
    col *= (0.75 + 0.5 * fine)[..., None]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = np.clip(col * 255, 0, 255)
    rgba[..., 3] = np.where(alpha, 255, 0)
    iu = np.arange(NU * NVT) % NU
    iv = np.arange(NU * NVT) // NU
    pw = weight[np.minimum((iv * h) // (NVT - 1), h - 1), (iu * w) // NU]
    pf = f[np.minimum((iv * h) // (NVT - 1), h - 1), (iu * w) // NU]
    return tp.data_texture(rgba, srgb=True), np.clip(pw * (0.4 + 1.4 * pf), 0, 1).astype(np.float32)


# ---- renderer ----------------------------------------------------------------
canvas = tp.Canvas("threepp x warp - net pen", width=W, height=H, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.AgX
renderer.tone_mapping_exposure = 0.72
renderer.render_scale = 1.0
renderer.gbuffer_msaa = 2
renderer.sun_angular_radius = 0.6
renderer.bloom_intensity = 0.10
renderer.bloom_clamp = 12.0
renderer.auto_exposure = True
AE_RANGE = (-2.2, 1.2)
renderer.set_auto_exposure_range(*AE_RANGE)
renderer.set_auto_exposure_speed(1.2)
renderer.fog_anisotropy = 0.55
renderer.deferred_ao = False       # RT AO reads the coincident cutout planes as walls: blackens twine and fouling

scene = tp.Scene()
sky = sky_env(SUN_DIR, below_horizon=(0.16, 0.24, 0.20), below_nadir=(0.05, 0.10, 0.08))
scene.environment = sky
scene.background = sky
sun = tp.DirectionalLight(0xfff0d8, 2.4)
sun.position.set(*(SUN_DIR * 1000.0))
scene.add(sun)

# ---- terrain: the real fjord around the pen ----------------------------------
# Loaded BEFORE the ocean because the pack size decides how far the water has to
# reach. There is ONE ocean either way -- a second sheet for the far field is
# what put two overlapping transmissive surfaces in the scene, and since you see
# through the near one, the far one read as a second water surface under it.
GEO = None
if TERRAIN:
    _t0 = time.perf_counter()
    # forest_focus at the origin = the pen: a 4 km pack is 16 M cells and every
    # cut looks at the same 600 m of wall, so the shell + forest budget is spent
    # there rather than thin across the square. scatter off -- the near-field
    # stones/tufts ring follows the CAMERA, and this camera lives over (and
    # under) water, so every cell it built would land on the seabed.
    GEO = tp.GeoScene(TERRAIN_DIR, forest_focus=(0.0, 0.0, 0.0), scatter=False)
    scene.add(GEO)
    # The DTM has no soundings; GeoScene's distance-to-shore bathymetry is what
    # puts water under the pen. Print it: if this is ~0 the pack is not centred
    # on open water and the pen is sitting on a beach.
    print(f"terrain: {os.path.basename(TERRAIN_DIR)} {GEO.pack_world_size:.0f} m, "
          f"{GEO.height_min:.0f}..{GEO.height_max:.0f} m, {GEO.stats}; "
          f"seabed under the pen {GEO.height_at(0.0, 0.0):.1f} m, "
          f"loaded in {time.perf_counter() - _t0:.1f} s")
else:
    # No terrain: the flat stand-in seabed the demo has always had.
    floor = tp.Mesh(tp.PlaneGeometry(400.0, 400.0), standard_material(0x03060a))
    floor.rotate_x(-math.pi / 2)
    floor.position.y = -45.0
    scene.add(floor)

# ---- the one ocean -----------------------------------------------------------
# Without terrain: the 320 m sheet the demo has always had, 0.83 m spacing.
# With terrain the same sheet has to span the pack, so instead of adding a
# second one the SINGLE mesh is stretched to the shore and its vertices are
# packed toward the pen by DisplacedMesh.warp:
#     x = halfRange * (coefA*u + (1 - coefA)*u^3),  u in [-1, 1]
# At 512 rows over +-2500 m with coefA 0.07 that is 0.68 m at the pen (finer
# than the old near sheet), ~1.7 m at 50 m, ~4.7 m at the 200 m shore and 28 m
# at the pack edge, where a crest is a pixel. 512^2 verts against the 384^2 +
# 256^2 the two sheets cost, and one FFT pipeline instead of two.
OCEAN_SIZE = (GEO.pack_world_size * 1.25) if TERRAIN else 320.0
ocean = tp.Ocean(size=OCEAN_SIZE, resolution=512 if TERRAIN else 384, wind_speed=7.0,
                 wind_theta=0.6, choppiness=0.8, fft_size=512, fetch=9e3)   # fjord in a fresh breeze
ocean.params.foam_amount = 0.0                 # whitecaps print as white slabs from below
# Ocean sizes cascade 0 from the mesh extent (Ocean.cpp: tileSize0 = size), so a
# 5 km sheet would ask for a 5 km swell. Pin the tile: the sea stays the young
# fjord chop the pen cuts were lit for, whatever the mesh spans.
ocean.params.tile_size_0 = 320.0
ocean.params.tile_size_1 = 40.6
ocean.params.tile_size_2 = 2.98
if TERRAIN:
    ocean.warp.center_x = 0.0
    ocean.warp.center_z = 0.0
    ocean.warp.half_range = OCEAN_SIZE * 0.5
    ocean.warp.coef_a = 0.07
ocean.material.attenuation_color = tp.Color(0.16, 0.30, 0.24)   # fjord water from above: dark green-grey, not tropical teal
ocean.material.attenuation_distance = 2.5
if "--no-ocean" not in sys.argv:
    scene.add(ocean)

MURK_PLANE = WATER_Y + 0.42            # deep cuts: above every crest, or a crest shaded from above prints sun glints through the murk


def murk_plane_update():
    """The per-pixel over/under split rides the LOCAL wave through the lens; the +0.30 crest guard blends in with depth."""
    cp = camera.position
    k = min(max((WATER_Y - cp.y - 0.5) / 1.0, 0.0), 1.0)
    k = k * k * (3.0 - 2.0 * k)
    renderer.set_fog_water_surface_y(float(ocean.sample_height(cp.x, cp.z)) * (1.0 - k) + MURK_PLANE * k)
renderer.set_underwater_murk(MURK_SIGMA, MURK_COLOR)

# ---- net meshes --------------------------------------------------------------
GATHER = ((np.arange(NU + 1) % NU)[None, :] + NU * np.arange(NVT)[:, None]).reshape(-1)
rest_g = net.rest_host[GATHER]
radii = np.hypot(rest_g[:, 0], rest_g[:, 2]).reshape(NVT, NU + 1)
row_y = np.abs(np.diff(net.rest_host.reshape(NVT, NU, 3)[:, 0, :], axis=0))
row_len = np.concatenate([[0.0], np.cumsum(np.hypot(row_y[:, 0], row_y[:, 1]))])
uv_net = np.stack([np.tile(np.arange(NU + 1), NVT) * (2 * np.pi * radii.reshape(-1) / NU) / TILE_M,
                   np.repeat(row_len, NU + 1) / TILE_M], 1).astype(np.float32)
uv_foul = np.stack([np.tile(np.arange(NU + 1) / NU, NVT),
                    np.repeat(np.arange(NVT) / (NVT - 1), NU + 1)], 1).astype(np.float32)
# The tear patch: a window of grid cells around the tear with its own texel-resolution net texture.
IU_C, IV_C = int(round(S0 / (2 * np.pi * PEN_R) * NU)), int(round(D0 / (PEN_D / (NV - 1))))
IU0, IU1, IV0, IV1 = IU_C - 8, IU_C + 8, IV_C - 11, IV_C + 11
PATCH_U = (IU0 * 2 * np.pi * PEN_R / NU / MESH_M, IU1 * 2 * np.pi * PEN_R / NU / MESH_M)
PATCH_V = (row_len[IV0] / MESH_M, row_len[IV1] / MESH_M)
uv_patch = np.stack([(uv_net[:, 0] * TILE_MESHES - PATCH_U[0]) / (PATCH_U[1] - PATCH_U[0]),
                     (uv_net[:, 1] * TILE_MESHES - PATCH_V[0]) / (PATCH_V[1] - PATCH_V[0])], 1).astype(np.float32)

tile_tex, tile_nrm = net_tile()
patch_tex, patch_nrm = tear_patch()
foul_tex, foul_w = fouling_maps()
tint = np.float32([0.55, 0.62, 0.30])
vcol = ((1.0 - foul_w[GATHER][:, None]) * 1.0 + foul_w[GATHER][:, None] * tint[None, :]).astype(np.float32)


def net_material(tex, nrm):
    m = standard_material(0xffffff, roughness=0.78, metalness=0.0, side=tp.Side.Double)
    m.map = tex
    m.normal_map = nrm
    m.normal_scale = tp.Vector2(0.8, 0.8)
    m.alpha_test = 0.30
    m.vertex_colors = True
    m.translucency = 0.45              # thin twine reads back-lit from inside the pen
    m.translucency_color = 0x9aa48c
    return m


net_mat = net_material(tile_tex, tile_nrm)
patch_mat = net_material(patch_tex, patch_nrm)
foul_mat = standard_material(0xffffff, roughness=0.92, metalness=0.0, side=tp.Side.Double)
foul_mat.map = foul_tex
foul_mat.alpha_test = 0.5
foul_mat.translucency = 0.3
foul_mat.translucency_color = 0x8fa552


def net_index():
    """Full grid triangles split into (outside patch, inside patch); cut cells stay and stretch, the rim is texture."""
    iv, iu = np.meshgrid(np.arange(NVT - 1), np.arange(NU), indexing="ij")
    ga, gb = iv * (NU + 1) + iu, iv * (NU + 1) + iu + 1
    gc, gd = gb + NU + 1, ga + NU + 1
    tri = np.concatenate([np.stack([ga, gb, gc], -1), np.stack([ga, gc, gd], -1)]).reshape(-1, 3)
    cu, cv = tri[:, 0] % (NU + 1), tri[:, 0] // (NU + 1)
    inp = (cu >= IU0) & (cu < IU1) & (cv >= IV0) & (cv < IV1)
    return tri[~inp].reshape(-1).astype(np.uint32), tri[inp].reshape(-1).astype(np.uint32)


NET_INDEX, PATCH_INDEX = net_index()


def net_geometry(uv, index):
    g = tp.PlaneGeometry(1.0, 1.0, NU, NVT - 1)
    g.set_attribute("position", rest_g)
    g.set_attribute("uv", uv)
    g.set_attribute("color", vcol)
    g.set_index(index)
    return g


net_geo = net_geometry(uv_net, NET_INDEX)
patch_geo = net_geometry(uv_patch, PATCH_INDEX)
foul_geo = net_geometry(uv_foul, np.concatenate([NET_INDEX, PATCH_INDEX]))
net_meshes = []
for geo, mat in ((net_geo, net_mat), (patch_geo, patch_mat), (foul_geo, foul_mat)):
    m = tp.Mesh(geo, mat)
    m.frustum_culled = False
    m.cast_shadow = True
    m.receive_shadow = True
    scene.add(m)
    net_meshes.append(m)


@wp.kernel
def gather_net(pos: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3), ids: wp.array(dtype=int), lift: float,
               out_p: wp.array(dtype=wp.vec3), out_n: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    j = ids[i]
    out_p[i] = pos[j] + nrm[j] * lift
    out_n[i] = nrm[j]


_gather_ids = wp.array(GATHER.astype(np.int32), dtype=int, device=device)
net_vk = []                                           # per net mesh: (pos, nrm) VkInteropArray pair once armed


def net_on_frame(k):
    def cb():
        t0 = time.perf_counter()
        p, n = net_vk[k]
        wp.launch(gather_net, dim=len(GATHER), device=device,
                  inputs=[net.pos, net.nrm, _gather_ids, 0.0005 if k == 2 else 0.0, p.array, n.array])
        wp.synchronize_device(device)
        if prof:
            prof.add("net gather (in render)", time.perf_counter() - t0)
    return cb


def arm_net_interop():
    """After the first render: all three net meshes go zero-copy, or none (net_upload is one host path)."""
    if VkInteropArray is None or not device.is_cuda or not (INTEROP and NET_INTEROP):
        return
    for k, m in enumerate(net_meshes):
        h = renderer.enable_vertex_interop(m, net_on_frame(k))
        if h is None:
            break
        try:
            net_vk.append(tuple(VkInteropArray(hd, nb, wp.vec3, len(GATHER), device) for hd, nb in h))
        except Exception as e:
            print(f"net interop import failed ({e}); host upload path")
            renderer.disable_vertex_interop(m)
            break
    if len(net_vk) < len(net_meshes):
        release_net_interop()
    else:
        atexit.register(release_net_interop)


def release_net_interop():
    pairs = list(net_vk)
    net_vk.clear()
    for (p, n), m in zip(pairs, net_meshes):
        p.close()
        n.close()
        renderer.disable_vertex_interop(m)


def net_upload():
    if net_vk:
        return
    pos = net.pos.numpy()
    nrm = net.nrm.numpy()
    net_host[0] = pos
    for g in (net_geo, patch_geo):
        g.update_attribute("position", pos[GATHER])
        g.update_attribute("normal", nrm[GATHER])
    # Coincident within the shadow-ray bias: a cutout plane is opaque to RT shadows.
    foul_geo.update_attribute("position", (pos + nrm * 0.0005)[GATHER])
    foul_geo.update_attribute("normal", nrm[GATHER])


# ---- collar ------------------------------------------------------------------
hdpe = standard_material(0x0b0d10, roughness=0.55)
hdpe_blue = standard_material(0x143b7a, roughness=0.5)
steel = standard_material(0x9aa0a6, roughness=0.4, metalness=0.9)
rope_mat = standard_material(0x2b2f2a, roughness=0.9)


def align_y(obj, d):
    """Rotate an object whose axis is +Y onto direction d."""
    d = np.asarray(d, np.float64)
    d /= np.linalg.norm(d)
    axis = np.cross([0.0, 1.0, 0.0], d)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        if d[1] < 0:
            obj.rotation.x = math.pi
        return
    obj.quaternion.set_from_axis_angle(tp.Vector3(*(axis / s)), math.atan2(s, d[1]))


def tube(a, b, r, mat, seg=10):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = float(np.linalg.norm(b - a))
    m = tp.Mesh(tp.CylinderGeometry(r, r, L, seg, 1), mat)
    m.position.set(*(0.5 * (a + b)))
    align_y(m, b - a)
    return m


collar = tp.Group()
for R, mat in ((PEN_R, hdpe), (PEN_R + 0.85, hdpe_blue)):
    ring = tp.Mesh(tp.TorusGeometry(R, 0.16, 14, 160), mat)
    ring.rotation.x = math.pi / 2
    ring.position.y = WATER_Y + 0.04
    ring.cast_shadow = True
    collar.add(ring)
rail = tp.Mesh(tp.TorusGeometry(PEN_R + 0.85, 0.025, 8, 160), hdpe)
rail.rotation.x = math.pi / 2
rail.position.y = WATER_Y + 1.05
collar.add(rail)
walk = tp.Mesh(tp.RingGeometry(PEN_R + 0.18, PEN_R + 0.68, 160, 1), standard_material(0x6e6656, roughness=0.95, side=tp.Side.Double))
walk.rotation.x = -math.pi / 2
walk.position.y = WATER_Y + 0.22
collar.add(walk)
for k in range(48):
    th = 2 * math.pi * k / 48
    c, s = math.cos(th), math.sin(th)
    br = tp.Mesh(tp.BoxGeometry(1.05, 0.08, 0.10), hdpe)
    br.position.set((PEN_R + 0.425) * c, WATER_Y + 0.12, (PEN_R + 0.425) * s)
    br.rotation.y = -th
    collar.add(br)
    if k % 2 == 0:
        collar.add(tube(((PEN_R + 0.85) * c, WATER_Y + 0.1, (PEN_R + 0.85) * s),
                        ((PEN_R + 0.85) * c, WATER_Y + 1.05, (PEN_R + 0.85) * s), 0.02, hdpe, 8))
    if k % 6 == 0:
        far = np.array([(PEN_R + 38.0) * c, WATER_Y - 22.0, (PEN_R + 38.0) * s])
        collar.add(tube(((PEN_R + 0.85) * c, WATER_Y - 0.05, (PEN_R + 0.85) * s), far, 0.022, rope_mat, 6))
scene.add(collar)

# ---- feed barge (fôrflåte) ---------------------------------------------------
BARGE_POS = np.array([22.0, 0.0, 16.0])          # ~27 m from pen centre; visible in the aerial approach
BARGE_YAW = 0.85                                   # broadside toward the approach camera
BARGE_L, BARGE_W = 20.0, 10.0
_b_bow  = np.array([math.cos(BARGE_YAW), 0.0, -math.sin(BARGE_YAW)])   # barge local +X in world
_b_port = np.array([math.sin(BARGE_YAW), 0.0,  math.cos(BARGE_YAW)])   # barge local +Z in world
_bcy, _bsy = math.cos(BARGE_YAW), math.sin(BARGE_YAW)


def pipe_along(pts, radius, sides, mat):
    """Tube mesh along a polyline (feed-line pipes)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    up = np.where(np.abs(d[:, 1:2]) < 0.9, np.float64([[0, 1, 0]]), np.float64([[1, 0, 0]]))
    b1 = np.cross(d, up)
    b1 /= np.maximum(np.linalg.norm(b1, axis=1, keepdims=True), 1e-9)
    b2 = np.cross(d, b1)
    ang = 2 * np.pi * np.arange(sides) / sides
    nrm = np.cos(ang)[None, :, None] * b1[:, None] + np.sin(ang)[None, :, None] * b2[:, None]
    v = (pts[:, None] + radius * nrm).reshape(-1, 3).astype(np.float32)
    nrm = nrm.reshape(-1, 3).astype(np.float32)
    i = np.arange(n - 1)[:, None] * sides + np.arange(sides)[None, :]
    j = np.arange(n - 1)[:, None] * sides + (np.arange(sides)[None, :] + 1) % sides
    idx = np.stack([i, j, j + sides, i, j + sides, i + sides], -1).reshape(-1).astype(np.uint32)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", v)
    geo.set_attribute("normal", nrm)
    geo.set_index(idx)
    m = tp.Mesh(geo, mat)
    m.cast_shadow = True
    return m


def corrugated_panel(w, h, n_ribs, depth):
    """BufferGeometry for a corrugated-steel panel in the XY plane (normal +Z).
    Vertical ribs with cosine profile; the specular response sells the cladding. The surface is
    constant in Y (Z and the normal are broadcast from a single row), so two rows are exact."""
    nx = n_ribs * 6 + 1
    xs = np.linspace(-w / 2, w / 2, nx)
    phase = xs * (n_ribs * 2 * np.pi / w)
    z = depth * np.cos(phase)
    dz = -depth * (n_ribs * 2 * np.pi / w) * np.sin(phase)
    inv = 1.0 / np.sqrt(dz ** 2 + 1.0)
    pos = np.stack([np.tile(xs, 2), np.repeat([-h / 2, h / 2], nx), np.tile(z, 2)], -1).astype(np.float32)
    nrm = np.stack([np.tile(-dz * inv, 2), np.zeros(2 * nx), np.tile(inv, 2)], -1).astype(np.float32)
    a = np.arange(nx - 1)
    idx = np.stack([a, a + 1, a + nx + 1, a, a + nx + 1, a + nx], -1).reshape(-1)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", pos)
    geo.set_attribute("normal", nrm)
    geo.set_index(idx.astype(np.uint32))
    return geo


def corr_wall(w, h, mat, pos, rot_y=0.0, depth=0.030, pitch=0.32):
    """A corrugated panel of rib `pitch` metres, rotated about Y and translated."""
    m = tp.Mesh(corrugated_panel(w, h, max(int(w / pitch), 3), depth), mat)
    m.rotation.y = rot_y
    m.position.set(*pos)
    m.cast_shadow = m.receive_shadow = True
    return m


def seg(a, b, r, sides=6):
    """A merged() item: capped cylinder from a to b. +Y is tilted onto the segment by
    rotate_z(-theta) then rotate_y(-phi), which BufferGeometry can express (there is no rotate-axis)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = b - a
    ln = float(np.linalg.norm(d))
    g = tp.CylinderGeometry(r, r, ln, sides, 1)
    g.rotate_z(-math.acos(max(-1.0, min(1.0, d[1] / max(ln, 1e-9)))))
    g.rotate_y(-math.atan2(d[2], d[0]))
    return g, tuple(0.5 * (a + b)), None


def merged(items, mat, shadow=True):
    """One Mesh from (geometry, (x, y, z), (rx, ry, rz) | None) items — collapses a swarm of small
    identical parts (fenders, railings, mullions) into a single draw call. Geometries are consumed."""
    P, N, I, off = [], [], [], 0
    for g, pos, rot in items:
        if rot is not None:
            if rot[0]:
                g.rotate_x(rot[0])
            if rot[1]:
                g.rotate_y(rot[1])
            if rot[2]:
                g.rotate_z(rot[2])
        g.translate(*pos)
        p = np.asarray(g.get_attribute("position"), np.float32).reshape(-1, 3)
        n = np.asarray(g.get_attribute("normal"), np.float32).reshape(-1, 3)
        ix = g.get_index()
        ix = np.arange(len(p), dtype=np.uint32) if ix is None else np.asarray(ix, np.uint32)
        P.append(p)
        N.append(n)
        I.append(ix + off)
        off += len(p)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", np.concatenate(P))
    geo.set_attribute("normal", np.concatenate(N))
    geo.set_index(np.concatenate(I))
    m = tp.Mesh(geo, mat)
    m.cast_shadow = m.receive_shadow = shadow
    return m


def _walk(path, cmds):
    """Feed plan commands into a Path/Shape: (x, z) is a line, ('q', cx, cz, x, z) a quadratic."""
    first = True
    for c in cmds:
        if c[0] == "q":
            path.quadratic_curve_to(c[1], -c[2], c[3], -c[4])
        elif first:
            path.move_to(c[0], -c[1])
        else:
            path.line_to(c[0], -c[1])
        first = False
    path.close_path()


def plan_shape(cmds, holes=()):
    """tp.Shape from a plan outline in barge (x, z). The shape's v axis is -z, so extruding +Z and
    rotating -90 deg about X lands the outline back on (x, z) with the extrusion running +Y."""
    s = tp.Shape()
    _walk(s, cmds)
    if holes:                                  # Shape.holes is a pybind copy: append() no-ops, assign
        hs = []
        for h in holes:
            p = tp.Path()
            _walk(p, h)
            hs.append(p)
        s.holes = hs
    return s


def plan_solid(cmds, y0, thick, mat, holes=(), seg_=6, shadow=True):
    """Solid prism from a plan (x, z) outline: bottom face at y0, top face at y0 + thick.
    steps stays 1 — threepp's ExtrudeGeometry divides depth by (steps * s), so steps > 1 collapses."""
    g = tp.ExtrudeGeometry(plan_shape(cmds, holes), depth=thick, bevel_enabled=False, curve_segments=seg_)
    g.rotate_x(-math.pi / 2)
    g.translate(0.0, y0, 0.0)
    m = tp.Mesh(g, mat)
    m.cast_shadow = m.receive_shadow = shadow
    return m


def offset_plan(pts, d, centre=(-2.8, 0.0)):
    """Offset an open (x, z) polyline d metres away from `centre` (d < 0 pulls it inboard).
    Mitred per edge: an averaged vertex normal pinches inward at a chamfer, and np.gradient skews it
    toward the longer neighbour — that is what tore the glazing open at the bow corners."""
    p = np.asarray(pts, np.float64)
    e = np.diff(p, axis=0)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    en = np.stack([-e[:, 1], e[:, 0]], 1)                        # per-edge left normal
    sg = np.sign(np.sum(en * (0.5 * (p[:-1] + p[1:]) - np.asarray(centre, np.float64)), 1, keepdims=True))
    en *= np.where(sg == 0.0, 1.0, sg)                           # outward, decided per edge
    m = np.empty_like(p)
    m[0], m[-1], m[1:-1] = en[0], en[-1], en[:-1] + en[1:]
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    sc = np.ones((len(p), 1))
    sc[1:-1] = 1.0 / np.maximum(np.sum(m[1:-1] * en[1:], 1, keepdims=True), 0.4)
    return p + d * sc * m


def lift(xz, y):
    """(x, z) polyline -> 3D polyline at height y."""
    xz = np.asarray(xz, np.float64)
    return np.stack([xz[:, 0], np.full(len(xz), y), xz[:, 1]], 1)


def ribbon(lower, upper, mat, shadow=True):
    """Quad strip between two matched 3D polylines. Normals come from (upper - lower) x tangent and
    the winding is chosen to agree with them, so the band survives back-face culling either way."""
    lo, up = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
    n = len(lo)
    nr = np.cross(up - lo, np.gradient(lo, axis=0))
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-9)
    mid = 0.5 * (lo + up)
    if float(np.sum(nr * (mid - mid.mean(0)))) < 0.0:                     # outward, decided once
        nr = -nr
    v = np.concatenate([lo, up]).astype(np.float32)
    nv = np.concatenate([nr, nr]).astype(np.float32)
    i = np.arange(n - 1)
    idx = np.stack([i, i + 1, i + 1 + n, i, i + 1 + n, i + n], -1).reshape(-1)
    a, b, c = v[idx[0]], v[idx[1]], v[idx[2]]
    if float(np.dot(np.cross(b - a, c - a), nv[idx[0]])) < 0.0:           # flip to match the normals
        idx = idx.reshape(-1, 3)[:, ::-1].reshape(-1)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", v)
    geo.set_attribute("normal", nv)
    geo.set_index(idx.astype(np.uint32))
    m = tp.Mesh(geo, mat)
    m.cast_shadow = m.receive_shadow = shadow
    return m


# ---- barge layout ------------------------------------------------------------
# Local frame: +X bow, +Z port, Y up. BARGE_YAW puts _b_port 0.996 radially outward from the pen, so
# every camera in the film sees the STARBOARD (-Z) flank: the walkway recess, the roof notch and the
# long glazing run all live on -Z, and the two inspection shots stand on that quarter too.
HD = 0.85                              # deck top above the hull datum
BX0, BX1, BZ = -8.8, 3.2, 3.45         # superstructure footprint
Y_MID = 3.35                           # upper-floor level: walkway soffit, glazing sill
Y_GLZ = 4.92                           # top of the glazing band
Y_EAVE = 5.15                          # roof underside (the soffit plane)
Y_WALL = 5.24                          # wall top, buried a few cm inside the roof to avoid coplanarity
Y_ROOF = 5.49                          # roof upper surface
COL_X0, COL_X1, COL_Z = -6.9, 1.2, -1.95   # covered walkway: starboard ground floor set inboard
HOSE_N = 11                            # feed hoses in the stern fan
# swept roof plan: deep overhang, a concave notch bitten out of the starboard edge, a wing sweeping
# forward to a blunt prow that cantilevers over the fore deck
ROOF_PLAN = [(-9.55, -4.60), (-6.20, -4.60), (-5.20, -3.30), (-3.20, -3.30), (-2.20, -4.60),
             (1.30, -4.60), (4.45, -3.20), (6.60, -0.45), (6.60, 0.45), (5.00, 2.60), (2.20, 4.60),
             (-9.55, 4.60)]
BOW_CH = 0.75                          # bow-corner chamfer: the wall is faceted, not square
# bridge glazing: long run down starboard, around the faceted bow, short return along port
GLZ_PLAN = [(-4.60, -BZ), (-1.40, -BZ), (1.20, -BZ), (BX1 - BOW_CH, -BZ), (BX1, -BZ + BOW_CH),
            (BX1, -0.90), (BX1, 0.90), (BX1, BZ - BOW_CH), (BX1 - BOW_CH, BZ), (-1.00, BZ)]
BOW_FACETS = [(BX1 - BOW_CH, BZ), (BX1, BZ - BOW_CH), (BX1, -BZ + BOW_CH), (BX1 - BOW_CH, -BZ)]


def build_barge():
    """Norwegian feed barge (forflate): one swept roof plane with a deep overhang and a dark shadow
    gap, raked wrap-around bridge glazing, a covered walkway carved out of the starboard ground
    floor, the dark feed-hall mouth in the stern, and the gantry the hose fan runs over.
    renderer.deferred_ao is off for this demo, so every recess is painted dark rather than shaded."""
    bg = tp.Group()
    # ---- materials ----
    pont_mat  = standard_material(0x080808, roughness=0.82)
    fend_mat  = standard_material(0x0a0a0a, roughness=0.90)
    hull_mat  = standard_material(0x16181a, roughness=0.74)
    deck_mat  = standard_material(0x6c6862, roughness=0.94)
    wall_mat  = standard_material(0xf2eee6, roughness=0.50)     # bright cladding: the white mass
    base_mat  = standard_material(0x24282a, roughness=0.70)
    roof_mat  = standard_material(0xfbf9f4, roughness=0.52)
    soff_mat  = standard_material(0x1b1e20, roughness=0.86)     # roof underside + fascia shadow gap
    win_mat   = standard_material(0x0b1119, roughness=0.06, metalness=0.55)
    frame_mat = standard_material(0x2c2f32, roughness=0.48, metalness=0.60)
    rail_m    = standard_material(0x8e9294, roughness=0.40, metalness=0.88)
    equip_mat = standard_material(0x605e58, roughness=0.62, metalness=0.45)
    red_mat   = standard_material(0xb82020, roughness=0.70)
    pipe_m    = standard_material(0x3a3e42, roughness=0.65)
    door_mat  = standard_material(0x4a525a, roughness=0.55, metalness=0.50)
    dark_mat  = standard_material(0x07090a, roughness=0.95)     # the feed-hall void
    trim_mat  = standard_material(0xdad6ce, roughness=0.45, metalness=0.25)
    L, W = BARGE_L, BARGE_W

    # ================================================================= HULL ===
    for zz in (-W / 2 + 1.2, W / 2 - 1.2):
        p = tp.Mesh(tp.CylinderGeometry(0.58, 0.58, L - 0.4, 14, 1), pont_mat)
        p.rotation.z = math.pi / 2
        p.position.set(0.0, -0.12, zz)
        p.cast_shadow = True
        bg.add(p)
    plating = []
    for zs in (-1, 1):
        plating.append((tp.BoxGeometry(L - 0.3, 0.72, 0.06), (0.0, HD / 2 + 0.01, zs * (W / 2 - 0.03)), None))
    for xs in (-1, 1):
        plating.append((tp.BoxGeometry(0.06, 0.72, W - 0.3), (xs * (L / 2 - 0.03), HD / 2 + 0.01, 0.0), None))
    bg.add(merged(plating, hull_mat))
    # bright rub rail capping the hull — the white deck-edge line that splits the black skirt from the house
    rub = [(tp.BoxGeometry(L, 0.10, 0.14), (0.0, HD - 0.02, zs * (W / 2 + 0.02)), None) for zs in (-1, 1)]
    rub += [(tp.BoxGeometry(0.14, 0.10, W), (xs * (L / 2 + 0.02), HD - 0.02, 0.0), None) for xs in (-1, 1)]
    bg.add(merged(rub, trim_mat))
    bg.add(merged([(tp.BoxGeometry(0.18, 0.14, W - 2.4), (-L / 2 + 1.5 + xi * (L - 3.0) / 6, 0.18, 0.0), None)
                   for xi in range(7)], standard_material(0x1e2022, roughness=0.70, metalness=0.50)))
    # fender skirt: overlapping cylinders the whole way round, one draw call. The strongest dark shape.
    FR = 0.38
    fen = [(tp.CylinderGeometry(FR, FR, 0.66, 9, 1), (-L / 2 + 0.65 + i * (L - 1.3) / 20, 0.06, zs * (W / 2 + 0.06)),
            (math.pi / 2, 0, 0)) for i in range(21) for zs in (-1, 1)]
    fen += [(tp.CylinderGeometry(FR, FR, 0.66, 9, 1), (xs * (L / 2 + 0.06), 0.06, -W / 2 + 0.7 + j * (W - 1.4) / 10),
             (0, 0, math.pi / 2)) for j in range(11) for xs in (-1, 1)]
    bg.add(merged(fen, fend_mat))

    # ================================================================= DECK ===
    dk = tp.Mesh(tp.BoxGeometry(L, 0.10, W), deck_mat)
    dk.position.y = HD - 0.05
    dk.cast_shadow = dk.receive_shadow = True
    bg.add(dk)
    bg.add(merged([(tp.BoxGeometry(L - 0.4, 0.006, 0.6), (0.0, HD + 0.006, zs * (W / 2 - 0.55)), None)
                   for zs in (-1, 1)], standard_material(0x5a5850, roughness=0.98), shadow=False))

    # ======================================================= SUPERSTRUCTURE ===
    # Ground floor: the starboard edge steps inboard between COL_X0..COL_X1, so the upper floor
    # cantilevers over a covered walkway — the deep dark band at deck level on the camera side.
    bg.add(plan_solid([(BX0, BZ)] + BOW_FACETS + [(COL_X1, -BZ), (COL_X1, COL_Z),
                       (COL_X0, COL_Z), (COL_X0, -BZ), (BX0, -2.55), (BX0 + 1.05, -2.55), (BX0 + 1.05, 2.55), (BX0, 2.55)],
                      HD, Y_MID - HD, wall_mat))
    bg.add(plan_solid([(BX0, BZ)] + BOW_FACETS + [(BX0, -2.55), (BX0 + 1.05, -2.55), (BX0 + 1.05, 2.55), (BX0, 2.55)],
                      Y_MID, Y_WALL - Y_MID, wall_mat))
    TRIM = 0.03                                                            # outward, on every face
    bg.add(plan_solid([(BX0 - TRIM, BZ + TRIM), (BX1 - BOW_CH, BZ + TRIM), (BX1 + TRIM, BZ - BOW_CH),
                       (BX1 + TRIM, -BZ + BOW_CH), (BX1 - BOW_CH, -BZ - TRIM), (BX0 - TRIM, -BZ - TRIM)],
                      HD, 0.33, base_mat))                                 # dark base trim
    # walkway: painted-dark back wall, soffit, jambs and floor, plus the columns holding the floor up
    CW, CXM = COL_X1 - COL_X0, (COL_X0 + COL_X1) / 2
    bg.add(merged([(tp.BoxGeometry(CW, Y_MID - HD - 0.30, 0.04), (CXM, (HD + 0.30 + Y_MID) / 2, COL_Z - 0.02), None),
                   (tp.BoxGeometry(CW, 0.05, BZ + COL_Z), (CXM, Y_MID - 0.04, (COL_Z - BZ) / 2), None),
                   (tp.BoxGeometry(CW, 0.04, BZ + COL_Z), (CXM, HD + 0.02, (COL_Z - BZ) / 2), None)]
                  + [(tp.BoxGeometry(0.06, Y_MID - HD, BZ + COL_Z), (x, (HD + Y_MID) / 2, (COL_Z - BZ) / 2), None)
                     for x in (COL_X0 + 0.03, COL_X1 - 0.03)], soff_mat, shadow=False))
    bg.add(merged([(tp.BoxGeometry(0.15, Y_MID - HD - 0.10, 0.15), (x, (HD + Y_MID) / 2, -BZ + 0.11), None)
                   for x in np.linspace(COL_X0 + 0.55, COL_X1 - 0.55, 5)], frame_mat))

    # corrugated cladding, 1 cm proud of the wall solid, on every face a camera can reach
    for w, h, pos, ry in (
            (BX1 - BOW_CH - BX0, Y_MID - HD - 0.33,
             ((BX0 + BX1 - BOW_CH) / 2, (HD + 0.33 + Y_MID) / 2, BZ + 0.01), 0.0),
            (-1.00 - BX0, Y_WALL - Y_MID, ((BX0 - 1.00) / 2, (Y_MID + Y_WALL) / 2, BZ + 0.01), 0.0),
            (COL_X0 - BX0, Y_MID - HD - 0.33, ((BX0 + COL_X0) / 2, (HD + 0.33 + Y_MID) / 2, -BZ - 0.01), math.pi),
            (BX1 - BOW_CH - COL_X1, Y_MID - HD - 0.33,
             ((COL_X1 + BX1 - BOW_CH) / 2, (HD + 0.33 + Y_MID) / 2, -BZ - 0.01), math.pi),
            (-4.60 - BX0, Y_WALL - Y_MID, ((BX0 - 4.60) / 2, (Y_MID + Y_WALL) / 2, -BZ - 0.01), math.pi),
            (2 * (BZ - BOW_CH), Y_MID - HD - 0.33, (BX1 + 0.01, (HD + 0.33 + Y_MID) / 2, 0.0), math.pi / 2),
            (BOW_CH * 1.414, Y_MID - HD - 0.33,
             (BX1 - BOW_CH / 2 + 0.007, (HD + 0.33 + Y_MID) / 2, -BZ + BOW_CH / 2 - 0.007), -math.pi / 4),
            (BOW_CH * 1.414, Y_MID - HD - 0.33,
             (BX1 - BOW_CH / 2 + 0.007, (HD + 0.33 + Y_MID) / 2, BZ - BOW_CH / 2 + 0.007), math.pi / 4)):
        bg.add(corr_wall(w, h, wall_mat, pos, ry))
    bg.add(merged([(tp.BoxGeometry(BX1 - BOW_CH - BX0 - 0.2, 0.022, 0.010),
                    ((BX0 + BX1 - BOW_CH) / 2, y, zs * (BZ + 0.035)), None)
                   for y in (2.05, 4.40) for zs in (-1, 1)],
                  standard_material(0xd8d4cc, roughness=0.46, metalness=0.25), shadow=False))

    # ---- stern: the feed-hall mouth, a rounded-rect hole in the aft face over a deep dark cavity ----
    hall = tp.Shape()                                            # (u, v) = (-z, y), extruded +X
    _walk(hall, [(BZ, HD + 0.02), (-BZ, HD + 0.02), (-BZ, Y_WALL), (BZ, Y_WALL)])
    ha, hb, hc, hd, hr = -2.55, 2.55, HD + 0.45, Y_WALL - 0.75, 0.85
    hole = tp.Path()
    _walk(hole, [(-(ha + hr), hc), (-(hb - hr), hc), ("q", -hb, hc, -hb, hc + hr), (-hb, hd - hr),
                 ("q", -hb, hd, -(hb - hr), hd), (-(ha + hr), hd), ("q", -ha, hd, -ha, hd - hr),
                 (-ha, hc + hr), ("q", -ha, hc, -(ha + hr), hc)])
    hall.holes = [hole]
    hg = tp.ExtrudeGeometry(hall, depth=0.12, bevel_enabled=False, curve_segments=6)
    hg.rotate_y(math.pi / 2)                                     # shape u -> -z, extrusion -> +x
    hg.translate(BX0 - 0.01, 0.0, 0.0)                           # inside the base trim at BX0 - TRIM
    hf = tp.Mesh(hg, wall_mat)
    hf.cast_shadow = hf.receive_shadow = True
    bg.add(hf)
    bg.add(merged([(tp.BoxGeometry(0.06, Y_WALL - HD, 5.10), (BX0 + 1.04, (HD + Y_WALL) / 2, 0.0), None),
                   (tp.BoxGeometry(1.05, Y_WALL - HD, 0.05), (BX0 + 0.52, (HD + Y_WALL) / 2, 2.53), None),
                   (tp.BoxGeometry(1.05, Y_WALL - HD, 0.05), (BX0 + 0.52, (HD + Y_WALL) / 2, -2.53), None),
                   (tp.BoxGeometry(1.05, 0.05, 5.10), (BX0 + 0.52, Y_WALL - 0.03, 0.0), None),
                   (tp.BoxGeometry(1.05, 0.05, 5.10), (BX0 + 0.52, HD + 0.03, 0.0), None)],
                  dark_mat, shadow=False))                       # 1.05 m of cavity behind the mouth
    lin = tp.Mesh(tp.BoxGeometry(0.20, 0.16, hb - ha + 0.5), trim_mat)   # bright lintel over the mouth
    lin.position.set(BX0 - 0.10, hd + 0.13, 0.0)
    bg.add(lin)
    rol = tp.Mesh(tp.CylinderGeometry(0.10, 0.10, hb - ha - 0.5, 8, 1), equip_mat)   # one lit beam inside
    rol.rotation.x = math.pi / 2
    rol.position.set(BX0 + 0.62, hd - 0.35, 0.0)
    bg.add(rol)

    # ---- raked wrap-around bridge glazing ----
    lo_xz, up_xz = offset_plan(GLZ_PLAN, 0.03), offset_plan(GLZ_PLAN, 0.44)
    lo, up = lift(lo_xz, Y_MID + 0.12), lift(up_xz, Y_GLZ)       # top overhangs the sill: the rake
    bg.add(ribbon(lift(offset_plan(GLZ_PLAN, 0.02), Y_MID - 0.06), lo + [0, 0.01, 0], trim_mat))   # sill
    bg.add(ribbon(lo, up, win_mat))
    bg.add(ribbon(up, lift(offset_plan(GLZ_PLAN, 0.46), Y_EAVE), frame_mat))   # dark header to the eave
    bg.add(merged([seg(lo[i] + 0.05 * (up[i] - lo[i]), up[i] + 0.02 * (up[i] - lo[i]), 0.036, 6)
                   for i in range(len(lo))], trim_mat, shadow=False))

    # ---- the roof: one swept plane, deep overhang, dark soffit and shadow gap under a bright fascia ----
    bg.add(plan_solid(ROOF_PLAN, Y_EAVE, 0.23, soff_mat))        # underside + lower fascia (the gap)
    bg.add(plan_solid(ROOF_PLAN, Y_EAVE + 0.21, Y_ROOF - Y_EAVE - 0.21, roof_mat))   # bright cap, 2 cm overlap

    # ================================================= ROOF EQUIPMENT ===
    # the row of dark silo lids across the aft roof — photo B's line of boxes
    bg.add(merged([(tp.BoxGeometry(0.80, 0.28, 1.10), (x, Y_ROOF + 0.14, 1.15), None)
                   for x in np.linspace(-7.10, 0.35, 7)], frame_mat))
    bg.add(merged([(tp.BoxGeometry(1.25, 0.62, 0.85), (-2.6, Y_ROOF + 0.31, -1.75), None),
                   (tp.BoxGeometry(0.92, 0.52, 0.68), (-0.4, Y_ROOF + 0.26, -1.75), None)], equip_mat))
    for xo, yb, mh, mr in ((0.9, Y_ROOF, 3.3, 0.026), (-8.2, Y_ROOF, 2.4, 0.022)):
        ma = tp.Mesh(tp.CylinderGeometry(mr, mr * 0.55, mh, 6, 1), rail_m)
        ma.position.set(xo, yb + mh / 2, 0.0)
        bg.add(ma)
    dome = tp.Mesh(tp.SphereGeometry(0.24, 10, 8), standard_material(0xe6e4e0, roughness=0.30))
    dome.position.set(1.9, Y_ROOF + 0.30, 0.0)
    bg.add(dome)
    bg.add(merged([seg((1.9, Y_ROOF + 0.30, -0.55), (1.9, Y_ROOF + 0.30, 0.55), 0.016, 6),
                   (tp.CylinderGeometry(0.11, 0.11, 1.7, 8, 1), (-7.4, Y_ROOF + 0.85, -1.5), None),
                   (tp.CylinderGeometry(0.15, 0.15, 0.06, 8, 1), (-7.4, Y_ROOF + 1.73, -1.5), None)], hull_mat))

    # ================================================ FORE DECK (bow) ===
    dav_x, dav_h = L / 2 - 2.0, 2.6
    dav = [seg((dav_x + 0.2, HD + 0.05, zs * 0.95), (dav_x - 0.35, HD + dav_h, zs * 0.25), 0.038, 8)
           for zs in (-1, 1)]
    dav += [seg((dav_x - 0.35, HD + dav_h, -0.25), (dav_x - 0.35, HD + dav_h, 0.25), 0.038, 8),
            seg((dav_x - 0.35, HD + dav_h - 0.08, 0.0), (dav_x + 1.9, HD + dav_h + 0.55, 0.0), 0.032, 8),
            (tp.CylinderGeometry(0.18, 0.18, 0.35, 10, 1), (dav_x - 0.15, HD + 0.23, 0.0), (math.pi / 2, 0, 0))]
    bg.add(merged(dav, equip_mat))
    bg.add(merged([(tp.CylinderGeometry(0.40, 0.40, 0.50, 12, 1), (BX1 + 1.3, HD + 0.45, -2.7), (math.pi / 2, 0, 0))]
                  + [(tp.CylinderGeometry(0.52, 0.52, 0.03, 12, 1), (BX1 + 1.3, HD + 0.45, -2.7 + dz),
                      (math.pi / 2, 0, 0)) for dz in (-0.26, 0.26)], equip_mat))
    for xo, w, h, dd, mt in ((1.6, 1.1, 0.85, 0.75, equip_mat), (3.0, 0.55, 0.55, 0.50, red_mat)):
        bx = tp.Mesh(tp.BoxGeometry(w, h, dd), mt)
        bx.position.set(BX1 + xo, HD + h / 2 + 0.05, 2.6)
        bx.cast_shadow = True
        bg.add(bx)
    lr = tp.Mesh(tp.TorusGeometry(0.22, 0.03, 8, 16), standard_material(0xee4400, roughness=0.60))
    lr.rotation.y = math.pi / 2
    lr.position.set(L / 2 - 0.15, HD + 0.75, -2.0)
    bg.add(lr)
    bg.add(merged([(tp.BoxGeometry(0.92, 2.05, 0.04), (xo, HD + 1.06,
                                                       (COL_Z if COL_X0 < xo < COL_X1 else -BZ) - 0.03), None)
                   for xo in (-2.2, 1.4)], door_mat))
    # feed manifold: header + branches on the fore deck
    bg.add(merged([(tp.CylinderGeometry(0.065, 0.065, 2.8, 8, 1), (BX1 + 0.5, HD + 0.52, 0.0), None)]
                  + [(tp.CylinderGeometry(0.05, 0.05, 2.0, 8, 1), (BX1 + 1.6, HD + 0.52, zo), (0, 0, math.pi / 2))
                     for zo in (-1.1, 0.0, 1.1)], pipe_m))

    # ============================================== AFT DECK (stern) ===
    bg.add(merged([(tp.CylinderGeometry(0.08, 0.10, 0.28, 8, 1), (-L / 2 + 0.55, HD + 0.19, zs * 3.0), None)
                   for zs in (-1, 1)], hull_mat))
    cont = tp.Mesh(tp.BoxGeometry(1.5, 1.15, 1.1), standard_material(0x2c4a30, roughness=0.72))
    cont.position.set(BX0 - 1.0, HD + 0.63, -2.6)
    cont.cast_shadow = True
    bg.add(cont)
    # stern hose gantry: the beam the feed fan runs over, with a stub per hose
    GY, GX = HD + 1.15, -L / 2 + 0.45
    gan = [seg((GX, GY, -4.1), (GX, GY, 4.1), 0.075, 8)]
    gan += [seg((GX, HD + 0.05, zs * 3.9), (GX, GY + 0.06, zs * 3.9), 0.055, 8) for zs in (-1, 1)]
    gan += [seg((GX, GY - 0.05, z), (GX - 0.55, HD + 0.30, z), 0.055, 6)
            for z in np.linspace(-3.55, 3.55, HOSE_N)]
    bg.add(merged(gan, equip_mat))

    # ======================================================= RAILINGS ===
    RH, RR, zr = 1.05, 0.018, W / 2 * 0.86
    rails = []

    def rail_run(a, b, n):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        for dy in (RH, RH * 0.48):
            rails.append(seg(a + [0, dy, 0], b + [0, dy, 0], RR, 6))
        for u in np.linspace(0.06, 0.94, n):
            p = a + u * (b - a)
            rails.append(seg(p, p + [0, RH, 0], RR, 6))

    for zs in (-1, 1):
        rail_run((-L / 2 + 0.25, HD, zs * zr), (L / 2 - 0.25, HD, zs * zr), 11)
    rail_run((L / 2 - 0.25, HD, -zr), (L / 2 - 0.25, HD, zr), 3)
    rail_run((-L / 2 + 0.25, HD, -zr), (-L / 2 + 0.25, HD, zr), 3)
    lad_x, lad_z = -3.0, -zr                                     # boarding ladder, camera side
    rails += [seg((lad_x - 0.18, ry, lad_z), (lad_x + 0.18, ry, lad_z), 0.012, 6)
              for ry in np.arange(HD - 0.55, HD + 0.06, 0.30)]
    rails += [seg((lad_x + dx, HD - 0.55, lad_z), (lad_x + dx, HD + RH, lad_z), 0.012, 6)
              for dx in (-0.18, 0.18)]
    bg.add(merged(rails, rail_m))
    return bg


barge = build_barge()
barge.rotation.y = BARGE_YAW
barge.position.set(*BARGE_POS)
scene.add(barge)

# feed fan: hoses leave the stern gantry, sweep astern and run to the pen. Each one is AIMED at its
# own point on the collar arc facing the barge, so it lands there by construction -- an angular fan
# with a reach test instead left the ones that missed running off to nothing.
_hose_mat = standard_material(0x8c9296, roughness=0.55)
_stern = BARGE_POS + (-BARGE_L / 2 - 0.55) * _b_bow
_R_COLLAR = PEN_R + 0.85
_barge_th = math.atan2(BARGE_POS[2], BARGE_POS[0])
for _k in range(HOSE_N):
    _f = _k / (HOSE_N - 1.0)
    _a = _stern + (-3.55 + 7.1 * _f) * _b_port + [0.0, WATER_Y + 0.02, 0.0]
    _th = _barge_th + (_f - 0.5) * 1.10                           # spread across the near collar arc
    _end = np.array([_R_COLLAR * math.cos(_th), WATER_Y + 0.02, _R_COLLAR * math.sin(_th)])
    _mid = _a + 0.42 * (_end - _a) + 0.30 * float(np.linalg.norm(_end - _a)) * -_b_bow
    _u = np.linspace(0, 1, 40)[:, None]
    _pts = (1 - _u) ** 2 * _a + 2 * (1 - _u) * _u * _mid + _u ** 2 * _end
    _pts[:, 1] = WATER_Y + 0.02                                   # ride the surface
    _fl = pipe_along(_pts, 0.05, 6, _hose_mat)
    _fl.frustum_culled = False
    scene.add(_fl)

# mooring buoys
_buoy_mat = standard_material(0xe8c820, roughness=0.65)
_buoys, _buoy_xz = [], []
for _th_b in (1.4, 2.2, 3.0, 3.9, 4.7, 5.5):
    _r_b = PEN_R + 16 + 4 * math.sin(_th_b * 2.3)
    _bx, _bz = _r_b * math.cos(_th_b), _r_b * math.sin(_th_b)
    _b = tp.Mesh(tp.SphereGeometry(0.35, 10, 6), _buoy_mat)
    _b.position.set(_bx, WATER_Y + 0.15, _bz)
    scene.add(_b)
    _buoys.append(_b)
    _buoy_xz.append((_bx, _bz))

# barge wave-following: 4-point hull sample → heave + damped pitch/roll
_bcw = [(BARGE_POS + dx * _b_bow + dz * _b_port)[[0, 2]]
        for dx, dz in ((-8, -3.5), (-8, 3.5), (8, -3.5), (8, 3.5))]


def barge_bob():
    hs = [ocean.sample_height(wx, wz) for wx, wz in _bcw]
    barge.position.y = sum(hs) * 0.25
    lp = math.atan2((hs[2] + hs[3] - hs[0] - hs[1]) * 0.5, 16.0) * 0.30
    lr = math.atan2((hs[1] + hs[3] - hs[0] - hs[2]) * 0.5, 7.0) * 0.30
    barge.rotation.x = lr * _bcy + lp * _bsy
    barge.rotation.z = -lr * _bsy + lp * _bcy
    for _b, (_bx, _bz) in zip(_buoys, _buoy_xz):
        _b.position.y = ocean.sample_height(_bx, _bz) + 0.15


gp = tp.FlockParams()
gp.seed, gp.bird_count, gp.perching, gp.birds_cast_shadow = 4711, 32, False, False
gp.home, gp.roam_radius, gp.cruise_altitude, gp.altitude_spread = tp.Vector3(0.0, 11.0, 0.0), 22.0, 11.0, 0.4
gp.cruise_speed, gp.max_speed, gp.mass_kg = 9.5, 17.0, 0.95
gp.shape.body_length, gp.shape.body_radius, gp.shape.wing_span, gp.shape.tail_fork = 0.58, 0.075, 1.42, 0.15
gp.plumage.back, gp.plumage.belly = tp.Color(0.48, 0.52, 0.57), tp.Color(0.88, 0.88, 0.86)
gp.plumage.cap, gp.plumage.leg, gp.plumage.wingtip_dark = tp.Color(0.80, 0.78, 0.72), tp.Color(0.85, 0.62, 0.22), 0.22
gp.w_cohesion, gp.w_alignment, gp.loner_fraction = 0.30, 0.45, 0.30
gulls = tp.Flock(gp)                                  # parked at y -9000 by transform once the film is under
scene.add(gulls)

# ---- the ROV -----------------------------------------------------------------
foam = standard_material(0xf2c21a, roughness=0.75)
acrylic = tp.MeshPhysicalMaterial()
acrylic.color = 0xffffff
acrylic.roughness = 0.05
acrylic.metalness = 0.0
acrylic.transmission = 0.9
acrylic.ior = 1.49
acrylic.thin_walled = True
acrylic.thickness = 0.004
alu = standard_material(0xb9bec4, roughness=0.35, metalness=0.95)
dark = standard_material(0x15171a, roughness=0.6)
cable = standard_material(0xf7d51d, roughness=0.6)


def side_plate(th=0.006, cell=0.0025):
    """Rounded HDPE plate with round cut-outs as a cell soup (the built pyd predates Shape/Extrude)."""
    hx, hy, rr = 0.20, 0.10, 0.03
    xs = np.arange(-hx, hx - 1e-6, cell)
    ys = np.arange(-hy, hy - 1e-6, cell)
    X, Y = np.meshgrid(xs + 0.5 * cell, ys + 0.5 * cell)
    keep = np.hypot(np.maximum(np.abs(X) - (hx - rr), 0), np.maximum(np.abs(Y) - (hy - rr), 0)) <= rr
    for cx, cy, r in ((-0.02, -0.005, 0.058), (0.135, 0.035, 0.030), (0.135, -0.045, 0.024),
                      (-0.145, 0.035, 0.026), (-0.145, -0.045, 0.026)):
        keep &= np.hypot(X - cx, Y - cy) > r
    P, N = [], []

    def quad(a, b, c, d, n):
        P.append(np.stack([a, b, c, a, c, d], 1).reshape(-1, 3))
        N.append(np.tile(np.float32(n), (len(a) * 6, 1)))

    I, J = np.nonzero(keep)
    xa, ya = xs[J], ys[I]
    xb, yb = xa + cell, ya + cell
    z0, z1 = np.zeros_like(xa), np.full_like(xa, th)
    v = lambda x, y, z: np.stack([x, y, z], 1)
    quad(v(xa, ya, z1), v(xb, ya, z1), v(xb, yb, z1), v(xa, yb, z1), (0, 0, 1))
    quad(v(xa, ya, z0), v(xa, yb, z0), v(xb, yb, z0), v(xb, ya, z0), (0, 0, -1))
    pad = np.pad(keep, 1)
    for di, dj, n in ((0, -1, (-1, 0, 0)), (0, 1, (1, 0, 0)), (-1, 0, (0, -1, 0)), (1, 0, (0, 1, 0))):
        edge = keep & ~pad[1 + di:1 + di + keep.shape[0], 1 + dj:1 + dj + keep.shape[1]]
        I, J = np.nonzero(edge)
        xa, ya = xs[J], ys[I]
        xb, yb = xa + cell, ya + cell
        z0, z1 = np.zeros_like(xa), np.full_like(xa, th)
        if dj: x = xa if dj < 0 else xb; quad(v(x, ya, z0), v(x, yb, z0), v(x, yb, z1), v(x, ya, z1), n)
        else: y = ya if di < 0 else yb; quad(v(xa, y, z0), v(xb, y, z0), v(xb, y, z1), v(xa, y, z1), n)
    g = tp.BufferGeometry()
    g.set_attribute("position", np.concatenate(P).astype(np.float32))
    g.set_attribute("normal", np.concatenate(N).astype(np.float32))
    return g


def thruster(pos, axis):
    g = tp.Group()
    nz = tp.Mesh(tp.CylinderGeometry(0.048, 0.048, 0.052, 24, 1, True), standard_material(0x0d0f12, roughness=0.5, side=tp.Side.Double))
    g.add(nz)
    hub = tp.Mesh(tp.CylinderGeometry(0.015, 0.015, 0.062, 12, 1), dark)
    g.add(hub)
    for k in range(3):
        st = tp.Mesh(tp.BoxGeometry(0.004, 0.006, 0.034), dark)
        st.position.set(0.0, -0.02, 0.031)
        st.rotation.y = 2 * math.pi * k / 3
        arm = tp.Group(); arm.add(st); arm.rotation.y = 2 * math.pi * k / 3
        g.add(arm)
    prop = tp.Group()
    for k in range(3):
        bl = tp.Mesh(tp.BoxGeometry(0.030, 0.0025, 0.013), standard_material(0x2a2e33, roughness=0.45))
        bl.position.set(0.026, 0.0, 0.0)
        bl.rotation.x = math.radians(32)
        arm = tp.Group(); arm.add(bl); arm.rotation.y = 2 * math.pi * k / 3
        prop.add(arm)
    prop.position.y = 0.008
    g.add(prop)
    g.position.set(*pos)
    align_y(g, axis)
    return g, prop


SPOT_I = 1000.0


def build_rov():
    rov = tp.Group()
    props = []
    for z in (-0.165, 0.165):
        pl = tp.Mesh(side_plate(), hdpe)
        pl.position.set(0.0, 0.0, z - 0.003)
        pl.cast_shadow = True
        rov.add(pl)
    top = tp.Mesh(tp.BoxGeometry(0.30, 0.005, 0.33), hdpe)
    top.position.set(0.0, 0.105, 0.0)
    rov.add(top)
    for x in (-0.19, 0.19):
        rov.add(tube((x, -0.095, -0.16), (x, -0.095, 0.16), 0.008, hdpe))
    for z in (-0.105, 0.105):
        fb = tp.Mesh(tp.BoxGeometry(0.30, 0.055, 0.085), foam)
        fb.position.set(0.0, 0.135, z)
        rov.add(fb)
    for y, z, r, L, filler in ((0.035, 0.0, 0.051, 0.30, 0x2b2f38), (-0.06, -0.075, 0.038, 0.26, 0x1f3a6a),
                               (-0.06, 0.075, 0.038, 0.26, 0x1f3a6a)):
        t = tp.Mesh(tp.CylinderGeometry(r, r, L, 32, 1, True), acrylic)
        t.rotation.z = math.pi / 2
        t.position.set(0.0, y, z)
        rov.add(t)
        inner = tp.Mesh(tp.BoxGeometry(L * 0.8, r * 0.9, r * 1.3), standard_material(filler, roughness=0.6))
        inner.position.set(0.0, y - 0.15 * r, z)
        rov.add(inner)
        for x in (-0.5 * L - 0.012, 0.5 * L + 0.012):
            cap = tp.Mesh(tp.CylinderGeometry(r + 0.004, r + 0.004, 0.024, 32, 1), alu)
            cap.rotation.z = math.pi / 2
            cap.position.set(x, y, z)
            rov.add(cap)
    dome = tp.Mesh(tp.SphereGeometry(0.052, 24, 16, -math.pi / 2, math.pi), acrylic)
    dome.rotation.y = math.pi
    dome.position.set(0.15 + 0.024 + 0.003, 0.035, 0.0)
    rov.add(dome)
    cam = tp.Mesh(tp.BoxGeometry(0.03, 0.03, 0.03), dark)
    cam.position.set(0.175, 0.035, 0.0)
    rov.add(cam)
    lens = tp.Mesh(tp.CylinderGeometry(0.009, 0.009, 0.012, 16, 1), standard_material(0x0a0c14, roughness=0.15))
    lens.rotation.z = math.pi / 2
    lens.position.set(0.195, 0.035, 0.0)
    rov.add(lens)
    s2 = 0.5 ** 0.5
    for x, z, ax in ((0.17, -0.20, (s2, 0, -s2)), (0.17, 0.20, (s2, 0, s2)),
                     (-0.17, -0.20, (s2, 0, s2)), (-0.17, 0.20, (s2, 0, -s2))):
        g, p = thruster((x, -0.02, z), ax)
        rov.add(g); props.append(p)
    for z in (-0.165, 0.165):
        g, p = thruster((-0.02, -0.005, z), (0, 1, 0))
        rov.add(g); props.append(p)
    lights = []
    for z in (-0.12, 0.12):
        body = tp.Mesh(tp.CylinderGeometry(0.02, 0.02, 0.045, 16, 1), dark)
        body.rotation.z = math.pi / 2
        body.position.set(0.19, -0.075, z)
        rov.add(body)
        disc = tp.Mesh(tp.CircleGeometry(0.017, 24), standard_material(0xfff4e0, roughness=0.3, emissive=0xfff2d8, emissive_intensity=120.0))
        lights.append(disc.material)
        disc.rotation.y = math.pi / 2
        disc.position.set(0.2135, -0.075, z)
        rov.add(disc)
        spot = tp.SpotLight(tp.Color(1.0, 0.95, 0.85), SPOT_I, 3.5, math.radians(21.0), 0.55, 3.0)   # the cone is gone by ~2.5 m
        spot.position.set(0.215, -0.075, z)
        tgt = tp.Group()
        tgt.position.set(4.0, -0.35, z)
        rov.add(tgt)
        spot.set_target(tgt)
        rov.add(spot)
        lights.append(spot)
    return rov, props, lights


rov, props, rov_lights = build_rov()
rov_pitch = tp.Group()
rov_pitch.add(rov)
rov_yaw = tp.Group()
rov_yaw.add(rov_pitch)
scene.add(rov_yaw)

e_r = np.array([math.cos(TEAR_TH), 0.0, math.sin(TEAR_TH)])
e_t = np.array([-math.sin(TEAR_TH), 0.0, math.cos(TEAR_TH)])
TEAR_C = np.array([PEN_R * e_r[0], TEAR_Y, PEN_R * e_r[2]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---- patrol: closed spline (deg from the tear meridian, stand-off from the net, y) ---------
PATROL_PTS = [(-70, 1.5, -0.5), (-62, 1.5, -1.6), (-50, 1.5, -3.2), (-32, 1.5, -4.2), (-16, 1.8, -3.6),
              (-7, 2.5, -3.15), (0, 1.5, TEAR_Y), (9, 1.5, -2.75), (22, 1.5, -1.9), (34, 1.6, -0.8),
              (5, 3.8, -0.5), (-40, 3.5, -0.6)]
HOVER = TEAR_C - 1.5 * e_r
A_MAX, C_DRAG = 0.3, 0.3 / 0.8 ** 2                   # thrust-limited acceleration; quadratic drag, 0.8 m/s terminal
V_NET, V_OPEN, YAW_RATE_MAX = 0.5, 0.8, math.radians(25.0)
PITCH_CAP, ROLL_CAP, NET_CLEAR = math.radians(6.0), math.radians(3.0), 0.95


def patrol_xyz(deg, off, y):
    th = TEAR_TH + math.radians(deg)
    return np.array([(PEN_R - off) * math.cos(th), y, (PEN_R - off) * math.sin(th)])


def loop_spline(pts, s):
    """Closed Catmull-Rom through pts at fractions s."""
    n = len(pts)
    x = (np.asarray(s, np.float64) % 1.0) * n
    i = np.floor(x).astype(int)
    t = (x - i)[:, None]
    p0, p1, p2, p3 = (pts[(i + k) % n] for k in (-1, 0, 1, 2))
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def catmull(y, u):
    """Catmull-Rom through the control values (scalars or points), sampled at u in [0, 1]."""
    n = len(y)
    x = np.clip(u, 0.0, 1.0) * (n - 1)
    i = np.clip(np.floor(x).astype(int), 0, n - 2)
    t = (x - i)[..., None] if y.ndim > 1 else x - i
    p0, p1, p2, p3 = (y[np.clip(i + k, 0, n - 1)] for k in (-1, 0, 1, 2))
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def build_patrol(dt=1.0 / 60.0):
    """Per-frame rows (x, y, z, yaw, pitch, roll, thrust fwd/lat/vert): a thrust-limited, drag-damped vehicle chasing a carrot on the spline."""
    P = np.array([patrol_xyz(*q) for q in PATROL_PTS])
    fine = loop_spline(P, np.linspace(0, 1, 4001)[:-1])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.roll(fine, -1, 0) - fine, axis=1))])
    L = cum[-1]
    at = lambda s: loop_spline(P, np.interp(np.asarray(s) % L, cum, np.linspace(0, 1, 4001)))
    p, v, k = fine[0].copy(), np.zeros(3), 0
    t0 = fine[8] - fine[0]
    yaw, yaw_rate, pitch, roll = math.atan2(-t0[2], t0[0]), 0.0, 0.0, 0.0
    rows = []
    while k < len(fine) - 1:
        win = fine[k:k + 80]                           # progress = nearest fine sample ahead, monotone
        k += int(np.argmin(np.linalg.norm(win - p, axis=1)))
        carrot, ahead = at([cum[k] + 0.8, cum[k] + 0.3])
        tang = ahead - fine[k]
        tang /= max(np.linalg.norm(tang), 1e-9)
        near = min(np.linalg.norm(p - HOVER) / 2.0, 1.0)
        v_des = (V_NET if np.hypot(p[0], p[2]) > PEN_R - 3.0 else V_OPEN) * (0.12 + 0.88 * near)
        want = carrot - p
        want *= v_des / max(np.linalg.norm(want), 1e-9)
        a_cmd = want - v                                # 1/s velocity loop, then the thrust limit
        n = np.linalg.norm(a_cmd)
        if n > A_MAX:
            a_cmd *= A_MAX / n
        v = v + (a_cmd - C_DRAG * np.linalg.norm(v) * v) * dt
        p = p + v * dt
        r = np.hypot(p[0], p[2])
        if r > PEN_R - 1.2:                             # never closer than 1.2 m to the resting wall; the cloth is handled live
            p[[0, 2]] *= (PEN_R - 1.2) / r
            rad = np.array([p[0], 0.0, p[2]]) / (PEN_R - 1.2)
            v -= max(np.dot(v, rad), 0.0) * rad
        look = tang * near + (TEAR_C - p) * (1 - near)
        dy = (math.atan2(-look[2], look[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate += (max(-YAW_RATE_MAX, min(YAW_RATE_MAX, 1.2 * dy)) - yaw_rate) * (1 - math.exp(-dt / 0.4))
        yaw += yaw_rate * dt
        fwd, right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)]), np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        tf, tl, tv = np.dot(a_cmd, fwd) / A_MAX, np.dot(a_cmd, right) / A_MAX, a_cmd[1] / A_MAX
        # Passively stable hull: trim is the thrust moment against the metacentric restoring moment, lagged 0.6 s.
        pitch += (max(-PITCH_CAP, min(PITCH_CAP, -math.radians(4.0) * tf)) - pitch) * (1 - math.exp(-dt / 0.6))
        roll += (max(-ROLL_CAP, min(ROLL_CAP, math.radians(3.0) * tl)) - roll) * (1 - math.exp(-dt / 0.6))
        rows.append([*p, yaw, pitch, roll, tf, tl, tv])
    return np.array(rows)


PATROL = build_patrol()
T_LOOP = len(PATROL) / 60.0
_hover_i = int(np.argmin(np.linalg.norm(PATROL[:, :3] - HOVER, axis=1)))
_wall_i = int(np.argmin(np.linalg.norm(PATROL[:, :3] - patrol_xyz(-42, 1.5, -3.8), axis=1)))
PATROL_T0 = {"p3_sonar_tear": _hover_i / 60.0 - SECONDS, "p3_hud": _wall_i / 60.0 - SECONDS}.get(SHOT, 0.0) % T_LOOP
rov_pos, rov_R, rov_thrust = PATROL[0, :3].copy(), np.eye(3), np.zeros(3)


T_WALL = _wall_i / 60.0
ROV_HOLD = [0.0]                                      # film: seconds the ROV holds station at the wall while the camera sweeps
LIFT = [None]                                         # film: crane path (walkway -> water) for negative patrol time


def patrol_clock(t):
    """Patrol time -> baked row time: the descent, then a station hold, then the wall run."""
    t = max(t, 0.0)
    return t if t < T_WALL else max(t - ROV_HOLD[0], T_WALL)


def lift_at(t):
    return LIFT[0](t + FILM_T_OFF) - PATROL[0, :3] if LIFT[0] is not None and t < 0.0 else np.zeros(3)


rov_off = np.zeros(3)                                 # live push-back off the sagging cloth
net_host = [None]                                     # last downloaded cloth positions (net_upload)
ROV_STATS = {"min_d": float("inf"), "max_roll": 0.0, "max_pitch": 0.0}


def net_nearest(p):
    """Nearest cloth particle: distance, position, local sheet normal."""
    pos = net_host[0] if net_host[0] is not None else net.pos.numpy()
    j = int(np.argmin(((pos - p) ** 2).sum(1)))
    iv, iu = divmod(j, NU)
    n = np.cross(pos[iv * NU + (iu + 1) % NU] - pos[iv * NU + (iu - 1) % NU],
                 pos[min(iv + 1, NVT - 1) * NU + iu] - pos[max(iv - 1, 0) * NU + iu])
    return float(np.linalg.norm(p - pos[j])), pos[j], n / max(np.linalg.norm(n), 1e-9)


_TEAR_RING = np.nonzero((net.cls == -1)
                        & (np.linalg.norm(net.rest_host - TEAR_C, axis=1) > 1.2)
                        & (np.linalg.norm(net.rest_host - TEAR_C, axis=1) < 2.2))[0]


# the ring's centroid sits 0.11 m inside the wall (a chord across 620 particles); take that
# constant out so tear_now() is exactly TEAR_C at rest and only reports the real drift
_TEAR_BIAS = TEAR_C - net.rest_host[_TEAR_RING].mean(0) if len(_TEAR_RING) else np.zeros(3)


def tear_now():
    """Where the hole IS: the mean of the intact twine ringing it, on the deformed wall.
    TEAR_C is the rest position and the wall bows ~1 m off it under the current."""
    pos = net_host[0]
    return TEAR_C if pos is None or not len(_TEAR_RING) else pos[_TEAR_RING].mean(0) + _TEAR_BIAS


def patrol_row(t):
    """The baked patrol at an arbitrary time, LERPed between the two bracketing 60 Hz rows.
    `PATROL[int(clock * 60)]` snapped the whole pose to whichever row the truncation landed
    on, so every frame whose clock did not fall exactly on a bake step repeated or skipped
    one -- the judder on the way down. Same fix as the car: blend the states either side
    (warp_mudsnow_drive.py's pose_prev/pose_cur + nlerp) instead of picking one."""
    f = patrol_clock(t) * 60.0
    i0 = int(f) % len(PATROL)
    r0, r1 = PATROL[i0], PATROL[(i0 + 1) % len(PATROL)]
    a = f - math.floor(f)
    r = r0 + (r1 - r0) * a
    for k in (3, 4, 5):                                 # yaw/pitch/roll: never the long way round
        r[k] = r0[k] + ((r1[k] - r0[k] + math.pi) % (2.0 * math.pi) - math.pi) * a
    return r


def rov_pose(t, dt=1.0 / 60.0):
    global rov_pos, rov_R, rov_thrust, rov_off
    x, y, z, yaw, pitch, roll, tf, tl, tv = patrol_row(t)
    base = np.array([x, y, z]) + lift_at(t)
    cand = base + rov_off
    d, q, _ = net_nearest(cand)
    if d < NET_CLEAR:                                   # push along the line from the nearest particle,
        # ramped: the nearest-particle argmin flips as the hull slides down the wall, and an
        # instant correction turns each flip into a visible sideways step
        rov_off += (NET_CLEAR - d) / max(d, 1e-6) * (cand - q) * (1.0 - math.exp(-dt / 0.15))
        cand = base + rov_off
        d = net_nearest(cand)[0]
    else:
        rov_off *= math.exp(-dt / 2.0)
    if T_WALL <= t < T_WALL + ROV_HOLD[0]:              # station keeping: idle thrust
        tf, tl, tv = 0.2 * tf, 0.2 * tl, 0.2 * tv
    ROV_STATS["min_d"] = min(ROV_STATS["min_d"], d)
    ROV_STATS["max_roll"] = max(ROV_STATS["max_roll"], abs(roll))
    ROV_STATS["max_pitch"] = max(ROV_STATS["max_pitch"], abs(pitch))
    rov_yaw.position.set(*cand)
    rov_yaw.rotation.y = yaw
    rov_pitch.rotation.z = pitch
    rov.rotation.x = roll
    rov_pos, rov_R, rov_thrust = cand, rot_y(yaw) @ rot_z(pitch) @ rot_x(roll), np.array([tf, tl, tv])
    wet = min(max((WATER_Y - 0.08 - cand[1]) / 0.25, 0.0), 1.0)           # lamps come on as the ROV submerges
    for k, l in enumerate(rov_lights):
        if k % 2:
            l.intensity = SPOT_I * wet
        else:
            l.emissive_intensity = 120.0 * wet


def rov_report():
    print(f"ROV: min net distance {ROV_STATS['min_d']:.2f} m, max |roll| {math.degrees(ROV_STATS['max_roll']):.1f} deg, "
          f"max |pitch| {math.degrees(ROV_STATS['max_pitch']):.1f} deg")


rov_pose(PATROL_T0)

# ---- tether: rope from the tail to a surface point on the collar that trails the ROV ------------
TETHER_LOCAL = np.array([-0.19, 0.11, 0.0])
tether_th = math.atan2(rov_pos[2], rov_pos[0]) - 0.2


def tether_pins(dt):
    global tether_th
    th = math.atan2(rov_pos[2], rov_pos[0])
    tether_th += ((th - tether_th + math.pi) % (2 * math.pi) - math.pi) * (1 - math.exp(-dt / 8.0))
    top = np.array([(PEN_R - 0.35) * math.cos(tether_th - 0.15), WATER_Y - 0.05, (PEN_R - 0.35) * math.sin(tether_th - 0.15)])
    return rov_pos + rov_R @ TETHER_LOCAL, top


def tube_verts(pts, r, sides):
    """Ring vertices + normals along a polyline (no twist control; fine at r = 6 mm)."""
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    a = np.where(np.abs(d[:, 1:2]) < 0.9, np.float64([[0, 1, 0]]), np.float64([[1, 0, 0]]))
    b1 = np.cross(d, a)
    b1 /= np.maximum(np.linalg.norm(b1, axis=1, keepdims=True), 1e-9)
    b2 = np.cross(d, b1)
    ang = 2 * np.pi * np.arange(sides) / sides
    n = np.cos(ang)[None, :, None] * b1[:, None] + np.sin(ang)[None, :, None] * b2[:, None]
    return (pts[:, None] + r * n).reshape(-1, 3).astype(np.float32), n.reshape(-1, 3).astype(np.float32)


def tube_index(n, sides):
    i = np.arange(n - 1)[:, None] * sides + np.arange(sides)[None, :]
    j = np.arange(n - 1)[:, None] * sides + (np.arange(sides)[None, :] + 1) % sides
    return np.stack([i, j, j + sides, i, j + sides, i + sides], -1).reshape(-1).astype(np.uint32)


TETHER_U = np.linspace(0.0, 1.0, 4 * (ROPE_N - 1) + 1)          # spline through the particles: no segment kinks
_ta, _tb = tether_pins(0.0)
rope.reset(_ta, _tb)
tether_geo = tp.BufferGeometry()
_tv, _tn = tube_verts(catmull(rope.pos.numpy().astype(np.float64), TETHER_U), 0.006, 8)
tether_geo.set_attribute("position", _tv)
tether_geo.set_attribute("normal", _tn)
tether_geo.set_index(tube_index(len(TETHER_U), 8))
tether = tp.Mesh(tether_geo, cable)
tether.frustum_culled = False
scene.add(tether)


def tether_upload():
    v, n = tube_verts(catmull(rope.pos.numpy().astype(np.float64), TETHER_U), 0.006, 8)
    tether_geo.update_attribute("position", v)
    tether_geo.update_attribute("normal", n)


# ---- thruster bubbles: a host-fed billboard field, ring-buffered -------------------------------
BUB_CAP = 512
_bc = tp.ParticleField.Config()
_bc.capacity = BUB_CAP
_bc.ownership = tp.ParticleField.Ownership.HostRing
_bc.w_semantic = tp.ParticleField.WSemantic.Radius
_bc.uniform_radius = 0.004
_bc.host_stable_slots = True
bubbles = tp.ParticleField.create(_bc)
bubbles.frustum_culled = False
bubbles.set_billboard_repr(tp.Color(0.85, 0.95, 1.0), tp.Color(0.50, 0.72, 0.78), 0.9, 1.0)
_bb = bubbles.billboard_repr
_bb.lod_near = _bb.lod_fade = _bb.stretch_seconds = _bb.size_taper = _bb.glow = 0.0
_bb.softness = 0.45
_bb.fade_power = 1.2
_bb.near_fade = 0.1
bubbles.set_live_count(0)
scene.add(bubbles)
REAR_THR = np.array([[-0.22, -0.02, -0.23], [-0.22, -0.02, 0.23]])
bub_p, bub_v = np.zeros((BUB_CAP, 3)), np.zeros((BUB_CAP, 3))
bub_age, bub_life, bub_r = np.full(BUB_CAP, 1e9), np.ones(BUB_CAP), np.full(BUB_CAP, 0.004)
bub_buf = np.zeros((BUB_CAP, 4), np.float32)
bub_rng, bub_next, bub_acc = np.random.default_rng(31), 0, 0.0


def bubbles_step(dt):
    global bub_next, bub_acc
    push = min(max((rov_thrust[0] - 0.25) / 0.75, 0.0), 1.0)
    bub_acc += 70.0 * push * dt
    k, bub_acc = int(bub_acc), bub_acc % 1.0
    fwd = rov_R @ [1.0, 0.0, 0.0]
    for m in range(k):
        i, bub_next = bub_next, (bub_next + 1) % BUB_CAP
        bub_p[i] = rov_pos + rov_R @ (REAR_THR[m % 2] + bub_rng.normal(0, 0.012, 3))
        bub_v[i] = -fwd * 0.35 + bub_rng.normal(0, 0.05, 3)
        bub_age[i], bub_life[i], bub_r[i] = 0.0, bub_rng.uniform(1.4, 2.4), bub_rng.uniform(0.0015, 0.005)
    bub_age[:] += dt
    bub_v[:, 1] += (0.22 - bub_v[:, 1]) * dt / 0.5
    bub_v[:, [0, 2]] *= math.exp(-dt / 0.6)
    bub_p[:] += bub_v * dt
    alive = (bub_age < bub_life) & (bub_p[:, 1] < WATER_Y - 0.05)   # nothing airborne
    bub_buf[:, :3] = bub_p
    bub_buf[:, 3] = np.where(alive, bub_r * (1 + 0.4 * bub_age / bub_life), -1.0)
    bubbles.submit(bub_buf, dt)



# ---- fish: procedural salmon v2 ---------------------------------------------
FISH_N = max(cli_arg("--fish", 400, int), 2)
F_RINGS, F_SIDES = 24, 24
F_K = 5.6
FISH_SPEED_MIN, FISH_SPEED_MAX = 0.25, 1.6
MILL_SPEED = 0.75
LEAK_FRAC = 0.05
LEAK_T0 = [0.0]                                       # world time the leak schedule starts (the film sets it)
EYE_U, EYE_TH = 0.085, 0.25
TEX_W, TEX_H, BODY_ROWS = 1024, 512, 392
BODY_V = BODY_ROWS / TEX_H
FIN_BANDS = {k: ((402 + 36 * k) / TEX_H, (434 + 36 * k) / TEX_H) for k in range(3)}   # dark fins / pectoral / pelvic+anal


HALF_H = np.float32([0.024, 0.060, 0.082, 0.096, 0.106, 0.106, 0.099, 0.086, 0.069, 0.049, 0.030, 0.014])
HALF_W = np.float32([0.016, 0.040, 0.051, 0.059, 0.063, 0.061, 0.056, 0.048, 0.038, 0.027, 0.016, 0.006])
KEEL = np.float32([1.0, 1.05, 1.15, 1.22, 1.25, 1.24, 1.20, 1.14, 1.08, 1.02, 1.0, 1.0])


def skin_fn(u, th):
    """The skin at body fraction u (0 nose, 1 tail root) and ring angle th; +Z is the nose."""
    h, w, kb = catmull(HALF_H, u), catmull(HALF_W, u), catmull(KEEL, u)
    s, c = np.sin(th), np.cos(th)
    eye = 1.0 + 0.10 * np.exp(-((u - EYE_U) / 0.035) ** 2) * (np.exp(-(1 - np.cos(th - EYE_TH)) / 0.05)
                                                              + np.exp(-(1 - np.cos(th + EYE_TH - np.pi)) / 0.05))
    jaw = 1.0 + 0.30 * np.exp(-((u - 0.05) / 0.035) ** 2) * np.clip(-s, 0, 1) ** 2     # lower jaw lobe
    y = h * s * np.where(s < 0, kb * (1.0 - 0.18 * c * c) * jaw, 1.0) * eye - 0.15 * h
    return np.stack([w * c * eye, y, 0.5 - u], -1)


def fan(outline, nrm, ray, camber, shift=(0.0, 0.0, 0.0), band=0, flip=False):
    """A thin fin fanned from its centre, cambered along nrm; uv = (across the rays, root->tip) in the fin band."""
    outline = np.asarray(outline, np.float64) + np.float64(shift)
    v = np.concatenate([[outline.mean(0)], outline])
    tris = [[0, 1 + k, 1 + (k + 1) % len(outline)] for k in range(len(outline))]
    nrm, ray = np.float64(nrm) / np.linalg.norm(nrm), np.float64(ray) / np.linalg.norm(ray)
    perp = np.cross(nrm, ray)
    perp /= np.linalg.norm(perp)
    rel = v - outline[0]
    al, ac = rel @ ray, rel @ perp
    la, lc = max(al.max() - al.min(), 1e-6), max(ac.max() - ac.min(), 1e-6)
    al, ac = (al - al.min()) / la, (ac - ac.min()) / lc
    v = v + (camber * 4 * ac * (1 - ac) * al)[:, None] * nrm
    n = nrm - np.outer(camber * 4 * (1 - 2 * ac) * al / lc, perp) - np.outer(camber * 4 * ac * (1 - ac) / la, ray)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    v0, v1 = FIN_BANDS[band]
    return v, np.asarray(tris), n, np.stack([1 - ac if flip else ac, v0 + (v1 - v0) * al], 1)


def salmon():
    """Canonical unit-length salmon: folded/flared positions and normals, uv, body fraction u, kind, index."""
    S1 = F_SIDES + 1                                # ring closes on a duplicated belly vertex (uv seam)
    u = np.linspace(0.0, 1.0, F_RINGS) ** 0.85
    th = -np.pi / 2 + 2.0 * np.pi * np.arange(S1) / F_SIDES
    U, TH = np.meshgrid(u, th, indexing="ij")
    P = skin_fn(U, TH)
    e = 1e-3
    du = skin_fn(np.clip(U + e, 0, 1), TH) - skin_fn(np.clip(U - e, 0, 1), TH)
    dt = skin_fn(U, TH + e) - skin_fn(U, TH - e)
    N = np.cross(du, dt)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    V = np.broadcast_to((np.arange(S1) / F_SIDES * BODY_V)[None, :], U.shape)
    pos, nrm, uu, kind = [P.reshape(-1, 3)], [N.reshape(-1, 3)], [U.reshape(-1)], [np.zeros(U.size)]
    uv = [np.stack([U, V], -1).reshape(-1, 2)]
    tris = []
    for i in range(F_RINGS - 1):
        for j in range(F_SIDES):
            a = i * S1 + j
            tris += [[a, a + S1, a + 1], [a + 1, a + S1, a + S1 + 1]]
    nv = F_RINGS * S1
    pos.append([[0.0, -0.15 * HALF_H[0], 0.505], [0.0, -0.15 * HALF_H[-1], -0.5]])
    nrm.append([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    uu.append([0.0, 1.0])
    kind.append([0, 0])
    uv.append([[0.0, 0.5 * BODY_V], [1.0, 0.5 * BODY_V]])
    for j in range(F_SIDES):
        tris.append([nv, j + 1, j])
        tris.append([nv + 1, nv - S1 + j, nv - S1 + j + 1])
    nv += 2

    def add(f, k):
        nonlocal nv
        v, t, n, tuv = f
        pos.append(v)
        nrm.append(n)
        uv.append(tuv)
        tris.extend((np.asarray(t) + nv).tolist())
        uu.append(0.5 - v[:, 2])
        kind.append(np.full(len(v), k))
        nv += len(v)

    top = lambda x: float(skin_fn(x, np.pi / 2)[1])
    bot = lambda x: float(skin_fn(x, -np.pi / 2)[1])
    z = lambda x: 0.5 - x
    add(fan([[0, 0.012, -0.46], [0, 0.06, -0.55], [0, 0.115, -0.65], [0, 0.13, -0.70], [0, 0.07, -0.675],
             [0, 0.02, -0.66], [0, -0.02, -0.66], [0, -0.07, -0.675], [0, -0.125, -0.70], [0, -0.11, -0.65],
             [0, -0.055, -0.55], [0, -0.016, -0.46]], [1, 0, 0], [0, 0, -1], 0.006), 1)
    add(fan([[0, top(x) + h, z(x)] for x, h in ((0.45, 0), (0.455, 0.009), (0.468, 0.015), (0.49, 0.017), (0.52, 0.013),
                                                (0.55, 0.006), (0.565, 0), (0.51, 0))], [1, 0, 0], [0, 1, -0.5], 0.004), 1)
    add(fan([[0, top(x) + h, z(x)] for x, h in ((0.80, 0), (0.81, 0.006), (0.825, 0.008), (0.84, 0.003), (0.845, 0))],
            [1, 0, 0], [0, 1, -0.3], 0.001), 1)
    add(fan([[0, bot(x) + h, z(x)] for x, h in ((0.75, 0), (0.76, -0.016), (0.78, -0.024), (0.81, -0.012),
                                                (0.825, -0.002), (0.79, 0))], [1, 0, 0], [0, -1, -0.5], 0.003, band=2), 1)
    for sgn in (1.0, -1.0):
        r = [sgn * 0.014, bot(0.55) + 0.004, z(0.55)]
        add(fan([[0, 0, 0], [sgn * 0.012, -0.012, -0.024], [sgn * 0.024, -0.024, -0.05], [sgn * 0.009, -0.021, -0.046],
                 [0, -0.005, -0.015]], [0.3 * sgn, -1, 0], [sgn * 0.4, -0.45, -1], 0.003, r, band=2, flip=sgn > 0), 1)
    for sgn in (1.0, -1.0):
        r = skin_fn(0.235, -0.3 * np.pi if sgn > 0 else 1.3 * np.pi) + [sgn * 0.003, 0, 0]
        add(fan([[0, 0, 0], [sgn * 0.006, -0.006, -0.03], [sgn * 0.012, -0.012, -0.078], [sgn * 0.010, -0.028, -0.064],
                 [sgn * 0.004, -0.022, -0.018]], [sgn * 0.8, -0.6, 0], [sgn * 0.1, -0.2, -1], -0.003, r, band=1), 2 if sgn > 0 else 3)
    pos, nrm, uv = np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv)
    uu, kind = np.concatenate(uu), np.concatenate(kind).astype(np.int32)
    pos1, nrm1 = pos.copy(), nrm.copy()
    for sgn, k in ((1.0, 2), (-1.0, 3)):           # flared pectorals: swept out and laid flat
        sel = kind == k
        r = pos[sel][1]
        R = rot_z(-sgn * 0.7) @ rot_y(sgn * 1.05)
        pos1[sel] = (pos[sel] - r) @ R.T + r
        nrm1[sel] = nrm[sel] @ R.T
    return (pos.astype(np.float32), pos1.astype(np.float32), nrm.astype(np.float32), nrm1.astype(np.float32),
            uv.astype(np.float32), uu.astype(np.float32), kind, np.asarray(tris, np.uint32).reshape(-1))


def hash01(*ints):
    h = np.zeros(np.broadcast(*ints).shape, np.uint32)
    for k, a in enumerate(ints):
        h = (h ^ (np.asarray(a, np.uint32) * np.uint32([0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D][k % 3]))) * np.uint32(0x27D4EB2F)
        h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)) / float(0xFFFFFF)


def srgb8(lin):
    c = np.clip(lin, 0, 1)
    return (np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


def fish_albedo():
    """(TEX_H, TEX_W, 3) linear RGB + (TEX_H, TEX_W, 4) normal map: body band (u nose->tail, v belly->back->belly), fin bands."""
    rng = np.random.default_rng(7)
    U, V = np.meshgrid((np.arange(TEX_W) + 0.5) / TEX_W, (np.arange(BODY_ROWS) + 0.5) / BODY_ROWS)
    px, py = U * TEX_W, V * BODY_ROWS
    d = np.abs(V - 0.5) / 0.5                          # 0 back, 1 belly
    rows = BODY_ROWS / 2                               # rows per unit d
    ysc = 0.55 * TEX_W / BODY_ROWS                     # rows -> x-texel units, so shapes are round on the fish
    n = fbm(*grid(BODY_ROWS, TEX_W, 4, 12), 4, 12, rng, 4) - 0.5
    de = d + 0.04 * n - 0.06 * (1 - np.clip((U - 0.12) / 0.10, 0, 1))          # dark top reaches lower on the head
    sm = lambda x, a, b: np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))
    back, flank, belly = np.float32([0.03, 0.05, 0.06]), np.float32([0.33, 0.37, 0.39]), np.float32([0.64, 0.66, 0.64])
    t1, t2 = sm(de, 0.30, 0.40)[..., None], sm(de, 0.62, 0.88)[..., None]
    col = (back * (1 - t1) + flank * t1) * (1 - t2) + belly * t2
    g = np.clip((de - 0.35) / 0.4, 0, 1)[..., None]    # blue -> purple -> copper sheen down the flank
    sheen = (1 - g) ** 2 * np.float32([0.94, 0.97, 1.06]) + 2 * g * (1 - g) * np.float32([1.02, 0.95, 1.04]) + g ** 2 * np.float32([1.06, 0.98, 0.93])
    col = col * (1 + 0.5 * (sheen - 1) * t1 * (1 - t2)) * (1 + 0.08 * n[..., None] * np.float32([1.0, 1.1, 0.8]))
    col *= 1 - 0.12 * np.exp(-((d - 0.46) * rows / 1.6) ** 2)[..., None] * (U > 0.21)[..., None]      # lateral line
    ue = 0.19 + 0.03 * np.sin(np.pi * np.clip((d - 0.05) / 0.9, 0, 1))                                 # gill cover rear edge
    ge = (U - ue) * TEX_W
    scl = np.clip((U - ue) / 0.03, 0, 1)                # scales start behind the gill cover
    a1, a2 = px / 11.4 + py * ysc / 11.4, px / 11.4 - py * ysc / 11.4
    ta, tb = np.abs(a1 % 1 - 0.5), np.abs(a2 % 1 - 0.5)
    edge = np.exp(-(np.minimum(ta, tb) / 0.09) ** 2)
    col *= (1 - 0.16 * edge * scl)[..., None]                                                          # ~90 diamond scales
    hd = (1 - sm(U, ue - 0.02, ue + 0.02))[..., None]                                                 # head weight
    col = col * (1 - hd) + col * np.float32([0.84, 0.88, 0.87]) * hd
    op = sm(U, 0.10, 0.15) * (1 - sm(U, ue - 0.015, ue + 0.01)) * (d > 0.12)                          # operculum plate
    col *= (1 + 0.16 * op * (0.6 + 0.4 * sm(d, 0.15, 0.5)))[..., None]
    col *= (1 - 0.14 * np.exp(-((ge - 5.0) / 7.0) ** 2) * (ge > -2) * (d > 0.12))[..., None]           # soft shade behind it
    for sg in (1, -1):
        vm = 0.5 + sg * (0.30 + 0.06 * U / 0.125)
        m = np.exp(-((V - vm) * BODY_ROWS / 3.2) ** 2) * sm(U, 0.015, 0.07) * (1 - sm(U, 0.10, 0.13))
        col *= (1 - 0.40 * m)[..., None]                                                               # mouth seam
        jw = sm(sg * (V - 0.5), 0.27 + 0.06 * U / 0.125, 0.35 + 0.06 * U / 0.125) * (1 - sm(U, 0.09, 0.13))
        col = col * (1 - 0.35 * jw)[..., None] + np.float32([0.58, 0.47, 0.45]) * (0.35 * jw)[..., None]   # lower jaw
        r = np.hypot(px - 0.055 * TEX_W, (py - (0.5 - sg * 0.20) * BODY_ROWS) * ysc)
        col *= (1 - 0.6 * np.clip((2.2 - r) * 0.8, 0, 1))[..., None]                                   # nostril
    X, Y = px, py * ysc
    cov = np.zeros_like(U)
    rim = fbm(*grid(BODY_ROWS, TEX_W, 32, 128), 32, 128, rng, 3) - 0.5
    for dx in (-1, 0, 1):                              # sparse irregular black dots, back and upper flank
        for dy in (-1, 0, 1):
            cx, cy = 48, 24
            jx, jy = (px // cx).astype(int) + dx + 8, (py // cy).astype(int) + dy + 8
            sx = (jx - 8 + 0.15 + 0.7 * hash01(jx, jy, 1)) * cx
            sy = (jy - 8 + 0.15 + 0.7 * hash01(jx, jy, 2)) * cy
            su, sd = sx / TEX_W, np.abs(sy / BODY_ROWS - 0.5) / 0.5
            p = 0.30 * (1 - sm(sd, 0.34, 0.46)) * (su > ue.mean())
            on = hash01(jx, jy, 3) < p
            r0 = 4.0 + 5.0 * hash01(jx, jy, 5)                # <= ~2 scales across
            ex = 0.8 + 0.4 * hash01(jx, jy, 8)
            r = np.hypot((X - sx) * ex, (Y - sy * ysc) / ex) * (1 + 0.5 * rim)
            cov = np.maximum(cov, np.clip((r0 - r) * 0.7 + 0.5, 0, 1) * on)
    for dx in (-1, 0, 1):                              # round dots on the gill cover / head; speckles on top
        for dy in (-1, 0, 1):
            cx, cy = 22, 14
            jx, jy = (px // cx).astype(int) + dx + 8, (py // cy).astype(int) + dy + 8
            sx = (jx - 8 + 0.2 + 0.6 * hash01(jx, jy, 11)) * cx
            sy = (jy - 8 + 0.2 + 0.6 * hash01(jx, jy, 12)) * cy
            su, sd = sx / TEX_W, np.abs(sy / BODY_ROWS - 0.5) / 0.5
            eye_far = np.hypot((su - EYE_U) * TEX_W / 16.4, (sd - 0.42) * rows / 11.4) > 2.0
            dot = (hash01(jx, jy, 13) < 0.22) & (su > 0.11) & (su < ue.mean()) & (sd > 0.2) & (sd < 0.62) & eye_far
            spk = (hash01(jx, jy, 14) < 0.5) & (su > 0.03) & (su < ue.mean() + 0.02) & (sd < 0.22)
            r = np.hypot(X - sx, Y - sy * ysc)
            cov = np.maximum(cov, np.clip((3.2 - r) * 1.2 + 0.5, 0, 1) * dot)
            cov = np.maximum(cov, np.clip((1.3 - r) * 1.5 + 0.5, 0, 1) * spk)
    col = col * (1 - 0.95 * cov[..., None]) + np.float32([0.012, 0.015, 0.018]) * (0.95 * cov)[..., None]
    for sg in (1, -1):
        r = np.hypot((px - EYE_U * TEX_W) / 16.4, (py - (0.5 - sg * 0.21) * BODY_ROWS) / 11.4)
        disc = lambda r0: np.clip((r0 - r) * 3.5, 0, 1)[..., None]
        col *= 1 - 0.22 * (disc(1.55) - disc(1.05))
        col = col * (1 - disc(1.12)) + np.float32([0.03, 0.035, 0.04]) * disc(1.12)
        col = col * (1 - disc(0.95)) + (np.float32([0.40, 0.29, 0.09]) * (0.55 + 0.6 * r)[..., None]) * disc(0.95)
        col = col * (1 - disc(0.52)) + np.float32([0.008, 0.008, 0.01]) * disc(0.52)
    tex = np.tile(np.float32([0.05, 0.06, 0.065]), (TEX_H, TEX_W, 1))
    tex[:BODY_ROWS] = col
    fa, fb = a1 % 1 - 0.5, a2 % 1 - 0.5                    # in-scale coords; +fa+fb points to the tail
    ca, cb = np.floor(a1).astype(int) + 64, np.floor(a2).astype(int) + 64
    cup = -1.3 * np.clip(1 - (fa * fa + fb * fb) / 0.25, 0, 1)
    ridge = 0.9 * np.clip((fa + fb - 0.22) / 0.2, 0, 1) * np.clip((0.5 - np.maximum(np.abs(fa), np.abs(fb))) / 0.1, 0, 1)
    tilt = 11.4 * 0.052 * (fa * (hash01(ca, cb, 21) - 0.5) + fb * (hash01(ca, cb, 22) - 0.5)) * 2.0     # 2-4 deg per scale
    hgt = np.zeros((TEX_H, TEX_W), np.float32)
    hgt[:BODY_ROWS] = (cup + ridge + tilt) * scl
    rough = np.full((TEX_H, TEX_W), 0.5, np.float32)
    rough[:BODY_ROWS] = 0.42 * (1 - scl) + (0.30 + 0.25 * edge) * scl
    ac = (np.arange(TEX_W) + 0.5) / TEX_W
    ray = np.exp(-(((ac * 11) % 1 - 0.5) * TEX_W / 11 / 3.5) ** 2)[None, :]
    for k, (v0, v1) in FIN_BANDS.items():
        r0, r1 = int(v0 * TEX_H) - 2, int(v1 * TEX_H) + 2
        al = np.clip((np.arange(r1 - r0) + 0.5 - 2) / (r1 - r0 - 4), 0, 1)[:, None]
        n2 = fbm(*grid(r1 - r0, TEX_W, 1, 16), 1, 16, rng, 3) - 0.5
        fin = np.float32([0.05, 0.06, 0.065]) * (1 - 0.3 * ray * np.clip(al * 4, 0, 1) + 0.15 * n2)[..., None]
        fin *= (1 - 0.4 * sm(al, 0.8, 1.0))[..., None]
        if k == 1:                                     # pectoral: pink base
            w = (1 - np.clip(al / 0.35, 0, 1)) * np.ones_like(ray)
            fin = np.float32([0.45, 0.22, 0.22]) * w[..., None] + fin * (1 - w)[..., None]
        if k == 2:                                     # pelvic / anal: whitish leading tip
            w = np.clip((0.16 - ac) / 0.10, 0, 1)[None, :] * np.clip((al - 0.3) / 0.4, 0, 1)
            fin = np.float32([0.60, 0.60, 0.58]) * w[..., None] + fin * (1 - w)[..., None]
        tex[r0:r1] = fin
    gy, gx = np.gradient(hgt)
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nmap = np.full((TEX_H, TEX_W, 4), 255, np.uint8)
    nmap[..., :3] = np.clip((nrm * 0.5 + 0.5) * 255, 0, 255)
    return tex, nmap, np.repeat((rough * 255).astype(np.uint8)[..., None], 3, -1)


def fish_texture():
    tex, nmap, rough = fish_albedo()
    return tp.data_texture(srgb8(tex), srgb=True), tp.data_texture(nmap, srgb=False), tp.data_texture(rough, srgb=False)


def fish_tint(rng):
    """(FISH_N, 1, 3) per-fish albedo multiplier: brightness plus a slight blue-vs-olive hue."""
    tint = rng.uniform(0.82, 1.06, (FISH_N, 1, 1)).astype(np.float32)
    hue = rng.uniform(0.0, 1.0, (FISH_N, 1, 1)).astype(np.float32)
    return tint * (1.0 - 0.15 * hue * np.float32([0.0, 0.3, 1.0]))


@wp.func
def wrap_pi(a: float) -> float:
    return a - 2.0 * wp.pi * wp.floor((a + wp.pi) / (2.0 * wp.pi))


@wp.kernel
def boids(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
          pos_o: wp.array(dtype=wp.vec3), vel_o: wp.array(dtype=wp.vec3),
          yaw: wp.array(dtype=float), pitch: wp.array(dtype=float), roll: wp.array(dtype=float),
          beat: wp.array(dtype=float), amp: wp.array(dtype=float), brake: wp.array(dtype=float),
          flen: wp.array(dtype=float), pref: wp.array(dtype=wp.vec3), leak_t: wp.array(dtype=float),
          uni: wp.array(dtype=wp.vec3), n: int, t: float, dt: float):
    i = wp.tid()
    p = pos[i]
    v = vel[i]
    sp = wp.max(wp.length(v), 1.0e-4)
    fwd = v / sp
    coh = wp.vec3(0.0, 0.0, 0.0)
    ali = wp.vec3(0.0, 0.0, 0.0)
    sep = wp.vec3(0.0, 0.0, 0.0)
    cnt = float(0.0)
    for j in range(n):
        if j != i:
            d = pos[j] - p
            r2 = wp.dot(d, d)
            if r2 < 4.0:
                coh += d
                ali += vel[j]
                cnt += 1.0
            if r2 < 0.42:
                sep -= d / wp.max(r2, 1.0e-3)
    if cnt > 0.0:
        coh = coh / cnt
        ali = ali / cnt - v
    acc = coh * 1.2 + ali * 1.8 + sep * 0.7
    rov = uni[0]
    tear = uni[1]
    e_r = uni[2]
    pen_r = uni[3][0]
    pen_d = uni[3][1]
    r = wp.sqrt(p[0] * p[0] + p[2] * p[2])
    rad = wp.vec3(p[0], 0.0, p[2]) / wp.max(r, 1.0e-3)
    tan = wp.vec3(-rad[2], 0.0, rad[0])
    leaking = leak_t[i] >= 0.0 and t > leak_t[i] + uni[3][2]
    near_tear = wp.length(p - tear) < 2.4
    goal = wp.vec3(0.0, 0.0, 0.0)
    if leaking:
        acc = acc * 0.25
        pj = wp.vec3(0.0, pref[i][2], 0.0)
        if r < pen_r:
            if wp.length(p - tear) > 1.6:
                goal = tear - e_r * 1.2 + pj
            else:
                goal = tear + e_r * 1.5 + pj
        else:
            goal = tear + e_r * 6.0 + tan * 3.0 * pref[i][2]
        g = goal - p
        want = wp.normalize(wp.vec3(g[0], 0.3 * g[1], g[2])) * 0.7    # a level run at the hole
        acc += (want - v) * 1.4
    else:
        want = tan * MILL_SPEED - rad * (r - pref[i][0]) * 0.45 + wp.vec3(0.0, (pref[i][1] - p[1]) * 0.45, 0.0)
        acc += (want - v) * 1.1
    dwall = pen_r - r
    if not (leaking and near_tear):
        if dwall > 0.0 and dwall < 2.2:
            acc -= rad * (2.2 - dwall) * 4.0
        if dwall <= 0.0 and dwall > -1.0:
            acc += rad * (1.0 + dwall) * 6.0
    if p[1] > -0.7:
        acc += wp.vec3(0.0, -6.0 * (p[1] + 0.7), 0.0)
    if p[1] < -pen_d + 1.0:
        acc += wp.vec3(0.0, 4.0 * (-pen_d + 1.0 - p[1]), 0.0)
    dr = p - rov
    lr = wp.length(dr)
    rr = 2.0
    if leaking:
        rr = 0.7
    if lr < rr:
        acc += dr / wp.max(lr, 1.0e-3) * (1.0 - lr / rr) * 5.0
    dc = p - uni[4]
    lc = wp.length(dc)
    if i > 0 and lc < 2.6:                            # keep the shot camera clear; fish 0 is the close-up hero
        acc += dc / wp.max(lc, 1.0e-3) * (1.0 - lc / 2.6) * 9.0
    acc += wp.vec3(0.0, -0.4 * v[1], 0.0)
    v2 = v + acc * dt
    s2 = wp.max(wp.length(v2), 1.0e-4)
    v2 = v2 * (wp.clamp(s2, FISH_SPEED_MIN, FISH_SPEED_MAX) / s2)
    vel_o[i] = v2
    pos_o[i] = p + v2 * dt
    s2 = wp.length(v2)
    f2 = v2 / s2
    dy = wrap_pi(wp.atan2(f2[0], f2[2]) - yaw[i])
    yaw[i] = yaw[i] + dy * 0.35
    pitch[i] = pitch[i] + (-wp.asin(wp.clamp(f2[1], -1.0, 1.0)) - pitch[i]) * 0.2
    roll[i] = roll[i] + (wp.clamp(-1.2 * dy / dt, -0.6, 0.6) - roll[i]) * (1.0 - wp.exp(-dt / 0.25))
    beat[i] = beat[i] + dt * 2.0 * wp.pi * (0.9 + 1.1 * s2 / wp.max(flen[i], 0.3))
    amp[i] = amp[i] + ((0.035 + 0.055 * wp.min(s2, 1.4) / 1.4) - amp[i]) * (1.0 - wp.exp(-dt / 0.4))
    b = wp.clamp(-wp.dot(acc, fwd) * 0.5, 0.0, 1.0) * 0.8 + 0.1
    brake[i] = brake[i] + (b - brake[i]) * (1.0 - wp.exp(-dt / 0.3))


@wp.kernel
def skin(c0: wp.array(dtype=wp.vec3), c1: wp.array(dtype=wp.vec3),
         n0: wp.array(dtype=wp.vec3), n1: wp.array(dtype=wp.vec3), uu: wp.array(dtype=float),
         pos: wp.array(dtype=wp.vec3), yaw: wp.array(dtype=float), pitch: wp.array(dtype=float),
         roll: wp.array(dtype=float), beat: wp.array(dtype=float), amp: wp.array(dtype=float),
         brake: wp.array(dtype=float), flen: wp.array(dtype=float),
         out_p: wp.array(dtype=wp.vec3), out_n: wp.array(dtype=wp.vec3), nv: int):
    tid = wp.tid()
    f = tid // nv
    v = tid - f * nv
    fl = brake[f]
    c = c0[v] * (1.0 - fl) + c1[v] * fl
    nn = wp.normalize(n0[v] * (1.0 - fl) + n1[v] * fl)
    u = uu[v]
    a = amp[f]
    # Wave envelope: grows from 30 % back, with a small counter-swing of the head.
    if u < 0.3:
        env = -0.12 * (0.3 - u) / 0.3
        denv = 0.12 / 0.3
    else:
        env = ((u - 0.3) / 0.7) * ((u - 0.3) / 0.7)
        denv = 2.0 * (u - 0.3) / 0.49
    ph = F_K * u - beat[f]
    s = a * env * wp.sin(ph)
    th = wp.atan(-a * (denv * wp.sin(ph) + env * F_K * wp.cos(ph)))
    ct = wp.cos(th)
    st = wp.sin(th)
    lp = wp.vec3(s + c[0] * ct, c[1], c[2] - c[0] * st) * flen[f]
    ln = wp.vec3(nn[0] * ct + nn[2] * st, nn[1], -nn[0] * st + nn[2] * ct)
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), yaw[f]) * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pitch[f]) \
        * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), roll[f])
    out_p[tid] = wp.quat_rotate(q, lp) + pos[f]
    out_n[tid] = wp.quat_rotate(q, ln)


class School:
    def __init__(self):
        c0, c1, n0, n1, uv, uu, kind, tri = salmon()
        self.nv = len(c0)
        rng = np.random.default_rng(20260903)
        self.col = np.ascontiguousarray(np.broadcast_to(fish_tint(rng), (FISH_N, self.nv, 3)).reshape(-1, 3))
        self.uv = np.ascontiguousarray(np.tile(uv, (FISH_N, 1)))
        self.index = np.ascontiguousarray((tri[None, :] + (np.arange(FISH_N, dtype=np.uint32) * self.nv)[:, None]).reshape(-1))
        L = rng.uniform(0.45, 0.75, FISH_N).astype(np.float32)
        pref = np.stack([rng.uniform(2.4, 4.6, FISH_N), rng.uniform(-4.2, -1.4, FISH_N), rng.uniform(-0.25, 0.25, FISH_N)], 1)
        leak = np.full(FISH_N, -1.0, np.float32)
        nl = max(int(LEAK_FRAC * FISH_N), 3)
        who = rng.choice(np.arange(1, FISH_N), nl, replace=False)      # fish 0 is the close-up hero
        leak[who] = 0.5 + 0.9 * np.arange(nl)
        th = rng.uniform(0.0, 2 * np.pi, FISH_N)
        rr = rng.uniform(2.2, 4.8, FISH_N)
        p0 = np.stack([rr * np.cos(th), pref[:, 1] + rng.normal(0, 0.5, FISH_N), rr * np.sin(th)], 1)
        p0[who] = (TEAR_C - e_r * rng.uniform(1.5, 3.5, nl)[:, None] + e_t * rng.uniform(-2.0, 2.0, nl)[:, None]
                   + np.float32([0, 1, 0]) * rng.uniform(-0.6, 0.6, nl)[:, None])
        v0 = np.stack([-np.sin(th), np.zeros(FISH_N), np.cos(th)], 1) * 0.7

        def A(a, dt=float):
            return wp.array(np.ascontiguousarray(a, np.float32), dtype=dt, device=device)

        self.pos, self.vel = A(p0, wp.vec3), A(v0, wp.vec3)
        self.pos2, self.vel2 = A(p0, wp.vec3), A(v0, wp.vec3)
        self.yaw = A(np.arctan2(v0[:, 0], v0[:, 2]))
        self.pitch, self.roll = A(np.zeros(FISH_N)), A(np.zeros(FISH_N))
        self.beat, self.amp = A(rng.uniform(0, 2 * np.pi, FISH_N)), A(np.full(FISH_N, 0.06))
        self.brake, self.len = A(np.full(FISH_N, 0.1)), A(L)
        self.pref, self.leak = A(pref, wp.vec3), A(leak)
        self.uni = wp.zeros(5, dtype=wp.vec3, device=device)
        self.c0, self.c1, self.n0, self.n1, self.uu = A(c0, wp.vec3), A(c1, wp.vec3), A(n0, wp.vec3), A(n1, wp.vec3), A(uu)
        self.out_p = wp.zeros(FISH_N * self.nv, dtype=wp.vec3, device=device)
        self.out_n = wp.zeros(FISH_N * self.nv, dtype=wp.vec3, device=device)

    def step(self, t, dt, rov_pos, cam_pos):
        self.uni.assign(np.asarray([rov_pos, tear_now(), e_r, [PEN_R, PEN_D, LEAK_T0[0]], cam_pos], np.float32))
        wp.launch(boids, dim=FISH_N, device=device,
                  inputs=[self.pos, self.vel, self.pos2, self.vel2, self.yaw, self.pitch, self.roll, self.beat,
                          self.amp, self.brake, self.len, self.pref, self.leak, self.uni, FISH_N, t, dt])
        self.pos, self.pos2 = self.pos2, self.pos
        self.vel, self.vel2 = self.vel2, self.vel

    def skin(self):
        wp.launch(skin, dim=FISH_N * self.nv, device=device,
                  inputs=[self.c0, self.c1, self.n0, self.n1, self.uu, self.pos, self.yaw, self.pitch, self.roll,
                          self.beat, self.amp, self.brake, self.len, self.out_p, self.out_n, self.nv])

    def hero(self):
        """(position, forward) of fish 0 on the host."""
        p, y = self.pos.numpy()[0], float(self.yaw.numpy()[0])
        return p, np.array([math.sin(y), 0.0, math.cos(y)])


school = School()
school.step(0.0, 1.0 / 60.0, np.zeros(3), np.zeros(3))
school.skin()
_fp0, _fn0 = school.out_p.numpy(), school.out_n.numpy()
fish_mat = tp.MeshPhysicalMaterial()
fish_mat.color = 0xffffff
fish_mat.roughness, fish_mat.metalness = 1.0, 0.15    # roughness lives in the map
fish_mat.specular_intensity = 0.2      # skin/water IOR contrast is small: F0 ~0.008, else the sky env chromes the back
fish_mat.iridescence, fish_mat.iridescence_ior, fish_mat.iridescence_thickness_nm = 0.5, 1.3, 350.0
fish_mat.side = tp.Side.Double
fish_mat.vertex_colors = True
fish_mat.map, fish_mat.normal_map, fish_mat.roughness_map = fish_texture()
fish_mat.normal_scale = tp.Vector2(0.7, 0.7)
fish_geo = tp.BufferGeometry()
fish_geo.set_attribute("position", _fp0)
fish_geo.set_attribute("normal", _fn0)
fish_geo.set_attribute("uv", school.uv)
fish_geo.set_attribute("color", school.col)
fish_geo.set_index(school.index)
fish = tp.Mesh(fish_geo, fish_mat)
fish.frustum_culled = False
fish.cast_shadow = False
fish.receive_shadow = True
scene.add(fish)


fish_vk = None                                        # (pos, nrm) VkInteropArray pair once armed


def fish_on_frame():
    """Inside render(): skin straight into the renderer's vertex buffers, synchronously."""
    t0 = time.perf_counter()
    school.skin()
    wp.synchronize_device(device)
    if prof:
        prof.add("fish skin (in render)", time.perf_counter() - t0)


def arm_fish_interop():
    """After the first render: CUDA->Vulkan zero copy for the school; host upload stays the fallback."""
    global fish_vk
    if VkInteropArray is None or not device.is_cuda or not INTEROP:
        return
    h = renderer.enable_vertex_interop(fish, fish_on_frame)
    if h is None:
        return
    try:
        fish_vk = tuple(VkInteropArray(hd, nb, wp.vec3, FISH_N * school.nv, device) for hd, nb in h)
    except Exception as e:
        print(f"fish interop import failed ({e}); host upload path")
        renderer.disable_vertex_interop(fish)
        return
    school.out_p, school.out_n = fish_vk[0].array, fish_vk[1].array
    atexit.register(release_fish_interop)


def release_fish_interop():
    global fish_vk
    if fish_vk is None:
        return
    pair, fish_vk = fish_vk, None
    school.out_p, school.out_n = _fish_p, _fish_n
    for a in pair:
        a.close()
    renderer.disable_vertex_interop(fish)


_fish_p, _fish_n = school.out_p, school.out_n


def fish_step(t, dt):
    cp = camera.position
    school.step(t, dt, rov_pos, np.array([cp.x, cp.y, cp.z]))
    mark("fish boids", gpu=True)
    if fish_vk is not None:
        return
    school.skin()
    p, n = school.out_p.numpy(), school.out_n.numpy()
    mark("fish skin", gpu=True)
    fish_geo.update_attribute("position", p)
    fish_geo.update_attribute("normal", n)
    mark("fish upload")


# ---- marine snow --------------------------------------------------------------
MOTE_CAP, MOTE_HALF, MOTE_TOP, MOTE_BOTTOM = 20_000, 9.0, WATER_Y - 0.55, -12.0   # the slab stays under the surface
_uc = tp.ParticleField.Config()
_uc.capacity = MOTE_CAP
_uc.ownership = tp.ParticleField.Ownership.Renderer
_uc.w_semantic = tp.ParticleField.WSemantic.Radius
_uc.uniform_radius = 0.007
motes = tp.ParticleField.create(_uc)
motes.frustum_culled = False
MOTE_BASE = 0.055
motes.set_billboard_repr(tp.Color(0.80, 0.90, 0.86), tp.Color(0.52, 0.72, 0.74), MOTE_BASE, 1.0)
_mb = motes.billboard_repr
_mb.lod_near = 0.0
_mb.lod_fade = 0.0
_mb.stretch_seconds = 0.0
_mb.softness = 0.85
_mb.fade_power = 0.0
_mb.size_taper = 0.0
_mb.near_fade = 0.35
_mb.bright_jitter = 0.75
_mb.glow = 0.0
_me = motes.emitter
_me.spawn_center = tp.Vector3(0.0, 0.5 * (MOTE_TOP + MOTE_BOTTOM), 0.0)
_me.spawn_half_extent = tp.Vector3(MOTE_HALF, 0.5 * (MOTE_TOP - MOTE_BOTTOM), MOTE_HALF)
_me.velocity = tp.Vector3(0.0, -0.022, 0.0)          # snow sinks, so the slab top is the ceiling
_me.speed_spread = 0.012
_me.wind = tp.Vector3(0.045, 0.0, 0.020)
_me.drift_amplitude = 0.10
_me.drift_frequency = 0.13
_me.drift_scale = 3.5
_me.lifetime = 40.0
_me.duty_cycle = 1.0
_me.size = 0.007
_me.size_jitter = 0.70
_me.seed = 20260903
_me.follow = True
_me.follow_snap = 0.0
motes.set_emitter(_me)
motes.set_emitter_time(0.0, 1.0 / 60.0)
motes.set_live_count(MOTE_CAP)
scene.add(motes)

# ---- sonar: Warp ray fan against a wp.Mesh of net + collar + fish ------------------------------
SON_BEAMS, SON_VS, SON_BINS, SON_RANGE, SON_FOV = 256, 8, 512, 20.0, 130.0
SON_W, SON_H, SON_VIEW = 512, 384, 4.0                  # 20 m of bins, 4 m on the display: the wall arc fills the fan


@wp.kernel
def sonar(mesh: wp.uint64, mat: wp.array(dtype=int), refl: wp.array(dtype=float), frame: wp.array(dtype=wp.vec3),
          out: wp.array2d(dtype=float), nb: int, nvs: int, nbins: int, rng: float):
    tid = wp.tid()
    i = tid // nvs
    j = tid - i * nvs
    az = (-0.5 * SON_FOV + SON_FOV * (float(i) + 0.5) / float(nb)) * wp.pi / 180.0
    el = (-6.0 + 12.0 * (float(j) + 0.5) / float(nvs)) * wp.pi / 180.0
    d = frame[1] * (wp.cos(el) * wp.cos(az)) + frame[2] * (wp.cos(el) * wp.sin(az)) + frame[3] * wp.sin(el)
    q = wp.mesh_query_ray(mesh, frame[0], d, rng)
    if q.result:
        b = wp.min(int(q.t / rng * float(nbins)), nbins - 1)
        wp.atomic_max(out, i, b, refl[mat[q.face]] * (0.35 + 0.65 * wp.abs(wp.dot(q.normal, d))) * wp.exp(-0.10 * q.t))


def net_solid_tris():
    """Periodic-grid triangles minus the invisible hole membrane, so the tear is a gap to the sonar."""
    iv, iu = np.meshgrid(np.arange(NVT - 1), np.arange(NU), indexing="ij")
    a, b = iv * NU + iu, iv * NU + (iu + 1) % NU
    tri = np.concatenate([np.stack([a, b, b + NU], -1), np.stack([a, b + NU, a + NU], -1)]).reshape(-1, 3)
    return tri[(net.cls[tri] != -2).all(1)]


def collar_tris(seg=96, sides=8):
    th, ph = np.meshgrid(2 * np.pi * np.arange(seg) / seg, 2 * np.pi * np.arange(sides) / sides, indexing="ij")
    r = PEN_R + 0.16 * np.cos(ph)
    pts = np.stack([r * np.cos(th), WATER_Y + 0.04 + 0.16 * np.sin(ph), r * np.sin(th)], -1).reshape(-1, 3)
    i, j = np.arange(seg)[:, None], np.arange(sides)[None, :]
    a, b, c, d = i * sides + j, i * sides + (j + 1) % sides, ((i + 1) % seg) * sides + (j + 1) % sides, ((i + 1) % seg) * sides + j
    return pts, np.stack([a, b, c, a, c, d], -1).reshape(-1, 3)


def fish_cage():
    """Sonar proxy per fish: every 4th body ring x every 4th side of the skinned mesh (42 verts, 72 tris)."""
    S1 = F_SIDES + 1
    rings, sides = np.append(np.arange(0, F_RINGS, 4), F_RINGS - 1), np.arange(0, F_SIDES, 4)
    ids = (rings[:, None] * S1 + sides[None, :]).reshape(-1)
    i, j = np.arange(len(rings) - 1)[:, None], np.arange(len(sides))[None, :]
    a, b = i * len(sides) + j, i * len(sides) + (j + 1) % len(sides)
    tri = np.stack([a, b, b + len(sides), a, b + len(sides), a + len(sides)], -1).reshape(-1, 3)
    ids = (ids[None, :] + (np.arange(FISH_N) * school.nv)[:, None]).reshape(-1)
    tri = (tri[None] + (np.arange(FISH_N) * len(rings) * len(sides))[:, None, None]).reshape(-1, 3)
    return wp.array(ids.astype(np.int32), dtype=int, device=device), tri


@wp.kernel
def gather(src: wp.array(dtype=wp.vec3), ids: wp.array(dtype=int), dst: wp.array(dtype=wp.vec3), off: int):
    i = wp.tid()
    dst[off + i] = src[ids[i]]


SON_EVERY = 3
_net_tri = net_solid_tris()
_col_pts, _col_tri = collar_tris()
_cage_ids, _cage_tri = fish_cage()
_n_net, _n_fish = net.n, len(_cage_ids)
_son_pts = wp.zeros(_n_net + _n_fish + len(_col_pts), dtype=wp.vec3, device=device)
wp.copy(_son_pts, wp.array(_col_pts.astype(np.float32), dtype=wp.vec3, device=device), _n_net + _n_fish, 0, len(_col_pts))
_son_idx = np.concatenate([_net_tri.reshape(-1), _cage_tri.reshape(-1) + _n_net, _col_tri.reshape(-1) + _n_net + _n_fish])
_son_idx_wp = wp.array(_son_idx.astype(np.int32), dtype=int, device=device)
son_mesh = None
son_mat = wp.array(np.concatenate([np.zeros(len(_net_tri)), np.full(len(_cage_tri), 2), np.ones(len(_col_tri))]).astype(np.int32),
                   dtype=int, device=device)
son_refl = wp.array(np.float32([1.0, 1.0, 0.35]), dtype=float, device=device)
son_frame = wp.zeros(4, dtype=wp.vec3, device=device)
son_out = wp.zeros((SON_BEAMS, SON_BINS), dtype=float, device=device)
_yy, _xx = np.mgrid[0:SON_H, 0:SON_W]
_dx, _dy = _xx - SON_W / 2, (SON_H - 12) - _yy
_r_m = np.hypot(_dx, _dy) / (SON_H - 24) * SON_VIEW
_bear = np.degrees(np.arctan2(_dx, np.maximum(_dy, 1e-6)))
SON_VALID = (_r_m < SON_VIEW) & (np.abs(_bear) < 0.5 * SON_FOV) & (_dy > 0)
SON_BEAM = np.clip(((_bear + 0.5 * SON_FOV) / SON_FOV * SON_BEAMS).astype(int), 0, SON_BEAMS - 1)
SON_BIN = np.clip((_r_m / SON_RANGE * SON_BINS).astype(int), 0, SON_BINS - 1)
SON_RINGS = SON_VALID & (np.abs(_r_m - np.round(_r_m)) < 0.0125) & (_r_m > 0.5)
SON_TICK = SON_VALID & (np.abs(_dx) < 0.8) & (_r_m > 0.5)
SON_EDGE = ((np.abs(np.abs(_bear) - 0.5 * SON_FOV) < 0.35) & (_r_m < SON_VIEW) & (_dy > 0)) | ((np.abs(_r_m - SON_VIEW) < 0.03) & (np.abs(_bear) < 0.5 * SON_FOV))
_k = np.linspace(0, 1, 256)[:, None]
_c0, _c1, _c2 = np.float32([0.02, 0.01, 0.0]), np.float32([0.62, 0.26, 0.05]), np.float32([1.0, 0.86, 0.55])
SON_LUT = srgb8(np.where(_k < 0.5, _c0 + (_c1 - _c0) * (_k / 0.5), _c1 + (_c2 - _c1) * ((_k - 0.5) / 0.5)))
_rb = (np.arange(SON_BINS) + 0.5) / SON_BINS * SON_RANGE
SON_NEAR = (0.35 * np.exp(-((_rb - 0.3) / 0.08) ** 2)).astype(np.float32)[None, :]
son_img = np.zeros((SON_H, SON_W, 4), np.uint8)
son_hist = np.zeros((3, SON_BEAMS, SON_BINS), np.float32)
SON_FRAME = (_yy < 2) | (_yy >= SON_H - 2) | (_xx < 2) | (_xx >= SON_W - 2)
# Flat index maps on the bottom-up image (what update_data takes), so the fan draw is gathers, not mask indexing.
SON_PIX = np.flatnonzero(SON_VALID[::-1])
SON_SRC = (SON_BEAM * SON_BINS + SON_BIN)[::-1].reshape(-1)[SON_PIX]
SON_LUT4 = np.concatenate([SON_LUT, (191 + (np.arange(256) >> 2)).astype(np.uint8)[:, None]], 1)
SON_OVER = np.zeros((SON_H, SON_W, 4), np.uint8)
SON_OVER[SON_RINGS | SON_EDGE] = (70, 42, 14, 255)
SON_OVER[SON_TICK] = np.maximum(SON_OVER[SON_TICK], np.uint8([120, 80, 30, 255]))
SON_OVER = np.ascontiguousarray(SON_OVER[::-1])
SON_FRAME_I = np.flatnonzero(SON_FRAME[::-1])
SON_TILT = -15.0                                        # mount pitch, deg: the 12 deg band sits on the hole, below the top flap


def sonar_step(frame_i):
    if frame_i % SON_EVERY or not HUD_LIVE[0]:
        return
    global son_mesh
    wp.copy(_son_pts, net.pos, 0, 0, _n_net)
    wp.launch(gather, dim=_n_fish, device=device, inputs=[school.out_p, _cage_ids, _son_pts, _n_net])
    son_mesh = wp.Mesh(points=_son_pts, indices=_son_idx_wp, bvh_constructor="lbvh")
    mark("sonar bvh build", gpu=True)
    o = rov_pos + rov_R @ [0.20, 0.09, 0.0]
    R = rov_R @ rot_z(math.radians(SON_TILT))
    son_frame.assign(np.asarray([o, R @ [1, 0, 0], R @ [0, 0, 1], R @ [0, 1, 0]], np.float32))
    son_out.zero_()
    wp.launch(sonar, dim=SON_BEAMS * SON_VS, device=device,
              inputs=[son_mesh.id, son_mat, son_refl, son_frame, son_out, SON_BEAMS, SON_VS, SON_BINS, SON_RANGE])
    son_hist[(frame_i // SON_EVERY) % 3] = son_out.numpy()
    mark("sonar kernel", gpu=True)
    a = son_hist.max(0)                                                   # 3-frame persistence
    a = np.maximum(a, np.maximum(np.roll(a, 1, 1), np.roll(a, -1, 1)))   # +-1 bin: the wall reads as a solid arc
    a = a * np.random.default_rng(1000 + frame_i).uniform(0.7, 1.3, (SON_BEAMS, SON_BINS)).astype(np.float32) + SON_NEAR
    v = (255 * (1 - np.exp(-4.0 * a))).astype(np.uint8)
    flat = son_img.reshape(-1, 4)
    flat[:] = (6, 6, 6, 191)
    flat[SON_PIX] = SON_LUT4[v.reshape(-1)[SON_SRC]]
    np.maximum(son_img, SON_OVER, out=son_img)
    flat[SON_FRAME_I] = (208, 138, 42, 255)
    son_tex.update_data(son_img)
    mark("sonar draw")


# ---- insets: ROV camera view + sonar as screen-space sprites ---------------------------------
CAM_W, CAM_H, HUD_M = 480, 270, 16
rov_cam = tp.PerspectiveCamera(80.0, CAM_W / CAM_H, 0.05, 200.0)
FONT = tp.FontLoader().default_font()


def hud_panel(w, h, ax, label, channels=3):
    tex = tp.data_texture(np.zeros((h, w, channels), np.uint8), srgb=True)
    tex.generate_mipmaps = False
    tex.wrap_s = tex.wrap_t = tp.TextureWrapping.ClampToEdge
    mat = tp.SpriteMaterial()
    mat.map = tex
    sp = tp.Sprite(mat)
    sp.screen_space = True
    sp.screen_anchor.set(ax, 0.0)
    sp.center.set(ax, 0.0)
    sp.position.set(HUD_M if ax == 0 else -HUD_M, HUD_M, 0.0)
    sp.scale.set(w, h, 1.0)
    scene.add(sp)
    lab = tp.TextSprite(FONT, world_scale=18.0)
    lab.set_text(label)
    lab.set_color(0xe6f0ee)
    lab.screen_space = True
    lab.screen_anchor.set(ax, 0.0)
    lab.center.set(ax, 0.0)
    lab.position.set(HUD_M if ax == 0 else -HUD_M, HUD_M + h + 6, 0.0)
    scene.add(lab)
    return tex, (sp, lab)


cam_tex, _hud_a = hud_panel(CAM_W, CAM_H, 0.0, "ROV CAM")
son_tex, _hud_b = hud_panel(SON_W, SON_H, 1.0, "SONAR 130 deg", 4)
HUD_ON = not SHOT or SHOT.startswith("p3")
HUD_LIVE = [HUD_ON]                                  # the film parks the insets per cut
ROV_VIEW, PIP = 0, False                             # PIP is resolved at renderer setup; hud_park reads it before that


def hud_park(live):
    HUD_LIVE[0] = live
    for _o in _hud_b + (_hud_a[1:] if PIP else _hud_a):      # off-screen, never visible=False
        _o.position.x = (HUD_M if _o in _hud_a else -HUD_M) if live else -9000.0


if not HUD_ON:
    hud_park(False)


def rov_cam_place():
    p = rov_pos + rov_R @ [0.16, 0.035, 0.0]
    rov_cam.position.set(*p)
    rov_cam.look_at(*(p + rov_R @ [1.0, 0.0, 0.0]))


def hud_update():
    if PIP:                                           # composited on the device; the rect follows the window height
        renderer.set_view_display_rect(ROV_VIEW, HUD_M if HUD_LIVE[0] else -9000, renderer.size()[1] - HUD_M - CAM_H, CAM_W, CAM_H)
    elif ROV_VIEW and HUD_LIVE[0] and frame_i % 4 == 0:   # fallback: a device-idle readback, 15 Hz
        rgb = renderer.read_view_rgb_pixels(ROV_VIEW)
        if rgb.size:
            cam_tex.update_data(np.ascontiguousarray(rgb[::-1]))



# ---- the filming drone (ported from warp_sailboat.py) ------------------------
#  A camera vehicle, not a quadrotor sim: the attitude is kinematic -- it leans the way
#  the acceleration says it must be leaning for the path it is on, which is all the eye
#  reads at filming distance. Names are _dr_-prefixed: `props` is already the ROV's.
DRONE_SPAN = 0.45
DRONE_ARM = DRONE_SPAN * 0.5 * 0.7071
DRONE_G = 9.81

drone = tp.Group()
drone.rotation.order = tp.RotationOrder.YXZ
drone.visible = False
scene.add(drone)

_dr_body = standard_material(0x2c333d, roughness=0.42, metalness=0.25)
_dr_trim = standard_material(0x515a66, roughness=0.55)
# Bare blades, no blur disc: at this span and filming distance a disc reads as a grey
# plate bolted to the arm, while the blades alias into a filmed prop's counter-rotation.
_dr_prop = standard_material(0x14171c, roughness=0.60, side=tp.Side.Double)
_dr_lens = standard_material(0x07090c, roughness=0.05, metalness=0.2)

_dh = tp.Mesh(tp.BoxGeometry(0.115, 0.052, 0.170), _dr_body)
_dh.cast_shadow = True
drone.add(_dh)
_dc = tp.Mesh(tp.SphereGeometry(0.055, 14, 10), _dr_trim)
_dc.scale.set(1.0, 0.52, 1.30)
_dc.position.set(0.0, 0.026, 0.005)
drone.add(_dc)
for _s in (1.0, -1.0):
    _bm = tp.Mesh(tp.BoxGeometry(DRONE_SPAN, 0.015, 0.021), _dr_trim)
    _bm.rotation.y = _s * math.radians(45.0)
    _bm.cast_shadow = True
    drone.add(_bm)

_dr_props = []
for _sx, _sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
    _px, _pz = _sx * DRONE_ARM, _sz * DRONE_ARM
    _pod = tp.Mesh(tp.CylinderGeometry(0.017, 0.020, 0.032, 10, 1), _dr_body)
    _pod.position.set(_px, 0.013, _pz)
    drone.add(_pod)
    _hub = tp.Group()
    _hub.position.set(_px, 0.035, _pz)
    drone.add(_hub)
    _bl = tp.Mesh(tp.BoxGeometry(0.126, 0.0024, 0.016), _dr_prop)
    _bl.rotation.z = math.radians(9.0)      # a little pitch, so it reads as a blade
    _hub.add(_bl)
    _dr_props.append((_hub, 1.0 if _sx * _sz > 0 else -1.0))

_dr_leds = []
for _col, _x, _z in ((0xff1408, -1.02, 1.02), (0x18ff3c, 1.02, 1.02)):
    _m = standard_material(0x0a0a0a, roughness=1.0, emissive=_col, emissive_intensity=6.0)
    _sp = tp.Mesh(tp.SphereGeometry(0.012, 10, 8), _m)
    _sp.position.set(_x * DRONE_ARM, 0.004, _z * DRONE_ARM)
    drone.add(_sp)
    _dr_leds.append(_m)
_dr_strobe = standard_material(0x0a0a0a, roughness=1.0, emissive=0xffffff, emissive_intensity=0.0)
_st = tp.Mesh(tp.SphereGeometry(0.013, 10, 8), _dr_strobe)
_st.position.set(0.0, -0.030, -0.030)
drone.add(_st)

_dr_gimbal = tp.Group()
_dr_gimbal.position.set(0.0, -0.038, 0.082)
drone.add(_dr_gimbal)
_gb = tp.Mesh(tp.SphereGeometry(0.027, 14, 10), _dr_body)
_gb.scale.set(1.0, 1.0, 0.85)
_dr_gimbal.add(_gb)
_gl = tp.Mesh(tp.CylinderGeometry(0.014, 0.017, 0.018, 12, 1), _dr_lens)
_gl.rotate_x(math.pi / 2)
_gl.position.set(0.0, 0.0, 0.022)
_dr_gimbal.add(_gl)

drone_state = {
    "pos": np.array([0.0, 19.0, 0.0]), "vel": np.zeros(3), "acc": np.zeros(3),
    "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "spin": 0.0, "spool": 0.0,
    "throttle": 0.5, "have": False,
}
DRONE_LIVE = [True]            # terrain_scan replays cam_approach 600+ times: freeze the machine


def _dr_step01(a, b, x):
    x = min(max((x - a) / (b - a), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _dr_wrap(a, b, k):
    return a + ((b - a + math.pi) % (2.0 * math.pi) - math.pi) * k


def drone_set_pose(pos, look_at, dt=1.0 / 60.0):
    """Put the drone at `pos` looking at `look_at`. THE call a shot makes.
    Attitude is derived, never authored: smoothed velocity and acceleration give the lean a
    machine on that path must have. Yaw follows the velocity heading once it is actually
    moving and the look bearing when it is not, so a hover does not spin on noise."""
    d = drone_state
    pp = np.asarray(pos, np.float64).copy()
    tgt = np.asarray(look_at, np.float64).copy()
    dt = max(float(dt), 1e-4)
    if not d["have"]:
        d["have"] = True
        d["vel"][:] = 0.0
        d["acc"][:] = 0.0
        d["yaw"] = math.atan2((tgt - pp)[0], (tgt - pp)[2])
    else:
        v = (pp - d["pos"]) / dt
        vprev = d["vel"].copy()
        d["vel"] += (v - d["vel"]) * (1.0 - math.exp(-dt / 0.12))
        d["acc"] += ((d["vel"] - vprev) / dt - d["acc"]) * (1.0 - math.exp(-dt / 0.22))
    d["pos"] = pp
    sp = float(np.linalg.norm(d["vel"][[0, 2]]))
    to = tgt - pp
    yaw_t = _dr_wrap(math.atan2(to[0], to[2]),
                     math.atan2(d["vel"][0], d["vel"][2]) if sp > 1e-3 else math.atan2(to[0], to[2]),
                     _dr_step01(0.6, 2.5, sp))
    d["yaw"] = _dr_wrap(d["yaw"], yaw_t, 1.0 - math.exp(-dt / 0.15))
    fwd = np.array([math.sin(d["yaw"]), 0.0, math.cos(d["yaw"])])
    lat = np.array([math.cos(d["yaw"]), 0.0, -math.sin(d["yaw"])])
    lim = math.radians(25.0)
    pt = max(-lim, min(lim, math.atan2(float(np.dot(d["acc"], fwd)), DRONE_G)))
    rl = max(-lim, min(lim, -math.atan2(float(np.dot(d["acc"], lat)), DRONE_G)))
    k = 1.0 - math.exp(-dt / 0.10)
    d["pitch"] += (pt - d["pitch"]) * k
    d["roll"] += (rl - d["roll"]) * k
    d["throttle"] = float(np.clip(0.5 + d["acc"][1] / 9.0, 0.0, 1.0))
    drone.position.set(float(pp[0]), float(pp[1]), float(pp[2]))
    drone.rotation.set(d["pitch"], d["yaw"], d["roll"])
    dh = math.hypot(float(to[0]), float(to[2]))              # the gimbal holds the horizon
    _dr_gimbal.rotation.set(math.atan2(-float(to[1]), max(dh, 1e-3)) - d["pitch"], 0.0, -d["roll"])


def drone_tick(dt):
    """Props and the strobe. Runs whenever the machine is in the world."""
    d = drone_state
    d["spool"] += (1.0 - d["spool"]) * (1.0 - math.exp(-dt / 0.55))
    d["spin"] += (100.0 + 55.0 * d["throttle"]) * d["spool"] * dt
    for hub, sgn in _dr_props:
        hub.rotation.y = sgn * d["spin"]
    blink = 340.0 if (world_t % 1.05) < 0.065 else 0.0
    if blink != _dr_strobe.emissive_intensity:
        _dr_strobe.emissive_intensity = blink
        _dr_strobe.needs_update()


# ---- cameras -----------------------------------------------------------------
camera = tp.PerspectiveCamera(60.0, W / H, 0.05, 6000.0)   # the far shore is 2 km out and the
# ocean spans 5 km; 600 clipped both. Reverse-Z on a D32_SFLOAT buffer puts the precision at the
# near plane, so the far plane is nearly free -- 0.05 is what actually sets resolution here.
gulls.set_observer(camera)
murk_plane_update()
sun_h3 = np.array([SUN_H[0], 0.0, SUN_H[1]])
SHOTS = {
    "p1_net_wide": (1.0 * e_r + [0, -2.6, 0], TEAR_C + [0, 0.6, 0], 62.0),
    "p1_collar_below": (-1.5 * sun_h3 + [0, -5.0, 0], 0.85 * PEN_R * sun_h3 + [0, 0.3, 0], 78.0),
    "p1_rov_hero": None,
    "p1_tear": (TEAR_C - 2.0 * e_r + [0, 0.15, 0], TEAR_C, 55.0),
    "p2_school": (-1.2 * sun_h3 + [0, -6.2, 0], 4.5 * sun_h3 + [0, -0.6, 0], 72.0),
    "p2_fish_close": None,
    "p2_leak": (TEAR_C + 2.6 * e_r - 1.4 * e_t + [0, 0.35, 0], TEAR_C + 0.6 * e_r, 52.0),
    "p3_hud": None,
    "p3_sonar_tear": None,
    "barge": (BARGE_POS + 13.0 * _b_bow - 9.5 * _b_port + [0, 10.0, 0], BARGE_POS + [0, 2.4, 0], 46.0),
    "barge_stern": (BARGE_POS - 21.0 * _b_bow - 4.5 * _b_port + [0, 4.4, 0], BARGE_POS - 8.8 * _b_bow + [0, 1.9, 0], 46.0),
}


def place(cam, key):
    if key == "p2_fish_close":
        hp, hf = school.hero()
        side = np.cross([0.0, 1.0, 0.0], hf)
        side *= 1.0 if np.dot(side, sun_h3) > 0 else -1.0
        cam.fov = 45.0
        cam.update_projection_matrix()
        cam.position.set(*(hp + 0.78 * side + 0.32 * hf + [0, 0.22, 0]))
        cam.look_at(*(hp + 0.05 * hf))
        return
    if SHOTS[key] is None:
        fwd = rov_R @ [1.0, 0.0, 0.0]
        inb = -np.array([rov_pos[0], 0.0, rov_pos[2]]) / np.hypot(rov_pos[0], rov_pos[2])
        if key == "p1_rov_hero":
            p, t, fov = rov_pos - 0.85 * e_r + 1.15 * fwd + [0, 0.30, 0], rov_pos + [0, 0.02, 0], 50.0
        elif key == "p3_hud":
            p, t, fov = rov_pos - 1.9 * fwd + 1.1 * inb + [0, 0.6, 0], rov_pos + 1.2 * fwd + [0, -0.2, 0], 62.0
        else:                                    # p3_sonar_tear: beside and above the ROV, looking past it at the tear
            p, t, fov = rov_pos - 1.5 * fwd + 1.2 * e_t + [0, 0.6, 0], TEAR_C, 50.0
    else:
        p, t, fov = SHOTS[key]
    cam.fov = fov
    cam.update_projection_matrix()
    cam.position.set(*net_clear(np.asarray(p, np.float64)))
    cam.look_at(*t)


# ---- step --------------------------------------------------------------------
class Prof:
    """--profile: per-stage wall time; gpu=True syncs Warp first so kernel time lands on its stage."""
    def __init__(self):
        self.acc, self.t0, self.on = {}, time.perf_counter(), True

    def add(self, name, s):
        if self.on:
            self.acc[name] = self.acc.get(name, 0.0) + s

    def mark(self, name, gpu=False):
        if not self.on:
            return
        if gpu:
            wp.synchronize_device(device)
        t = time.perf_counter()
        self.add(name, t - self.t0)
        self.t0 = t

    def report(self, frames, engine):
        inner = sum(v for k, v in self.acc.items() if k.endswith("(in render)"))   # ran inside render(): split out
        tot = sum(self.acc.values()) / frames
        print(f"--- {frames} frames, {1000 * tot:.1f} ms/frame = {1 / tot:.1f} fps (with per-stage GPU syncs)")
        for k, v in self.acc.items():
            print(f"{k:24s}{1000 * (v - inner if k == 'render' else v) / frames:8.2f} ms")
        for k, v in sorted(engine.items()):
            print(f"  engine {k:22s}{v / frames:8.2f} ms")


prof = Prof() if PROFILE else None
mark = prof.mark if PROFILE else (lambda name, gpu=False: None)
world_t = 0.0
sim_t = 0.0                                           # render clock: ocean, particle fields, clouds, shader time follow it, not the wall
frame_i = 0


def step(dt=1.0 / 60.0):
    global world_t, sim_t, frame_i
    world_t += dt
    sim_t += dt
    renderer.sim_time = sim_t                         # before every render; warm-up frames advance it too (never rewound)
    frame_i += 1
    mark("other")
    rov_pose(world_t + PATROL_T0, dt)
    pins = tether_pins(dt)
    rope.pins.assign(np.asarray(pins, np.float32))
    rope.pay_out(float(np.linalg.norm(pins[1] - pins[0])))
    sp = 0.15 * (1.0 + 0.35 * math.sin(0.21 * world_t))
    ang = 0.35 + 0.25 * math.sin(0.09 * world_t)
    net_step(np.array([sp * math.cos(ang), 0.0, sp * math.sin(ang)], np.float32), world_t)
    mark("net+tether solve", gpu=True)
    net_upload()
    mark("net upload")
    tether_upload()
    mark("tether tube")
    fish_step(world_t, dt)
    tf, tl, tv = rov_thrust
    for k, p in enumerate(props):
        cmd = (abs(tf) + 0.5 * abs(tl) + (0.4 if k >= 2 else 0.0) * max(tf, 0.0)) if k < 4 else abs(tv) + 0.15
        p.rotation.y += dt * 45.0 * min(cmd, 1.5) * (1 if k % 2 else -1)
    bubbles_step(dt)
    barge_bob()
    if GEO is not None:
        # Tile LOD + shell/forest LOD follow the ACTIVE camera. step() runs
        # before the cut's film_cam() places it, so the tiles see last frame's
        # position -- at 60 fps and these camera speeds that is centimetres,
        # far inside the split/merge dead band.
        GEO.update(camera.position)
    if gulls.position.y > -1.0:
        gulls.update(dt)
    mark("props+bubbles")
    rov_cam_place()
    hud_update()
    mark("rov cam inset")
    sonar_step(frame_i)
    cp = camera.position
    motes.set_follow_center(tp.Vector3(cp.x, cp.y, cp.z))
    motes.set_emitter_time(world_t, dt)
    motes.billboard_repr.intensity = MOTE_BASE * (1.0 + 0.9 * min(max(-cp.y, 0.0), 8.0) / 8.0)
    murk_plane_update()
    mark("motes")


# ---- film ----------------------------------------------------------------------
FPS, WARMUP = 60, 30
P0 = PATROL[0, :3]
R0 = np.array([P0[0], 0.0, P0[2]]) / np.hypot(P0[0], P0[2])
T0 = np.array([-R0[2], 0.0, R0[0]])
_anchor = {}
CAM_STATS = [float("inf"), float("inf"), 0.0]         # min camera-to-net particle distance: film, current cut, its time


def smooth(u, hold=0.0):
    """Ease in/out on [0, 1 - hold]; flat for the last `hold` of the cut."""
    u = min(max(u / max(1.0 - hold, 1e-6), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def pchip(xs, ys, x):
    """Monotone cubic through (xs, ys): eased, no overshoot, flat where ys repeats; zero slope at the ends."""
    h, d = np.diff(xs), np.diff(ys) / np.diff(xs)
    m = np.zeros(len(xs))
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] > 0:
            m[i] = 2.0 / (1.0 / d[i - 1] + 1.0 / d[i])
    i = int(np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2))
    u = min(max((x - xs[i]) / h[i], 0.0), 1.0)
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * ys[i] + (u ** 3 - 2 * u ** 2 + u) * h[i] * m[i]
            + (-2 * u ** 3 + 3 * u ** 2) * ys[i + 1] + (u ** 3 - u ** 2) * h[i] * m[i + 1])


class Path:
    """Centripetal Catmull-Rom through timed points, paced by a pchip schedule (a repeated point = a hold)."""
    def __init__(self, keys):
        self.t = np.array([k[0] for k in keys], np.float64)
        p = np.array([np.asarray(k[1], np.float64) for k in keys])
        d = np.sqrt(np.linalg.norm(np.diff(p, axis=0), axis=1))
        k = np.concatenate([[0.0], np.cumsum(np.maximum(d, 1e-4))])
        self.p = np.concatenate([[2 * p[0] - p[1]], p, [2 * p[-1] - p[-2]]])       # phantom ends
        self.k = np.concatenate([[2 * k[0] - k[1]], k, [2 * k[-1] - k[-2]]])
        self.knot = k

    def __call__(self, t):
        s = float(pchip(self.t, self.knot, t))
        i = int(np.clip(np.searchsorted(self.knot, s, side="right"), 1, len(self.knot) - 1))
        t0, t1, t2, t3 = self.k[i - 1:i + 3]
        p0, p1, p2, p3 = self.p[i - 1:i + 3]
        a1 = ((t1 - s) * p0 + (s - t0) * p1) / (t1 - t0)
        a2 = ((t2 - s) * p1 + (s - t1) * p2) / (t2 - t1)
        a3 = ((t3 - s) * p2 + (s - t2) * p3) / (t3 - t2)
        b1 = ((t2 - s) * a1 + (s - t0) * a2) / (t2 - t0)
        b2 = ((t3 - s) * a2 + (s - t1) * a3) / (t3 - t1)
        return ((t2 - s) * b1 + (s - t1) * b2) / (t2 - t1)


T_AIR = 8.0                                                      # the drone leg, prepended to cut 0
T_HAND = 5.0 + T_AIR                                             # the eye lets go of the drone
T_LIFT, T_WL, FILM_T_OFF, T_DIVE = 9.0 + T_AIR, 17.0 + T_AIR, 22.0 + T_AIR, 24.0 + T_AIR
T_SWEEP, T_HOLD0, T_HOLD1, T_FOLLOW, T_END = 31.0 + T_AIR, 36.0 + T_AIR, 44.0 + T_AIR, 49.0 + T_AIR, 54.0 + T_AIR
SONAR_S = 6.0
APPROACH = {}


def rov_at(tf):
    """Baked ROV at film time tf (no live clearance): pos, fwd, inboard."""
    x, y, z, yaw = patrol_row(tf - FILM_T_OFF)[:4]
    p = np.array([x, y, z]) + lift_at(tf - FILM_T_OFF)
    return p, np.array([math.cos(yaw), 0.0, -math.sin(yaw)]), -np.array([p[0], 0.0, p[2]]) / np.hypot(p[0], p[2])


DRONE_PATH = None


def build_drone_path():
    """Inbound on bearing 243 deg: the sun (38.6 deg up, bearing 26 deg) then sits off the
    machine's right shoulder, so it flies with a lit face and a rim rather than as a black
    blob over the glitter path. Ends on a standoff hover 3.3 m outside the collar, then
    climbs away to overwatch so it is still there when the camera leaves it."""
    return Path([
        (0.0, [-19.07, 19.00, -37.42]), (2.0, [-18.02, 18.20, -34.60]),
        (7.0, [-12.55, 15.30, -21.35]), (8.3, [-11.20, 14.60, -18.10]),
        (11.0, [-8.05, 12.40, -11.60]), (T_HAND, [-6.20, 11.50, -8.23]),
        (T_HAND + 5.0, [-8.21, 13.21, -7.65]), (T_HAND + 11.0, [-11.53, 15.20, -7.21]),
        (T_END, [-12.40, 16.00, -7.05])])


def build_approach():
    """Air -> handrail -> waterline -> under with the ROV -> across the pen -> hold -> back to the wall for the patrol."""
    W = R0 * (PEN_R + 0.43) + [0, WATER_Y + 0.36, 0]                 # the ROV on the walkway
    LIFT[0] = Path([(0.0, W), (T_LIFT, W), (T_LIFT + 2.5, W + [0, 1.4, 0]), (T_LIFT + 5.5, [P0[0], WATER_Y + 1.4, P0[2]]),
                    (FILM_T_OFF - 3.0, P0 + [0, 0.9, 0]), (FILM_T_OFF, P0)])
    R = lambda t, dy=0.1: rov_at(t)[0] + [0, dy, 0]
    WL = P0 - 2.5 * R0 + 0.6 * T0
    wl = lambda y: np.array([WL[0], WATER_Y + y, WL[2]])

    def desc(t, dy):
        p, _, inb = rov_at(t)
        return p + 2.3 * inb + [-inb[2], 0.0, inb[0]] + [0, dy, 0]

    def follow(t, back):
        p, fwd, inb = rov_at(t)
        return p - back * fwd + 1.1 * inb + [0, 0.6, 0]

    def ahead(t):
        p, fwd, _ = rov_at(t)
        return p + 1.2 * fwd + [0, -0.2, 0]
    ph, pf, _ = rov_at(T_HOLD0 + 4.0)                                # the ROV holding station at the wall
    H = -5.0 * np.array([ph[0], 0.0, ph[2]]) / np.hypot(ph[0], ph[2]) + [0, -4.7, 0]   # 2 m inside the opposite wall, under the band
    look = ph + [0, 0.9, 0] + 2.2 * pf                               # the ROV small, off centre; the window above
    mid = 0.5 * (H + follow(T_FOLLOW, 3.2)) * [0.75, 1.0, 0.75]
    # 0 -> T_HAND the eye chases the drone; T_HAND -> 18 it lets go and its own momentum
    # carries it round the enclosure onto the ROV. The 15.7 key is load-bearing: with only
    # the two ends the centripetal Catmull-Rom chords straight across the pen.
    APPROACH["eye"] = Path([
        (0.0, [-20.66, 20.50, -41.09]), (2.0, [-18.69, 19.75, -36.42]), (7.0, [-13.03, 16.94, -23.06]),
        (8.3, [-11.97, 16.23, -20.62]), (11.0, [-8.90, 14.00, -14.78]), (T_HAND, [-6.65, 13.10, -12.87]),
        (15.7, [-12.67, 13.05, -2.92]),
        (10.0 + T_AIR, R0 * (PEN_R + 5.0) + 2.0 * T0 + [0, 13.0, 0]),
        (13.5 + T_AIR, R0 * (PEN_R + 1.8) + 2.2 * T0 + [0, 3.0, 0]),
        (15.5 + T_AIR, R0 * (PEN_R - 2.0) + 1.6 * T0 + [0, 0.9, 0]),
        (T_WL, wl(0.10)), (T_WL + 4.0, wl(0.07)), (T_DIVE, wl(-0.12)),
        (26.0 + T_AIR, desc(26.0 + T_AIR, -1.0)), (28.5 + T_AIR, desc(28.5 + T_AIR, -1.2)),
        (T_SWEEP, desc(T_SWEEP, -1.4)),
        (T_HOLD0, H), (T_HOLD1, H), (T_HOLD1 + 2.5, mid),
        (T_FOLLOW, follow(T_FOLLOW, 3.2)), (T_FOLLOW + 2.5, follow(T_FOLLOW + 2.5, 2.55)), (T_END, follow(T_END, 1.9))])
    C = np.array([0.0, WATER_Y - 0.5, 0.0])
    # the chase target is the air over the pen, lifted so the drone flies across the
    # enclosure rather than across empty water; it settles as the eye lets go
    APPROACH["tgt"] = Path([
        (0.0, C + [0, 2.2, 0]), (7.0, C + [0, 2.2, 0]), (T_HAND, C + [0, 1.2, 0]),
        (10.0 + T_AIR, R(10.0 + T_AIR)), (13.5 + T_AIR, R(13.5 + T_AIR)), (15.5 + T_AIR, R(15.5 + T_AIR)),
        (T_WL, R(T_WL)), (T_WL + 4.0, R(T_WL + 4.0)), (T_DIVE, R(T_DIVE)),
        (26.0 + T_AIR, R(26.0 + T_AIR, 1.0)), (28.5 + T_AIR, R(28.5 + T_AIR, 1.0)), (T_SWEEP, R(T_SWEEP, 1.0)),
        (T_HOLD0, look), (T_HOLD1, look), (T_HOLD1 + 2.5, ahead(T_HOLD1 + 2.5)),
        (T_FOLLOW, ahead(T_FOLLOW)), (T_FOLLOW + 2.5, ahead(T_FOLLOW + 2.5)), (T_END, ahead(T_END))])
    # the crossfade the film already had: it just never had anything to hand over FROM
    APPROACH["w"] = (np.array([0.0, T_HAND, 10.0 + T_AIR, T_END]), np.array([0.0, 0.0, 1.0, 1.0]))
    APPROACH["fov"] = (np.array([0.0, T_DIVE, 26.0 + T_AIR, T_SWEEP, T_HOLD0, T_HOLD1, T_FOLLOW, T_END]),
                       np.array([55.0, 55.0, 66.0, 66.0, 64.0, 64.0, 62.0, 62.0]))
    global DRONE_PATH
    DRONE_PATH = build_drone_path()


def cam_approach(u, t):
    hud_park(t >= T_FOLLOW - 0.5)
    eye = APPROACH["eye"](t)
    if DRONE_PATH is not None and DRONE_LIVE[0]:
        # it flies the whole cut: on station it is still in frame behind the sweep, and it
        # parks with the gulls once the eye goes under
        drone_set_pose(DRONE_PATH(min(t, T_END)), [0.0, WATER_Y + 0.6, 0.0])   # it films the pen throughout
        drone_tick(1.0 / 60.0)
    drone.visible = eye[1] > WATER_Y - 0.5
    gulls.position.y = 0.0 if eye[1] > WATER_Y - 0.5 else -9000.0
    tgt = APPROACH["tgt"](t) + float(pchip(*APPROACH["w"], t)) * (rov_pos - rov_at(t)[0])
    return eye, tgt, float(pchip(*APPROACH["fov"], t))


def cam_sonar(u, t):
    fwd = rov_R @ [1.0, 0.0, 0.0]
    inb = -np.array([rov_pos[0], 0.0, rov_pos[2]]) / np.hypot(rov_pos[0], rov_pos[2])
    return rov_pos - 2.2 * fwd + 0.7 * inb + [0, 0.9, 0], rov_pos + 1.5 * fwd + [0, -0.1, 0], 58.0


def net_clear(eye, r=0.7):
    """Keep the eye r metres off the cloth along the local sheet normal (a radial push can slide in-plane)."""
    for _ in range(3):
        d, q, n = net_nearest(eye)
        if d >= r:
            break
        h = float(np.dot(eye - q, n))
        side = math.copysign(1.0, h if abs(h) > 0.15 else -np.dot([q[0], 0.0, q[2]], n))   # ambiguous: inside
        eye = eye + (side * r - h) * n
    return eye


def cam_tear(u, t):
    if "tear" not in _anchor:                         # the ROV keeps creeping: anchor the push on its pose at the cut
        fwd = rov_R @ [1.0, 0.0, 0.0]
        _anchor["tear"] = rov_pos - 1.5 * fwd - 1.2 * e_t + [0, 0.6, 0]   # the ROV creeps along +e_t: stay on the other side
    tc = tear_now()
    a, b = _anchor["tear"], tc - 1.9 * e_r - 1.15 * e_t + [0, 0.35, 0]
    e = smooth(u, hold=0.3)                           # the push stops 0.8 m off the twine and holds the last 3 s
    return a + (b - a) * e, tc + 0.3 * e_r * e, 50.0


def cam_hero(u, t):
    tc = tear_now()
    a = tc - 1.9 * e_r - 1.15 * e_t + [0, 0.35, 0]
    b = tc - 3.6 * e_r - 1.5 * e_t + [0, -1.7, 0]               # below the milling band, looking up at the hole and the window
    e = smooth(u, hold=0.15)
    return a + (b - a) * e, tc + (0.3 - 0.9 * e) * e_r + [0, 0.3 * e, 0], 50.0 + 8.0 * e


T_HOVER = _hover_i / 60.0
CUTS = [("approach", T_END, cam_approach, False), ("sonar", SONAR_S, cam_sonar, True),
        ("tear", 10.0, cam_tear, True), ("hero", 12.0, cam_hero, False)]
T_CUT = np.concatenate([[0.0], np.cumsum([c[1] for c in CUTS])])
TEST_T = (1.0, 4.0, 7.0, 10.0, 12.5, 14.5, 16.5, 18.0, 21.5, 25.0, 28.0, 34.0, 41.0, 46.0, 53.0, 58.0, 61.0)
POSTER_U = 0.9


TERRAIN_CLEAR = 3.0                                   # metres of air the eye keeps over the ground
TERRAIN_HIT = [0.0, 0.0]                              # worst violation found by the pre-scan: metres, film time


def terrain_clear(eye):
    """Never let the eye enter the fjord. Cheap: the seabed under the pen is 35 m down, so
    every underwater cut passes untouched; only an aerial that flew into the wall moves."""
    if GEO is None:
        return eye
    floor_y = GEO.height_at(float(eye[0]), float(eye[2])) + TERRAIN_CLEAR
    if eye[1] < floor_y:
        eye = np.array(eye, np.float64)
        eye[1] = floor_y
    return eye


def terrain_scan():
    """Walk the whole camera schedule against the ground BEFORE the fix and report the worst
    violation, so a path that grazes the wall is a printed number and not a surprise in cut 0."""
    if GEO is None:
        return
    # Cut 0 only: it is the one that starts 34 m in the air and flies in. The
    # other three sit at the pen under water, where the ground is 35 m below the
    # deepest of them -- and calling their cam functions here would cache
    # cam_tear's _anchor against the wrong ROV pose.
    if FILM or FILM_BENCH:                            # the aerial only exists in the film
        build_approach()
        g_y, worst, at = gulls.position.y, 0.0, 0.0
        DRONE_LIVE[0] = False                         # do not integrate attitude across a replay
        for i in range(int(T_END * 10) + 1):
            t = i / 10.0
            eye = np.asarray(cam_approach(min(t / T_END, 1.0), t)[0], np.float64)
            v = GEO.height_at(float(eye[0]), float(eye[2])) + TERRAIN_CLEAR - eye[1]
            if v > worst:
                worst, at = v, t
        DRONE_LIVE[0] = True
        gulls.position.y = g_y                        # cam_approach parks/unparks them: put it back
        TERRAIN_HIT[:] = [worst, at]
        print(f"terrain: approach path vs ground, worst clearance violation {worst:.2f} m at film t={at:.1f} s"
              f" ({'lifted by terrain_clear()' if worst > 0 else 'path is clear'}); "
              f"clearance floor {TERRAIN_CLEAR:.1f} m")
    # Is the pen in the wall's shadow at this sun? SUN_DIR is fixed (the tear
    # lighting depends on it), so the only honest thing to do is measure the
    # skyline along the sun bearing and print how far the sun clears it.
    el = math.degrees(math.asin(SUN_DIR[1]))
    rise = max(math.degrees(math.atan2(GEO.height_at(SUN_H[0] * r, SUN_H[1] * r) - WATER_Y, r))
               for r in range(20, 2000, 10))
    print(f"terrain: sun elevation {el:.1f} deg on bearing {math.degrees(math.atan2(SUN_H[0], SUN_H[1])):.0f} deg; "
          f"skyline along that bearing rises to {rise:.1f} deg -> sun clears the wall by {el - rise:.1f} deg")


def film_cam(cut, u, t):
    eye, tgt, fov = CUTS[cut][2](u, t)
    camera.fov = fov
    camera.update_projection_matrix()
    eye = net_clear(np.asarray(eye, np.float64))
    eye = terrain_clear(eye)
    d = net_nearest(eye)[0]
    if d < CAM_STATS[1]:
        CAM_STATS[1], CAM_STATS[2] = d, t
    CAM_STATS[0] = min(CAM_STATS[0], d)
    camera.position.set(*eye)
    camera.look_at(*tgt)


def world_t_rewind(dt):
    """Warm-up frames must not advance the story clock: the cut opens where the previous one ended."""
    global world_t
    world_t -= dt


def film_writer(path, preset="veryfast", crf="18"):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=None, macro_block_size=None,
                              ffmpeg_params=["-crf", crf, "-pix_fmt", "yuv420p", "-preset", preset])


def film_bench(n=90):
    """A/B/A/B over the first n frames of the descent cut: flush 3 + x264 slow vs flush 1 + veryfast (PROVISIONAL numbers)."""
    global PATROL_T0
    PATROL_T0 = -FILM_T_OFF
    ROV_HOLD[0] = T_CUT[2] - FILM_T_OFF - T_HOVER
    build_approach()
    dt, res = 1.0 / FPS, {}
    hud_park(False)
    for tag in ("A", "B", "A", "B"):
        flush, preset, crf = (3, "slow", "17") if tag == "A" else (1, "veryfast", "18")
        renderer.set_flush_frames(flush)
        w, prev, dup = film_writer(os.path.join(CAP_DIR, f"bench_{tag}.mp4"), preset, crf), None, 0
        t0 = time.perf_counter()
        for f in range(n):
            step(dt)
            film_cam(0, 0.0, T_DIVE + f * dt)
            renderer.render(scene, camera)
            px = renderer.read_pixels()
            w.append_data(px)
            dup += int(prev is not None and np.array_equal(px, prev))
            prev = px.copy()
        w.close()
        res.setdefault(tag, []).append(1e3 * (time.perf_counter() - t0) / n)
        print(f"  {tag} flush {flush} preset {preset}: {res[tag][-1]:.1f} ms/f, {dup} duplicate frames of {n}")
    for tag, v in res.items():
        print(f"bench {tag}: {np.mean(v):.1f} ms/f = {1e3 / np.mean(v):.1f} fps (PROVISIONAL, GPU shared)")


def run_film():
    global PATROL_T0
    from PIL import Image
    PATROL_T0 = -FILM_T_OFF
    ROV_HOLD[0] = T_CUT[2] - FILM_T_OFF - T_HOVER      # HOVER opens the tear cut
    LEAK_T0[0] = T_CUT[2] - 8.0
    build_approach()
    dt = 1.0 / FPS
    out = os.path.abspath(FILM_OUT)
    stem = os.path.splitext(out)[0]
    counts = [int(round(c[1] * FPS)) for c in CUTS]
    total = sum(counts)
    picks = set(np.linspace(0, total - 1, 16).round().astype(int).tolist())
    if FILM_TEST:
        picks = {int(T_CUT[k] * FPS) + int(q * (n - 1)) for k, n in enumerate(counts)
                 for q in ((0.05, 0.5, 0.95) if k else tuple(x / T_END for x in TEST_T))}
    print(f"film: {len(CUTS)} cuts, {total} frames = {total / FPS:.1f} s at {FPS} fps, {W}x{H}; "
          f"patrol offset {FILM_T_OFF:.1f} s, station hold {ROV_HOLD[0]:.1f} s")
    writer = None if FILM_TEST else film_writer(out)
    renderer.set_flush_frames(1)                      # one GPU frame per render(): the film reads every frame once
    sheet, rec, wall0 = [], 0, time.perf_counter()
    for k, (name, dur, _, hud) in enumerate(CUTS):
        n = counts[k]
        hud_park(hud)
        renderer.set_auto_exposure_speed(12.0)          # meter fast through the warm-up, cinematic during the cut
        film_cam(k, 0.0, 0.0)
        for _ in range(0 if FILM_TEST else WARMUP):
            step(dt)
            film_cam(k, 0.0, 0.0)
            renderer.render(scene, camera)
            world_t_rewind(dt)
        renderer.set_auto_exposure_speed(1.2)
        CAM_STATS[1] = float("inf")
        t0 = time.perf_counter()
        for f in range(n):
            u = f / max(n - 1, 1)
            step(dt)
            film_cam(k, u, f * dt)
            if writer is not None or any(rec + q in picks for q in range(21)):   # test mode: 20 warm frames per still
                if FILM_TEST:
                    renderer.set_auto_exposure_speed(1.2 if rec in picks else 12.0)
                renderer.render(scene, camera)
            if writer is not None or rec in picks:
                px = renderer.read_pixels()
                if writer is not None:
                    writer.append_data(px)
                if rec in picks:
                    sheet.append(Image.fromarray(px).resize((400, 225), Image.LANCZOS))
                    if FILM_TEST:
                        Image.fromarray(px).save(f"{stem}_c{k}_{f * 100 // max(n - 1, 1):02d}.png")   # native still
                if name == "hero" and f == int(POSTER_U * (n - 1)):
                    Image.fromarray(px).save(stem + "_poster.png")
            rec += 1
        print(f"  cut {k} {name:<10s} {n:4d} f  {1e3 * (time.perf_counter() - t0) / n:6.1f} ms/f  cam-net min {CAM_STATS[1]:.2f} m at {CAM_STATS[2]:.1f} s")
    if writer is not None:
        writer.close()
    cols = 4
    cs = Image.new("RGB", (400 * cols, 225 * ((len(sheet) + cols - 1) // cols)), (0, 0, 0))
    for i, im in enumerate(sheet):
        cs.paste(im, (400 * (i % cols), 225 * (i // cols)))
    cs.save(stem + "_contact.png")
    print(f"film -> {out}; contact {stem}_contact.png; wall {(time.perf_counter() - wall0) / 60:.1f} min")
    rov_report()
    print(f"camera: min net distance {CAM_STATS[0]:.2f} m")


if HEADLESS and SHOT and SHOT not in SHOTS:
    sys.exit(f"unknown shot {SHOT!r}; one of {', '.join(SHOTS)}")
rov_cam_place()
renderer.sim_time = 0.0
renderer.render(scene, camera)                 # a view shares the primary's pipelines: needs one frame first
ROV_VIEW = renderer.add_view(rov_cam, CAM_W, CAM_H)
if ROV_VIEW == 0:
    print("could not create the ROV camera view")
elif HUD_ON:
    PIP = renderer.set_view_display_rect(ROV_VIEW, HUD_M, H - HUD_M - CAM_H, CAM_W, CAM_H)
    if PIP:
        _hud_a[0].position.x = -9000.0             # the sprite copy is replaced by the composited view
arm_fish_interop()
arm_net_interop()
print(f"fish: {'interop' if fish_vk else 'host upload'}; net: {'interop' if net_vk else 'host upload'}; "
      f"ROV cam: {'pip' if PIP else 'readback'}")

terrain_scan()

if FILM_BENCH:
    film_bench()
elif FILM:
    run_film()
elif HEADLESS:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    place(camera, SHOT)
    frames = int(SECONDS * 60)
    WARM_B, t_bench = min(60, frames // 2), 0.0     # skip pipeline warm-up + the first tile bakes
    for i in range(frames):
        if i == WARM_B:
            t_bench = time.perf_counter()
        step()
        if SHOTS[SHOT] is None:
            place(camera, SHOT)
        renderer.render(scene, camera)
    t_bench, n_bench = time.perf_counter() - t_bench, frames - WARM_B
    renderer.save_frame(scene, camera, OUT)
    print(f"simulated {SECONDS:.1f} s ({frames} frames), wrote {OUT}")
    print(f"frame path: {1e3 * t_bench / n_bench:.1f} ms/f = {n_bench / t_bench:.1f} fps at {W}x{H}"
          f" over {n_bench} frames ({'terrain ON' if GEO is not None else 'terrain off'})")
    rov_report()
    if SHOT == "p1_tear":
        for _ in range(30):
            step()
            renderer.render(scene, camera)
        out_b = OUT[:-4] + "_b.png"
        renderer.save_frame(scene, camera, out_b)
        print(f"wrote {out_b} (+0.5 s)")
elif PROFILE:
    WARM, TIMED = 60, 300
    engine, n = {}, [0]

    def profile_frame():
        step()
        place(camera, "p3_hud")
        mark("other")
        renderer.render(scene, camera)
        mark("render")
        n[0] += 1
        if n[0] == WARM:
            prof.acc.clear()
        elif n[0] > WARM and prof.on:
            for k, v in renderer.frame_timings.items():
                engine[k] = engine.get(k, 0.0) + v

    while n[0] < WARM + TIMED and canvas.animate_once(profile_frame):
        pass
    prof.report(TIMED, engine)
    prof.on, n[0] = False, 0
    t0 = time.perf_counter()
    while n[0] < TIMED and canvas.animate_once(profile_frame):
        pass
    print(f"live, no syncs: {TIMED / (time.perf_counter() - t0):.1f} fps over {TIMED} frames")
else:
    place(camera, "p1_tear")
    camera.position.set(*(TEAR_C - 4.0 * e_r + 1.5 * e_t + [0, 0.6, 0]))
    orbit_loop(canvas, renderer, scene, camera, step, target=tuple(TEAR_C))

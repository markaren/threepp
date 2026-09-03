"""Net-pen inspection ROV: Warp-cloth net with a tear, collar, BlueROV2, and a Warp school of procedural salmon.

    python warp_netpen.py                          # window; drag to orbit, Esc quits
    python warp_netpen.py --shot p1_tear           # still -> aaa_caps/netpen/p1_tear.png
    python warp_netpen.py --shot p1_rov_hero --out x.png --seconds 8 --size 1600x900
Cameras: p1_net_wide p1_collar_below p1_rov_hero p1_tear p2_school p2_fish_close p2_leak. `--fish N` (400).
Vulkan only; Warp on CUDA if present.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import cli_arg, orbit_loop, parse_size, sky_env, standard_material

SHOT = cli_arg("--shot", "", str)
HEADLESS = bool(SHOT)
SECONDS = cli_arg("--seconds", 6.0, float)
W, H = parse_size(cli_arg("--size", "1600x900", str))
CAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "aaa_caps", "netpen")
OUT = cli_arg("--out", os.path.join(CAP_DIR, f"{SHOT}.png"), str)

# ---- world ------------------------------------------------------------------
WATER_Y = 0.0
PEN_R, PEN_D = 7.0, 8.0
SUN_DIR = np.array([0.62, 0.55, 0.30]); SUN_DIR /= np.linalg.norm(SUN_DIR)
SUN_H = SUN_DIR[[0, 2]] / np.linalg.norm(SUN_DIR[[0, 2]])
TEAR_TH = math.atan2(-SUN_H[1], -SUN_H[0])          # the wall the sun lights from inside
TEAR_Y, TEAR_R = -3.0, 0.65
MESH_M, TILE_MESHES = 0.045, 4
TILE_M = MESH_M * TILE_MESHES
MURK_SIGMA = 0.14
MURK_COLOR = tp.Color(0.018, 0.100, 0.108)

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
TONGUES = [(1.55, 0.52, 0.20), (2.85, 0.42, 0.15), (-0.30, 0.36, 0.13)]   # (rim angle, length, half-width)


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


net = Net()
graph = None
if device.is_cuda:
    with wp.ScopedCapture(device) as cap:
        net.launches()
    graph = cap.graph


def net_step(current, t):
    net.push_uniforms(current, t)
    if graph is not None:
        wp.capture_launch(graph)
    else:
        net.launches()


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
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
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
sky = sky_env(SUN_DIR, below_horizon=(0.30, 0.42, 0.48), below_nadir=(0.08, 0.16, 0.20))
scene.environment = sky
scene.background = sky
sun = tp.DirectionalLight(0xfff0d8, 3.2)
sun.position.set(*(SUN_DIR * 1000.0))
scene.add(sun)

ocean = tp.Ocean(size=320.0, resolution=384, wind_speed=4.5, wind_theta=0.6,
                 choppiness=0.5, fft_size=512, fetch=15e3)
ocean.params.foam_amount = 0.0                 # whitecaps print as white slabs from below
if "--no-ocean" not in sys.argv:
    scene.add(ocean)
floor = tp.Mesh(tp.PlaneGeometry(400.0, 400.0), standard_material(0x03060a))
floor.rotate_x(-math.pi / 2)
floor.position.y = -45.0
scene.add(floor)
renderer.set_fog_water_surface_y(WATER_Y)
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
for geo, mat in ((net_geo, net_mat), (patch_geo, patch_mat), (foul_geo, foul_mat)):
    m = tp.Mesh(geo, mat)
    m.frustum_culled = False
    m.cast_shadow = True
    m.receive_shadow = True
    scene.add(m)


def net_upload():
    pos = net.pos.numpy()
    nrm = net.nrm.numpy()
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
        disc.rotation.y = math.pi / 2
        disc.position.set(0.2135, -0.075, z)
        rov.add(disc)
        spot = tp.SpotLight(tp.Color(1.0, 0.95, 0.85), 220.0, 14.0, math.radians(21.0), 0.55, 2.0)
        spot.position.set(0.215, -0.075, z)
        tgt = tp.Group()
        tgt.position.set(4.0, -0.35, z)
        rov.add(tgt)
        spot.set_target(tgt)
        rov.add(spot)
        lights.append(spot)
    pts = [np.array([-0.19, 0.11, 0.0])]
    for k in range(1, 9):
        t = k / 8.0
        pts.append(np.array([-0.19 - 0.9 * t, 0.11 + 1.6 * t * t, 0.35 * t]))
    for a, b in zip(pts[:-1], pts[1:]):
        rov.add(tube(a, b, 0.006, cable, 8))
    return rov, props, lights


rov, props, rov_lights = build_rov()
scene.add(rov)
ROV_TH = TEAR_TH - math.radians(40.0)
ROV_R = PEN_R - 1.5
ROV_Y = -2.2
rov.position.set(ROV_R * math.cos(ROV_TH), ROV_Y, ROV_R * math.sin(ROV_TH))
ROV_YAW = -ROV_TH - math.pi / 2                   # +X forward along the wall toward the tear
rov.rotation.y = ROV_YAW

e_r = np.array([math.cos(TEAR_TH), 0.0, math.sin(TEAR_TH)])
e_t = np.array([-math.sin(TEAR_TH), 0.0, math.cos(TEAR_TH)])
TEAR_C = np.array([PEN_R * e_r[0], TEAR_Y, PEN_R * e_r[2]])

# ---- fish: procedural salmon v2 ---------------------------------------------
FISH_N = max(cli_arg("--fish", 400, int), 2)
F_RINGS, F_SIDES = 24, 24
F_K = 5.6
FISH_SPEED_MIN, FISH_SPEED_MAX = 0.25, 1.6
MILL_SPEED = 0.75
LEAK_FRAC = 0.05
EYE_U, EYE_TH = 0.085, 0.25
TEX_W, TEX_H, BODY_ROWS = 1024, 512, 392
BODY_V = BODY_ROWS / TEX_H
FIN_BANDS = {k: ((402 + 36 * k) / TEX_H, (434 + 36 * k) / TEX_H) for k in range(3)}   # dark fins / pectoral / pelvic+anal


def catmull(y, u):
    """Catmull-Rom through the control values, sampled at u in [0, 1]."""
    n = len(y)
    x = np.clip(u, 0.0, 1.0) * (n - 1)
    i = np.clip(np.floor(x).astype(int), 0, n - 2)
    t = x - i
    p0, p1, p2, p3 = (y[np.clip(i + k, 0, n - 1)] for k in (-1, 0, 1, 2))
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


HALF_H = np.float32([0.010, 0.048, 0.078, 0.096, 0.106, 0.106, 0.099, 0.086, 0.069, 0.049, 0.030, 0.014])
HALF_W = np.float32([0.008, 0.032, 0.049, 0.059, 0.063, 0.061, 0.056, 0.048, 0.038, 0.027, 0.016, 0.006])
KEEL = np.float32([1.0, 1.05, 1.15, 1.22, 1.25, 1.24, 1.20, 1.14, 1.08, 1.02, 1.0, 1.0])


def skin_fn(u, th):
    """The skin at body fraction u (0 nose, 1 tail root) and ring angle th; +Z is the nose."""
    h, w, kb = catmull(HALF_H, u), catmull(HALF_W, u), catmull(KEEL, u)
    s, c = np.sin(th), np.cos(th)
    eye = 1.0 + 0.10 * np.exp(-((u - EYE_U) / 0.035) ** 2) * (np.exp(-(1 - np.cos(th - EYE_TH)) / 0.05)
                                                              + np.exp(-(1 - np.cos(th + EYE_TH - np.pi)) / 0.05))
    y = h * s * np.where(s < 0, kb * (1.0 - 0.18 * c * c), 1.0) * eye - 0.15 * h
    return np.stack([w * c * eye, y, 0.5 - u], -1)


def fan(outline, nrm, ray, camber, shift=(0.0, 0.0, 0.0), band=0, flip=False):
    """A thin fin fanned from its centre, cambered along nrm; uv = (across the rays, root->tip) in the fin band."""
    outline = np.asarray(outline, np.float64) + np.float64(shift)
    v = np.concatenate([[outline.mean(0)], outline])
    tris = [[0, 1 + k, 1 + (k + 1) % len(outline)] for k in range(len(outline) - 1)]
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


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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
    add(fan([[0, top(x) + h, z(x)] for x, h in ((0.45, 0), (0.46, 0.020), (0.475, 0.028), (0.565, 0), (0.51, 0))],
            [1, 0, 0], [0, 1, -0.5], 0.004), 1)
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
    col = col * (1 + (sheen - 1) * t1 * (1 - t2)) * (1 + 0.08 * n[..., None] * np.float32([1.0, 1.1, 0.8]))
    col *= 1 - 0.12 * np.exp(-((d - 0.46) * rows / 1.6) ** 2)[..., None] * (U > 0.21)[..., None]      # lateral line
    ue = 0.19 + 0.03 * np.sin(np.pi * np.clip((d - 0.05) / 0.9, 0, 1))                                 # gill cover rear edge
    ge = (U - ue) * TEX_W
    scl = np.clip((U - ue) / 0.03, 0, 1)                # scales start behind the gill cover
    a1, a2 = px / 11.4 + py * ysc / 11.4, px / 11.4 - py * ysc / 11.4
    ta, tb = np.abs(a1 % 1 - 0.5), np.abs(a2 % 1 - 0.5)
    edge = np.exp(-(np.minimum(ta, tb) / 0.09) ** 2)
    col *= (1 - 0.16 * edge * scl)[..., None]                                                          # ~90 diamond scales
    head = (U < ue)[..., None]
    col = np.where(head, col * np.float32([0.80, 0.86, 0.84]) * (1 + 0.18 * ((U > 0.145) & (d > 0.18)))[..., None], col)
    col *= (1 - 0.55 * np.exp(-(ge / 1.5) ** 2) + 0.15 * np.exp(-((ge - 4) / 2.5) ** 2) * (ge > 0))[..., None]
    col *= (1 - 0.25 * np.exp(-((U - 0.145) * TEX_W / 1.5) ** 2) * (d > 0.18))[..., None]              # front crease
    for sg in (1, -1):
        vm = 0.5 + sg * (0.30 + 0.06 * U / 0.125)
        m = np.exp(-((V - vm) * BODY_ROWS / 2.2) ** 2) * (U < 0.125) * np.clip((0.125 - U) / 0.02, 0, 1)
        col *= (1 - 0.7 * m)[..., None]                                                                # mouth line
        jaw = ((sg * (V - 0.5) > 0.30 + 0.06 * U / 0.125) & (U < 0.12))[..., None]
        col = np.where(jaw, col * 0.5 + np.float32([0.62, 0.50, 0.48]) * 0.5, col)                     # pinkish lower jaw
        r = np.hypot(px - 0.055 * TEX_W, (py - (0.5 - sg * 0.20) * BODY_ROWS) * ysc)
        col *= (1 - 0.8 * np.clip((1.6 - r) * 2, 0, 1))[..., None]                                     # nostril
    X, Y = px, py * ysc
    cov = np.zeros_like(U)
    for dx in (-1, 0, 1):                              # sparse X / M marks on the back, 2-3 scales wide
        for dy in (-1, 0, 1):
            cx, cy = 52, 26
            jx, jy = (px // cx).astype(int) + dx + 8, (py // cy).astype(int) + dy + 8
            sx = (jx - 8 + 0.15 + 0.7 * hash01(jx, jy, 1)) * cx
            sy = (jy - 8 + 0.15 + 0.7 * hash01(jx, jy, 2)) * cy
            su, sd = sx / TEX_W, np.abs(sy / BODY_ROWS - 0.5) / 0.5
            p = 0.33 * (1 - sm(sd, 0.34, 0.44)) * (su > ue.mean())
            on = hash01(jx, jy, 3) < p
            ang = 0.6 + 0.5 * hash01(jx, jy, 5)
            ln = 11 + 6 * hash01(jx, jy, 6)
            rx, ry = X - sx, (Y - sy * ysc)
            for k in range(3):
                th = ang * (1 if k == 0 else -1) + (0.0 if k < 2 else 1.4)
                lx = rx * np.cos(th) + ry * np.sin(th)
                ly = -rx * np.sin(th) + ry * np.cos(th)
                dist = np.hypot(np.maximum(np.abs(lx) - ln * (0.5 if k == 2 else 1.0), 0), ly)
                use = on if k < 2 else on & (hash01(jx, jy, 7) < 0.4)
                cov = np.maximum(cov, np.clip((1.7 - dist) * 1.2 + 0.5, 0, 1) * use)
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
        disc = lambda r0: np.clip((r0 - r) * 12.0, 0, 1)[..., None]
        col *= 1 + 0.12 * (disc(1.32) - disc(1.12))
        col = col * (1 - disc(1.12)) + np.float32([0.03, 0.035, 0.04]) * disc(1.12)
        col = col * (1 - disc(0.95)) + (np.float32([0.75, 0.55, 0.12]) * (0.7 + 0.45 * r)[..., None]) * disc(0.95)
        col = col * (1 - disc(0.52)) + np.float32([0.008, 0.008, 0.01]) * disc(0.52)
    tex = np.tile(np.float32([0.05, 0.06, 0.065]), (TEX_H, TEX_W, 1))
    tex[:BODY_ROWS] = col
    hgt = np.zeros((TEX_H, TEX_W), np.float32)
    hgt[:BODY_ROWS] = (ta * tb) * scl * 4.0                # diamond scale domes for the normal map
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
    nrm = np.stack([-gx * 1.5, gy * 1.5, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nmap = np.full((TEX_H, TEX_W, 4), 255, np.uint8)
    nmap[..., :3] = np.clip((nrm * 0.5 + 0.5) * 255, 0, 255)
    return tex, nmap


def fish_texture():
    tex, nmap = fish_albedo()
    return tp.data_texture(srgb8(tex), srgb=True), tp.data_texture(nmap, srgb=False)


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
    leaking = leak_t[i] >= 0.0 and t > leak_t[i]
    near_tear = wp.length(p - tear) < 1.4
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
        want = wp.normalize(goal - p) * 0.85
        acc += (want - v) * 2.5
    else:
        want = tan * MILL_SPEED - rad * (r - pref[i][0]) * 0.45 + wp.vec3(0.0, (pref[i][1] - p[1]) * 0.45, 0.0)
        acc += (want - v) * 1.1
    dwall = pen_r - r
    if not (leaking and near_tear):
        if dwall > 0.0 and dwall < 1.2:
            acc -= rad * (1.2 - dwall) * 6.0
        if dwall <= 0.0 and dwall > -1.0:
            acc += rad * (1.0 + dwall) * 6.0
    if p[1] > -0.7:
        acc += wp.vec3(0.0, -6.0 * (p[1] + 0.7), 0.0)
    if p[1] < -pen_d + 1.0:
        acc += wp.vec3(0.0, 4.0 * (-pen_d + 1.0 - p[1]), 0.0)
    dr = p - rov
    lr = wp.length(dr)
    if lr < 2.0:
        acc += dr / wp.max(lr, 1.0e-3) * (1.0 - lr / 2.0) * 5.0
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
        pref = np.stack([rng.uniform(3.2, 5.6, FISH_N), rng.uniform(-4.2, -1.4, FISH_N), rng.uniform(-0.25, 0.25, FISH_N)], 1)
        leak = np.full(FISH_N, -1.0, np.float32)
        nl = max(int(LEAK_FRAC * FISH_N), 3)
        who = rng.choice(np.arange(1, FISH_N), nl, replace=False)      # fish 0 is the close-up hero
        leak[who] = 0.5 + 0.9 * np.arange(nl)
        th = rng.uniform(0.0, 2 * np.pi, FISH_N)
        rr = rng.uniform(2.6, 6.0, FISH_N)
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
        self.uni = wp.zeros(4, dtype=wp.vec3, device=device)
        self.c0, self.c1, self.n0, self.n1, self.uu = A(c0, wp.vec3), A(c1, wp.vec3), A(n0, wp.vec3), A(n1, wp.vec3), A(uu)
        self.out_p = wp.zeros(FISH_N * self.nv, dtype=wp.vec3, device=device)
        self.out_n = wp.zeros(FISH_N * self.nv, dtype=wp.vec3, device=device)

    def step(self, t, dt, rov_pos):
        self.uni.assign(np.asarray([rov_pos, TEAR_C, e_r, [PEN_R, PEN_D, 0.0]], np.float32))
        wp.launch(boids, dim=FISH_N, device=device,
                  inputs=[self.pos, self.vel, self.pos2, self.vel2, self.yaw, self.pitch, self.roll, self.beat,
                          self.amp, self.brake, self.len, self.pref, self.leak, self.uni, FISH_N, t, dt])
        self.pos, self.pos2 = self.pos2, self.pos
        self.vel, self.vel2 = self.vel2, self.vel
        wp.launch(skin, dim=FISH_N * self.nv, device=device,
                  inputs=[self.c0, self.c1, self.n0, self.n1, self.uu, self.pos, self.yaw, self.pitch, self.roll,
                          self.beat, self.amp, self.brake, self.len, self.out_p, self.out_n, self.nv])
        return self.out_p.numpy(), self.out_n.numpy()

    def hero(self):
        """(position, forward) of fish 0 on the host."""
        p, y = self.pos.numpy()[0], float(self.yaw.numpy()[0])
        return p, np.array([math.sin(y), 0.0, math.cos(y)])


school = School()
_fp0, _fn0 = school.step(0.0, 1.0 / 60.0, np.zeros(3))
fish_mat = tp.MeshPhysicalMaterial()
fish_mat.color = 0xffffff
fish_mat.roughness, fish_mat.metalness = 0.42, 0.15
fish_mat.specular_intensity = 0.2      # skin/water IOR contrast is small: F0 ~0.008, else the sky env chromes the back
fish_mat.side = tp.Side.Double
fish_mat.vertex_colors = True
fish_mat.map, fish_mat.normal_map = fish_texture()
fish_mat.normal_scale = tp.Vector2(0.35, 0.35)
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


def fish_step(t, dt):
    p, n = school.step(t, dt, np.array([rov.position.x, rov.position.y, rov.position.z]))
    fish_geo.update_attribute("position", p)
    fish_geo.update_attribute("normal", n)


# ---- marine snow --------------------------------------------------------------
MOTE_CAP, MOTE_HALF, MOTE_TOP, MOTE_BOTTOM = 20_000, 9.0, 0.0, -12.0
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
_me.velocity = tp.Vector3(0.0, 0.022, 0.0)
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

# ---- cameras -----------------------------------------------------------------
camera = tp.PerspectiveCamera(60.0, W / H, 0.05, 600.0)
rov_p = np.array([rov.position.x, rov.position.y, rov.position.z])
rov_f = np.array([math.cos(ROV_YAW), 0.0, -math.sin(ROV_YAW)])
sun_h3 = np.array([SUN_H[0], 0.0, SUN_H[1]])
SHOTS = {
    "p1_net_wide": (1.0 * e_r + [0, -2.6, 0], TEAR_C + [0, 0.6, 0], 62.0),
    "p1_collar_below": (-1.5 * sun_h3 + [0, -5.0, 0], 0.85 * PEN_R * sun_h3 + [0, 0.3, 0], 78.0),
    "p1_rov_hero": (rov_p - 0.85 * e_r + 1.15 * rov_f + [0, 0.30, 0], rov_p + [0, 0.02, 0], 50.0),
    "p1_tear": (TEAR_C - 2.0 * e_r + [0, 0.15, 0], TEAR_C, 55.0),
    "p2_school": (-1.2 * sun_h3 + [0, -6.2, 0], 4.5 * sun_h3 + [0, -0.6, 0], 72.0),
    "p2_fish_close": None,
    "p2_leak": (TEAR_C + 2.6 * e_r - 1.4 * e_t + [0, 0.35, 0], TEAR_C + 0.6 * e_r, 52.0),
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
    p, t, fov = SHOTS[key]
    cam.fov = fov
    cam.update_projection_matrix()
    cam.position.set(*p)
    cam.look_at(*t)


# ---- step --------------------------------------------------------------------
world_t = 0.0


def step(dt=1.0 / 60.0):
    global world_t
    world_t += dt
    sp = 0.15 * (1.0 + 0.35 * math.sin(0.21 * world_t))
    ang = 0.35 + 0.25 * math.sin(0.09 * world_t)
    net_step(np.array([sp * math.cos(ang), 0.0, sp * math.sin(ang)], np.float32), world_t)
    net_upload()
    fish_step(world_t, dt)
    for k, p in enumerate(props):
        p.rotation.y += dt * (38.0 + 6.0 * (k % 3)) * (1 if k % 2 else -1)
    rov.position.y = ROV_Y + 0.03 * math.sin(0.7 * world_t)
    rov.rotation.z = 0.02 * math.sin(0.5 * world_t + 1.0)
    rov.rotation.x = 0.015 * math.sin(0.43 * world_t)
    cp = camera.position
    motes.set_follow_center(tp.Vector3(cp.x, cp.y, cp.z))
    motes.set_emitter_time(world_t, dt)
    motes.billboard_repr.intensity = MOTE_BASE * (1.0 + 0.9 * min(max(-cp.y, 0.0), 8.0) / 8.0)


if HEADLESS:
    if SHOT not in SHOTS:
        sys.exit(f"unknown shot {SHOT!r}; one of {', '.join(SHOTS)}")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    place(camera, SHOT)
    frames = int(SECONDS * 60)
    for _ in range(frames):
        step()
        if SHOT == "p2_fish_close":
            place(camera, SHOT)
        renderer.render(scene, camera)
    renderer.save_frame(scene, camera, OUT)
    print(f"simulated {SECONDS:.1f} s ({frames} frames), wrote {OUT}")
    if SHOT == "p1_tear":
        for _ in range(30):
            step()
            renderer.render(scene, camera)
        out_b = OUT[:-4] + "_b.png"
        renderer.save_frame(scene, camera, out_b)
        print(f"wrote {out_b} (+0.5 s)")
else:
    place(camera, "p1_tear")
    camera.position.set(*(TEAR_C - 4.0 * e_r + 1.5 * e_t + [0, 0.6, 0]))
    orbit_loop(canvas, renderer, scene, camera, step, target=tuple(TEAR_C))

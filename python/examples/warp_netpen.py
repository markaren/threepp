"""Net-pen inspection ROV, phase 1: the pen, its Warp-cloth net, the collar and a BlueROV2 in the murk.

    python warp_netpen.py                          # window; drag to orbit, Esc quits
    python warp_netpen.py --shot p1_tear           # still -> aaa_caps/netpen/p1_tear.png
    python warp_netpen.py --shot p1_rov_hero --out x.png --seconds 8 --size 1600x900
Cameras: p1_net_wide  p1_collar_below  p1_rov_hero  p1_tear. Vulkan only; Warp on CUDA if present.
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


def wall_coords(p):
    """(arc offset from the tear meridian, height above the tear centre) for a wall point."""
    dth = (math.atan2(p[2], p[0]) - TEAR_TH + math.pi) % (2.0 * math.pi) - math.pi
    return PEN_R * dth, p[1] - TEAR_Y


def tear_radius(phi):
    return TEAR_R * (1.0 + 0.30 * math.sin(3.0 * phi + 1.1) + 0.18 * math.sin(7.0 * phi + 0.4) + 0.10 * math.sin(13.0 * phi + 2.0))


def in_tear(p):
    s, d = wall_coords(p)
    return math.hypot(s, d) < tear_radius(math.atan2(d, s))


def cut_edge(pa, pb):
    """Cut where the edge crosses the ragged rim (except its bottom hinge arc) or the centre slit."""
    ia, ib = in_tear(pa), in_tear(pb)
    sa, da = wall_coords(pa)
    sb, db = wall_coords(pb)
    if ia != ib:
        phi = math.atan2(0.5 * (da + db), 0.5 * (sa + sb))
        return not (-2.6 < phi < -0.55)
    return ia and ib and (sa * sb < 0.0) and 0.5 * (da + db) > -0.25


def net_mask(rest):
    mask = np.zeros(NU * NVT, np.int32)
    inside = np.zeros(NU * NVT, bool)
    for iv in range(NV):
        for iu in range(NU):
            i = iv * NU + iu
            if abs(wall_coords(rest[i])[0]) > 2.0 * TEAR_R or abs(rest[i][1] - TEAR_Y) > 2.0 * TEAR_R:
                continue
            inside[i] = in_tear(rest[i])
            for bit, (du, dv) in enumerate(OFFS):
                ju, jv = (iu + du) % NU, iv + dv
                if 0 <= jv < NV and cut_edge(rest[i], rest[jv * NU + ju]):
                    mask[i] |= 1 << bit
    return mask, inside


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
        mask, self.inside = net_mask(rest)
        iv = np.arange(self.n) // NU
        inv_mass = np.ones(self.n, np.float32)
        inv_mass[iv == 0] = 0.0                       # hung from the collar
        load = np.full(self.n, 0.10, np.float32)      # twine: barely negative buoyancy
        load[iv == NV - 1] = 2.4                      # sinker tube
        load[self.inside] = 0.55                      # torn flaps droop
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
def vnoise(h, w, cy, cx, rng):
    """Periodic-in-x value noise on a (h, w) grid with (cy, cx) cells."""
    g = rng.random((cy + 1, cx + 1)).astype(np.float32)
    g[:, cx] = g[:, 0]
    y = np.linspace(0, cy, h, endpoint=False)
    x = np.linspace(0, cx, w, endpoint=False)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = (y - y0)[:, None], (x - x0)[None, :]
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    return (g[y0][:, x0] * (1 - tx) * (1 - ty) + g[y0][:, x0 + 1] * tx * (1 - ty)
            + g[y0 + 1][:, x0] * (1 - tx) * ty + g[y0 + 1][:, x0 + 1] * tx * ty)


def fbm(h, w, cy, cx, rng, octaves=5):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        out += amp * vnoise(h, w, cy << k, cx << k, rng)
        tot += amp
        amp *= 0.5
    return out / tot


def net_tile(px=256, twine=0.11):
    """Knotted square mesh: (albedo+alpha, normal map) tiles of TILE_MESHES meshes."""
    rng = np.random.default_rng(3)
    u = (np.arange(px) + 0.5) / px * TILE_MESHES
    U, V = np.meshgrid(u, u)
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
    fibre = 0.82 + 0.36 * fbm(px, px, 16, 16, rng)
    twist = 0.5 + 0.5 * np.sin(2 * np.pi * (V * 18 + U * 18)) * (hv > hh) + 0.5 * np.sin(2 * np.pi * (U * 18 - V * 18)) * (hv <= hh)
    base = np.float32([0.40, 0.44, 0.34])
    col = base[None, None, :] * (fibre * (0.85 + 0.3 * twist))[..., None]
    col[hk > 0.98] *= 0.7
    rgba = np.zeros((px, px, 4), np.uint8)
    rgba[..., :3] = np.clip(col * 255, 0, 255)
    rgba[..., 3] = np.where(hgt > 0.0, 255, 0)
    gy, gx = np.gradient(hgt * 0.5 * hw * px / TILE_MESHES * 0.012)
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nmap = np.full((px, px, 4), 255, np.uint8)
    nmap[..., :3] = np.clip((nrm * 0.5 + 0.5) * 255, 0, 255)
    return tp.data_texture(rgba, srgb=True), tp.data_texture(nmap, srgb=False)


def fouling_maps(rest):
    """Biofouling: an unwrapped wall texture of clumps plus a per-particle tint weight."""
    rng = np.random.default_rng(11)
    h, w = 1024, 2048
    f = fbm(h, w, 18, 72, rng)
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    th = 2 * np.pi * u
    sunlit = 0.5 + 0.5 * np.cos(th - TEAR_TH)
    depth_w = np.clip(1.15 - 1.6 * v * NVT / NV, 0.0, 1.0)      # algae wants light
    weight = (0.35 + 0.65 * sunlit)[None, :] * depth_w[:, None]
    weight[v > (NV - 1) / (NVT - 1)] = 0.05
    thr = 0.70 - 0.24 * weight
    alpha = np.clip((f - thr) / 0.04, 0, 1)
    fine = fbm(h, w, 96, 384, np.random.default_rng(5))
    alpha *= np.clip(0.35 + 1.6 * (fine - 0.40), 0, 1)
    tint = fbm(h, w, 12, 48, np.random.default_rng(9))[..., None]
    col = np.float32([0.36, 0.42, 0.15]) * (1 - tint) + np.float32([0.30, 0.24, 0.10]) * tint
    col *= (0.6 + 0.8 * fine)[..., None]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = np.clip(col * 255, 0, 255)
    rgba[..., 3] = np.clip(alpha * 255, 0, 255)
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

tile_tex, tile_nrm = net_tile()
foul_tex, foul_w = fouling_maps(net.rest_host)
net_mat = standard_material(0xffffff, roughness=0.78, metalness=0.0, side=tp.Side.Double)
net_mat.map = tile_tex
net_mat.normal_map = tile_nrm
net_mat.normal_scale = tp.Vector2(0.8, 0.8)
net_mat.alpha_test = 0.30
net_mat.vertex_colors = True
foul_mat = standard_material(0xffffff, roughness=0.92, metalness=0.0, side=tp.Side.Double)
foul_mat.map = foul_tex
foul_mat.alpha_test = 0.5

tint = np.float32([0.55, 0.62, 0.30])
vcol = (1.0 - foul_w[GATHER][:, None]) * 1.0 + foul_w[GATHER][:, None] * tint[None, :]


def net_index():
    """Grid triangles minus every one that spans a cut spring, so the tear is a real hole."""
    mask = net.mask.numpy()
    iv, iu = np.meshgrid(np.arange(NVT - 1), np.arange(NU), indexing="ij")
    a, b = iv * NU + iu, iv * NU + (iu + 1) % NU
    c, d = b + NU, a + NU
    bit = lambda idx, k: (mask[idx] >> k) & 1
    ab, bc, cd, da, ac = bit(a, 1), bit(b, 3), bit(c, 0), bit(d, 2), bit(a, 7)
    ga, gb = iv * (NU + 1) + iu, iv * (NU + 1) + iu + 1
    gc, gd = gb + NU + 1, ga + NU + 1
    t1 = np.stack([ga, gb, gc], -1)[(ab | bc | ac) == 0]
    t2 = np.stack([ga, gc, gd], -1)[(ac | cd | da) == 0]
    return np.concatenate([t1, t2]).reshape(-1).astype(np.uint32)


NET_INDEX = net_index()
net_geo = tp.PlaneGeometry(1.0, 1.0, NU, NVT - 1)
net_geo.set_attribute("position", rest_g)
net_geo.set_attribute("uv", uv_net)
net_geo.set_attribute("color", vcol.astype(np.float32))
net_geo.set_index(NET_INDEX)
foul_geo = tp.PlaneGeometry(1.0, 1.0, NU, NVT - 1)
foul_geo.set_attribute("position", rest_g)
foul_geo.set_attribute("uv", uv_foul)
foul_geo.set_index(NET_INDEX)
for geo, mat in ((net_geo, net_mat), (foul_geo, foul_mat)):
    m = tp.Mesh(geo, mat)
    m.frustum_culled = False
    m.cast_shadow = True
    m.receive_shadow = True
    scene.add(m)


def net_upload():
    pos = net.pos.numpy()
    nrm = net.nrm.numpy()
    net_geo.update_attribute("position", pos[GATHER])
    net_geo.update_attribute("normal", nrm[GATHER])
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
e_r = np.array([math.cos(TEAR_TH), 0.0, math.sin(TEAR_TH)])
e_t = np.array([-math.sin(TEAR_TH), 0.0, math.cos(TEAR_TH)])
TEAR_C = np.array([PEN_R * e_r[0], TEAR_Y, PEN_R * e_r[2]])
rov_p = np.array([rov.position.x, rov.position.y, rov.position.z])
rov_f = np.array([math.cos(ROV_YAW), 0.0, -math.sin(ROV_YAW)])
sun_h3 = np.array([SUN_H[0], 0.0, SUN_H[1]])
SHOTS = {
    "p1_net_wide": (1.0 * e_r + [0, -2.6, 0], TEAR_C + [0, 0.6, 0], 62.0),
    "p1_collar_below": (-1.5 * sun_h3 + [0, -5.0, 0], 0.85 * PEN_R * sun_h3 + [0, 0.3, 0], 78.0),
    "p1_rov_hero": (rov_p - 0.85 * e_r + 1.15 * rov_f + [0, 0.30, 0], rov_p + [0, 0.02, 0], 50.0),
    "p1_tear": (TEAR_C - 2.0 * e_r + [0, 0.15, 0], TEAR_C, 55.0),
}


def place(cam, key):
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

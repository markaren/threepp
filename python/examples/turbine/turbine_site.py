"""The offshore wind turbine inspection site: the IEA 15 MW turbine on its monopile in 30 m of
North Sea, its rock scour protection, and the export cable in its cable protection system.

turbine.glb is what build_turbine_blender.py makes from turbine_spec.json. The seabed is ONE
analytic function, seabed_height(x, z); the mesh is built from it and a later sonar phase
scores against it.

    python turbine_site.py --shot site_far --out site_far.png [--size 1920x1080]
    python turbine_site.py --shot all --tag v01 [--outdir <dir>]

Importable without rendering: build_site(renderer) returns the scene and its helpers (see
its docstring); --shot rendering lives in main(). Headless only.
Frame: X upwind (the rotor is on +X of the tower), Y up, origin on the pile axis at MSL.
Needs a Vulkan build and turbine.glb.
"""
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(_HERE))                     # examples/ (demo_common)
sys.path.insert(0, _HERE)

import json

import threepp as tp
from demo_common import cli_arg, parse_size

import build_turbine_blender as tb
from turbine_out import out_dir

GLB = os.path.join(_HERE, "turbine.glb")
with open(os.path.join(_HERE, "turbine_spec.json"), encoding="utf-8") as fh:
    spec = json.load(fh)

T_SHOT = 12.0                     # the default still time (s of sea state)

# --------------------------------------------------------------------------- #
#  The seabed: ONE analytic function (ground truth for the sonar phase)
# --------------------------------------------------------------------------- #
DEPTH = 30.0
PILE_R = 0.5 * tb.val(spec["monopile"]["outer_diameter"])
R_ARMOUR = 22.5                   # armour radius, 2.25 pile diameters
T_ARMOUR = 1.2                    # armour layer thickness
TOE_W = 3.0                       # width of the berm's outer slope
CURRENT_AZ = math.radians(tb.val(spec["site"]["current_toward_deg"]))   # edge scour on this side
PIT_DEPTH, PIT_R, PIT_W, PIT_HALF_DEG = 2.0, R_ARMOUR + 3.0, 4.5, 40.0
DAMAGE_AZ, DAMAGE_HALF_DEG = math.radians(-95.0), 13.0                   # armour displaced here
CABLE_AZ = math.radians(spec["empties"]["cable_entry"]["azimuth_deg"])
STONE = 0.5                       # armour stone cell (m): stones 0.3-0.6 m
_brng = np.random.default_rng(5)
BOULDERS = [(r * math.cos(a), r * math.sin(a), br, bh) for r, a, br, bh in zip(
    np.concatenate([_brng.uniform(27.0, 40.0, 5), _brng.uniform(52.0, 160.0, 12)]), _brng.uniform(-np.pi, np.pi, 17),
    _brng.uniform(0.5, 1.3, 17), _brng.uniform(0.4, 1.0, 17))]      # glacial boulders on the sand


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash(i, j, k):
    return np.modf(np.abs(np.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453))[0]


def _stones(x, z, cell, seed):
    """Angular stones: Voronoi cells with tilted, bevelled tops and crevices at the cell
    borders (F2 - F1). Returns the relief over (x, z) and the stone's id hash."""
    ci, cj = np.floor(x / cell), np.floor(z / cell)
    f1 = np.full_like(x, 1e9)
    f2 = np.full_like(x, 1e9)
    top = np.zeros_like(x)
    sid = np.zeros_like(x)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = ci + di, cj + dj
            px = (i + 0.1 + 0.8 * _hash(i, j, seed)) * cell
            pz = (j + 0.1 + 0.8 * _hash(i, j, seed + 1)) * cell
            d = np.hypot(x - px, z - pz)
            hgt = cell * (0.25 + 0.5 * _hash(i, j, seed + 3))
            ta, tb_ = _hash(i, j, seed + 5) - 0.5, _hash(i, j, seed + 6) - 0.5
            t = hgt + 0.7 * (ta * (x - px) + tb_ * (z - pz))           # a tilted face per stone
            near = d < f1
            f2 = np.where(near, f1, np.minimum(f2, d))
            top = np.where(near, t, top)
            sid = np.where(near, _hash(i, j, seed + 4), sid)
            f1 = np.minimum(f1, d)
    bevel = np.clip((f2 - f1) / (0.3 * cell), 0.0, 1.0) ** 0.4        # steep broken sides, faceted tops
    return np.maximum(top, 0.05 * cell) * bevel, sid


def _ang_window(phi, centre, half_deg, soft_deg=8.0):
    d = np.abs((phi - centre + np.pi) % (2 * np.pi) - np.pi)
    return 1.0 - _smooth(math.radians(half_deg - soft_deg), math.radians(half_deg + soft_deg), d)


def seabed_parts(x, z, ripples=True, chips=True):
    """(height, rock relief, stone id) on a grid or a scalar pair. ripples=False / chips=False
    band-limit the bed for a mesh too coarse to carry them (they would alias into moire)."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    r, phi = np.hypot(x, z), np.arctan2(z, x)
    # sand: current ripples (0.6 m, flattened near the berm) on a gentle swell of the bed
    rip = 0.018 * np.sin(2 * np.pi * (0.94 * x + 0.34 * z) / 0.6 + 1.4 * np.sin(0.23 * x - 0.17 * z)
                         + 0.8 * np.sin(0.051 * z + 0.3))
    rip = rip * (0.5 + 0.5 * _smooth(R_ARMOUR + 1.0, R_ARMOUR + 8.0, r)) * float(ripples)
    sand = -DEPTH + rip + 0.25 * np.sin(0.031 * x + 0.4) * np.cos(0.027 * z)
    bould = np.zeros_like(x)
    for (bx, bz, br, bh) in BOULDERS:
        q = 1.0 - ((x - bx) ** 2 + (z - bz) ** 2) / (br * br)
        bould = np.maximum(bould, bh * np.sqrt(np.clip(q, 0.0, 1.0)) ** 0.7
                           * (1.0 + 0.12 * np.sin(5.0 * x + 3.0 * z + bx)))
    # the armour berm: flat top out to the toe slope
    env = _smooth(R_ARMOUR, R_ARMOUR - TOE_W, r)
    dmg = _ang_window(phi, DAMAGE_AZ, DAMAGE_HALF_DEG) * _smooth(8.0, 11.0, r) * _smooth(R_ARMOUR + 1.0, R_ARMOUR - 3.0, r)
    env_core = env * (1.0 - 0.85 * dmg)
    big, sid = _stones(x, z, STONE, 11.0)
    grav, gid = _stones(x, z, 0.22, 23.0)
    # displaced stones: dragged out beyond the toe of the damaged sector
    spill = _ang_window(phi, DAMAGE_AZ + math.radians(4.0), DAMAGE_HALF_DEG + 4.0) * \
        np.exp(-((r - (R_ARMOUR + 3.0)) / 2.2) ** 2)
    keep = np.clip(env_core + spill * (big > 0.25 * STONE), 0.0, 1.0)
    rock = keep * big + (1.0 - keep) * dmg * grav                     # filter gravel where the armour went
    if chips:                                                         # fracture facets on the stones (3 cm)
        rock = rock + 0.22 * _stones(x, z, 0.13, 31.0)[0] * _smooth(0.02, 0.08, rock)
    stone_id = np.where(keep * big >= (1.0 - keep) * dmg * grav, sid, gid)
    armour = env_core * (T_ARMOUR - 0.22) + dmg * 0.12
    # edge scour pit on the down-current side; the berm toe slumps into it
    pit = PIT_DEPTH * _ang_window(phi, CURRENT_AZ, PIT_HALF_DEG, 14.0) * np.exp(-((r - PIT_R) / PIT_W) ** 2)
    rock = rock + _ang_window(phi, CURRENT_AZ, PIT_HALF_DEG, 14.0) * np.exp(-((r - (R_ARMOUR + 1.2)) / 1.4) ** 2) * big * 0.8
    bould = bould * (1.0 - env)
    h = sand + armour + rock + bould - pit
    return h, rock + bould, np.where(bould > rock, 0.35, stone_id)


def seabed_height(x, z):
    """Seabed y (m) at world (x, z): the site's ground truth."""
    return seabed_parts(x, z)[0]



# --------------------------------------------------------------------------- #
#  Sky and the underwater image-based light
# --------------------------------------------------------------------------- #
SUN_DIR = np.array([0.29, 0.454, 0.843])      # afternoon sun, 27 deg up, from azimuth 71 deg
SUN_DIR /= np.linalg.norm(SUN_DIR)
HAZE = (0.80, 0.83, 0.86)                     # horizon haze: the sky's horizon and the air fog agree
FOG_DENSITY = 0.0005                          # a hazy North Sea day: ~84 % left at 350 m, ~5 % at the 6 km sheet edge


def _vnoise(u, v, seed):
    """Smooth value noise on the plane (unit cells)."""
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = _hash(iu, iv, seed), _hash(iu + 1, iv, seed)
    c, d = _hash(iu, iv + 1, seed), _hash(iu + 1, iv + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def _fbm(u, v, seed, octaves=6):
    out, amp, tot = np.zeros_like(u), 1.0, 0.0
    for k in range(octaves):
        out += amp * _vnoise(u, v, seed + 13.0 * k)
        tot += amp
        u, v = 2.03 * u + 17.1, 2.03 * v - 9.7
        amp *= 0.5
    return out / tot


def sky_env(sun_dir, w=2048, h=1024):
    """A float equirect: broken cumulus over a hazy North Sea (clouds projected on a 1.5 km deck,
    lit tops toward the sun, grey bases, blue gaps), a sun in a gap. The SAME texture is the
    background and the environment, so the water mirrors the clouds the camera sees."""
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.empty((h, w, 3), np.float32)
    d[..., 0] = np.cos(elev)[:, None] * np.cos(az)[None, :]
    d[..., 1] = np.sin(elev)[:, None]
    d[..., 2] = np.cos(elev)[:, None] * np.sin(az)[None, :]
    y = d[..., 1]
    haze = np.float32(HAZE)
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.5
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.4
    blue = haze * (1.0 - up) + np.float32([0.20, 0.34, 0.60]) * up
    col = np.where(y[..., None] >= 0.0, blue, haze * (1.0 - down) + np.float32([0.05, 0.07, 0.07]) * down)
    ang = np.arccos(np.clip(d @ np.asarray(sun_dir, np.float32), -1.0, 1.0))
    # the cloud deck
    yy = np.maximum(y, 0.03)
    pu, pv = d[..., 0] / yy * 0.9, d[..., 2] / yy * 0.9            # 1.5 km deck, ~1.7 km cells
    n = _fbm(pu, pv, 3.0)
    sh = np.float32(sun_dir)[[0, 2]] / np.linalg.norm(np.float32(sun_dir)[[0, 2]])
    n_sun = _fbm(pu + 0.08 * sh[0], pv + 0.08 * sh[1], 3.0)
    cover = _smooth(0.50, 0.66, n) * _smooth(0.0, 0.10, y) * (1.0 - np.exp(-(ang / math.radians(7.0)) ** 2))
    lit = np.clip(0.85 + 4.0 * (n - n_sun), 0.45, 1.35)            # the sun-facing flanks are brighter
    thick = _smooth(0.62, 0.85, n)
    cloud = (np.float32([1.20, 1.18, 1.12]) * (1.0 - thick[..., None]) + np.float32([0.42, 0.45, 0.50]) * thick[..., None])
    cloud = cloud * lit[..., None] * (1.0 + 1.5 * np.exp(-(ang / math.radians(18.0)) ** 2))[..., None]
    col = col * (1.0 - cover[..., None]) + cloud * cover[..., None]
    gap = (1.0 - cover)[..., None]
    col += gap * ((np.exp(-(ang / math.radians(1.6)) ** 2) * 40.0
                   + np.exp(-(ang / math.radians(12.0)) ** 2) * 1.2)[..., None] * np.float32([1.0, 0.93, 0.82]))
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


def uw_env_tex(w=256, h=128):
    """Image-based light under 30 m of water: dim green-grey from above, near black below."""
    el = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    t = (0.5 + 0.5 * np.sin(el))[:, None, None] ** 2.0
    col = np.float32([0.004, 0.007, 0.006]) * (1 - t) + np.float32([0.05, 0.085, 0.07]) * t
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = np.broadcast_to(col, (h, w, 3))
    return tp.float_texture(out)



# --------------------------------------------------------------------------- #
#  Marine growth colours, blade points
# --------------------------------------------------------------------------- #
_TP = tb.T(0.0, tb.val(spec["tower"]["top_y"]), 0.0) @ tb.rotor_node(spec)


def _sines(s_, y, seed, lams):
    """Smooth pseudo-noise in about [-1, 1] on the pile surface (arc length s_, height y)."""
    r = np.random.default_rng(seed)
    out = np.zeros_like(y)
    for lam in lams:
        th, ph = r.uniform(0, np.pi), r.uniform(0, 2 * np.pi)
        out += np.sin(2 * np.pi * (np.cos(th) * s_ + np.sin(th) * y) / lam + ph)
    return out / math.sqrt(len(lams))


def growth_colours(P):
    """A plain graded band: dark olive-brown and rough just under the splash zone, thinning with
    depth to grey-green over the coating; only low-frequency smooth variation."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    s_ = PILE_R * np.arctan2(z, x)
    n = _sines(s_, y, 1, (4.0, 6.5, 9.0))
    t = _smooth(-3.5, -24.0, y + 1.2 * n)
    top = np.array([0.055, 0.047, 0.028])
    deep = np.array([0.105, 0.12, 0.10])
    col = top[None, :] * (1.0 - t)[:, None] + deep[None, :] * t[:, None]
    col = col * (1.0 + 0.12 * n)[:, None]
    slime = np.array([0.09, 0.11, 0.045])
    w = _smooth(-2.9, -2.2, y)[:, None]
    return (col * (1.0 - w) + slime[None, :] * w).astype(np.float32)


def _colour_growth(o):
    if isinstance(o, tp.Mesh):
        o.geometry.set_attribute("color", growth_colours(o.geometry.get_attribute("position")))
        o.material.vertex_colors = True
        o.material.color = tp.Color(1.0, 1.0, 1.0)



PAINT = {"lep": ((0.50, 0.51, 0.51), 0.16, 0.0), "bond": ((0.50, 0.51, 0.50), 0.4, 0.0),
         "receptor": ((0.55, 0.56, 0.57), 0.35, 0.9), "id": ((0.04, 0.045, 0.05), 0.5, 0.0), "erosion": ((1.0, 1.0, 1.0), 0.95, 0.0)}


def upright_span_marks(blade):
    """The span numbers, upright for an observer with the tip down: text right runs round the
    section, text down runs toward the tip; unmirrored from outside. Just tipward of each tick."""
    from PIL import Image, ImageDraw, ImageFont
    bd = spec["blade_detail"]
    sm = bd["span_marks"]
    Lb = tb.val(spec["blade"]["length"])
    grid = tb.blade_rings(spec)
    font = ImageFont.truetype("arialbd.ttf", 48)
    out = []
    for s0 in range(sm["from_m"], int(Lb) - 5, sm["every_m"]):
        txt = f"{s0}"
        x0, y0, x1, y1 = font.getbbox(txt)
        im = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8), 0)
        ImageDraw.Draw(im).text((4 - x0, 4 - y0), txt, fill=255, font=font)
        ink = np.asarray(im) > 110                          # rows = text down, cols = text right
        px = sm["height_m"] / ink.shape[0]
        span0, arc0 = s0 + 0.12, sm["arc_m"]
        c, n = tb.blade_surface(spec, [span0, span0 + 0.1], [arc0, arc0 + 0.1], grid)
        sgn = float(np.sign(np.dot(np.cross(c[1, 0] - c[0, 0], c[0, 1] - c[0, 0]), n[0, 0])))
        sp = span0 + np.arange(ink.shape[0] + 1) * px
        ar = arc0 + (np.arange(ink.shape[1] + 1) - 0.5 * ink.shape[1]) * px * sgn
        P, N = tb.blade_surface(spec, sp, ar, grid)
        out.append((f"span_{s0}", P + N * 0.003, ink, None, "id"))
    return out


def add_blade_paint(scene):
    """Blade paint (spec blade_detail): LE protection strip, bond line, receptors, blade ID, and the
    erosion patch on blade_1; sheets from build_turbine_blender.blade_paint, a few mm proud."""
    mats = {}
    for k, (c, rough, metal) in PAINT.items():
        m = tp.MeshStandardMaterial()
        m.color = tp.Color(*c)
        m.roughness = rough
        m.metalness = metal
        m.vertex_colors = k == "erosion"
        mats[k] = m
    for b in (1, 2, 3):
        M = _TP @ tb.pitch_node(spec, b)
        sheets = [s for s in tb.blade_paint(spec, b) if not s[0].startswith("span_")] + upright_span_marks(b)
        for name, P, keep, col, mk in sheets:
            m_, n_ = P.shape[:2]
            W = P.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            a = (np.arange(m_ - 1)[:, None] * n_ + np.arange(n_ - 1)[None, :])
            if keep is not None:
                a = a[keep]
            a = a.reshape(-1)
            idx = np.stack([a, a + n_, a + 1, a + 1, a + n_, a + n_ + 1], 1)
            e1, e2 = W[idx[:, 1]] - W[idx[:, 0]], W[idx[:, 2]] - W[idx[:, 0]]
            yl = P.reshape(-1, 3)[idx[:, 0], 1]                        # the section centre (prebent axis) under each
            s = (yl + 1.0) / (tb.val(spec["blade"]["length"]) + 1.0)
            xo = tb.val(spec["blade"]["tip_prebend"]) * np.maximum((s - 0.2) / 0.8, 0.0) ** 2
            c = W[idx[:, 0]] - (np.stack([xo, yl, np.zeros_like(yl)], 1) @ M[:3, :3].T + M[:3, 3])
            if np.sum(np.cross(e1, e2) * c) < 0.0:                  # wind the sheet to face out
                idx = idx[:, [0, 2, 1, 3, 5, 4]]
            geo = tp.BufferGeometry()
            geo.set_attribute("position", W.astype(np.float32))
            if col is not None:
                geo.set_attribute("color", col.reshape(-1, 3).astype(np.float32))
            geo.set_index(idx.reshape(-1).astype(np.uint32))
            geo.compute_vertex_normals()
            mesh = tp.Mesh(geo, mats[mk])
            mesh.name = f"blade_{b}_{name}"
            mesh.receive_shadow = True
            scene.add(mesh)


def blade_point(i, span):
    """World point on blade i's pitch axis at `span` metres from the root (prebend included)."""
    L = tb.val(spec["blade"]["length"])
    s = span / L
    xoff = tb.val(spec["blade"]["tip_prebend"]) * max((s - 0.2) / 0.8, 0.0) ** 2
    M = _TP @ tb.pitch_node(spec, i)
    return (M @ np.array([xoff, -1.0 + s * (L + 1.0), 0.0, 1.0]))[:3]



# --------------------------------------------------------------------------- #
#  Seabed mesh (vertex coloured: sand, armour, gravel)
# --------------------------------------------------------------------------- #
FINE_HALF, FINE_STEP = 42.0, 0.09          # stones and ripples resolved out past the scour pit
CLOSE_BOX, CLOSE_STEP = (-10.5, -0.5, -10.5, -0.5), 0.03   # around the cable entry (az -135, r 5.5)


def grid_mesh(g, colour_fn, hole=None, drop=0.0, ripples=True, gz=None, hole_box=None, lift=0.0):
    """Seabed patch over the grid lines g (both axes) from seabed_parts; `hole` skips cells
    wholly inside that half-size (the fine patch covers them), `drop` lowers this patch there."""
    gz = g if gz is None else gz
    n = len(g)
    X, Z = np.meshgrid(g, gz)
    Y, rock, sid = seabed_parts(X, Z, ripples, chips=ripples)
    Y = Y + lift
    col = colour_fn(X, Z, Y, rock, sid).reshape(-1, 3).astype(np.float32)
    if drop:
        Y = Y - drop * (1.0 - _smooth(FINE_HALF + 2.0, FINE_HALF + 12.0, np.maximum(np.abs(X), np.abs(Z))))
    pos = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    a = (np.arange(len(gz) - 1)[:, None] * n + np.arange(n - 1)[None, :]).reshape(-1)
    if hole is not None:
        inside = (np.abs(g[:-1]) < hole) & (np.abs(g[1:]) < hole)
        a = a[(inside[:, None] & inside[None, :]).reshape(-1) == False]      # noqa: E712
    if hole_box is not None:                  # cells wholly inside a refinement patch
        x0, x1, z0, z1 = hole_box
        ix = (g[:-1] > x0) & (g[1:] < x1)
        iz = (gz[:-1] > z0) & (gz[1:] < z1)
        a = a[(iz[:, None] & ix[None, :]).reshape(-1) == False]           # noqa: E712
    idx = np.stack([a, a + n, a + 1, a + 1, a + n, a + n + 1], 1).reshape(-1).astype(np.uint32)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", pos)
    geo.set_attribute("color", col)
    geo.set_index(idx)
    geo.compute_vertex_normals()
    return geo


def seabed_colour(X, Z, Y, rock, sid):
    sand = np.array([0.30, 0.26, 0.19])
    stone = np.array([0.10, 0.10, 0.098])
    rusty = np.array([0.20, 0.15, 0.10])
    fouled = np.array([0.12, 0.115, 0.07])
    shell = np.array([0.46, 0.43, 0.37])
    t = _smooth(0.02, 0.08, rock)[..., None]
    v = (0.6 + 0.7 * sid)[..., None] * (0.85 + 0.3 * _hash(np.floor(X * 20.0), np.floor(Z * 20.0), 4.0))[..., None]
    kind = _hash(sid, 7.0, 1.0)[..., None]
    base = stone * (1.0 - _smooth(0.75, 0.9, kind)) + rusty * _smooth(0.75, 0.9, kind)
    top = _smooth(0.5, 0.9, rock / (0.6 * STONE))[..., None] * (0.3 + 0.7 * _hash(sid, 1.0, 3.0))[..., None]
    rc = (base * (1.0 - top) + fouled * top) * v
    # shell and debris patches on the sand: low-frequency patches of broken shell specks
    patch = _smooth(0.55, 0.85, 0.5 + 0.25 * (np.sin(0.21 * X + 1.3) * np.cos(0.17 * Z - 0.4)
                                              + np.sin(0.47 * X - 0.33 * Z + 2.0)))
    fleck = (_hash(np.floor(X * 11.0), np.floor(Z * 11.0), 9.0) > 0.72) * patch
    speck = 0.88 + 0.24 * _hash(np.floor(X * 9.0), np.floor(Z * 9.0), 5.0)
    sandc = sand * speck[..., None] * (1.0 - 0.25 * patch)[..., None]
    sandc = sandc * (1.0 - fleck[..., None]) + shell * fleck[..., None]
    return sandc * (1.0 - t) + rc * t



# --------------------------------------------------------------------------- #
#  The export cable in its CPS: out of the entry hole, hanging to the rock,
#  over the berm, buried beyond the toe
# --------------------------------------------------------------------------- #
CPS_R = 0.17                      # a 0.34 m sleeve, collars to 0.40 m


def cable_path():
    ce = spec["empties"]["cable_entry"]
    rn = np.array([math.cos(CABLE_AZ), 0.0, math.sin(CABLE_AZ)])
    tn = np.array([-math.sin(CABLE_AZ), 0.0, math.cos(CABLE_AZ)])
    r0, y0 = PILE_R + 0.5, ce["y"]
    rs = np.arange(r0, R_ARMOUR + 9.0, 0.05)
    lat = 0.6 * np.sin((rs - r0) * 0.18) * _smooth(r0 + 3.0, r0 + 8.0, rs)          # a slight lay meander
    xs, zs = rs * rn[0] + lat * tn[0], rs * rn[2] + lat * tn[2]
    # rests on stone crests: the highest bed under the tube's footprint, then stiffened
    top = np.full_like(rs, -1e9)
    for dl in (-0.25, 0.0, 0.25):
        for dr in (-0.3, 0.0, 0.3):
            top = np.maximum(top, seabed_height(xs + dr * rn[0] + dl * tn[0], zs + dr * rn[2] + dl * tn[2]))
    rest = top + CPS_R
    k = np.ones(21) / 21
    for _ in range(3):
        rest = np.maximum(np.convolve(np.pad(rest, 10, mode="edge"), k, mode="valid"), top + CPS_R * 0.8)
    # catenary-ish hang from the hole to touchdown
    r_td = r0 + 4.0
    i_td = int(np.searchsorted(rs, r_td))
    s = np.clip((rs - r0) / (r_td - r0), 0.0, 1.0)
    y_hang = y0 + (rest[i_td] - y0) * (3 * s ** 2 - 2 * s ** 3)
    y = np.where(rs < r_td, np.maximum(y_hang, rest), rest)
    # burial beyond the toe
    bur = 1.1 * _smooth(R_ARMOUR + 3.0, R_ARMOUR + 7.0, rs)
    y = y - bur
    return np.stack([xs, y, zs], 1), rs - r0


def tube(path, radius_fn, sides=18):
    P = np.asarray(path, float)
    Tg = np.gradient(P, axis=0)
    Tg /= np.linalg.norm(Tg, axis=1, keepdims=True)
    nrm = np.cross(Tg[0], [0.0, 1.0, 0.0])
    nrm /= np.linalg.norm(nrm)
    Ns = []
    for t in Tg:                                        # parallel transport
        nrm = nrm - t * np.dot(nrm, t)
        nrm /= np.linalg.norm(nrm)
        Ns.append(nrm.copy())
    Ns = np.array(Ns)
    Bs = np.cross(Tg, Ns)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    rr = radius_fn(s)
    a = 2 * np.pi * np.arange(sides + 1) / sides
    pos = (P[:, None, :] + rr[:, None, None] * (np.cos(a)[None, :, None] * Ns[:, None, :]
                                                + np.sin(a)[None, :, None] * Bs[:, None, :])).reshape(-1, 3)
    n = len(P)
    i = np.arange(n - 1)[:, None] * (sides + 1) + np.arange(sides)[None, :]
    i = i.reshape(-1)
    idx = np.stack([i, i + 1, i + sides + 1, i + 1, i + sides + 2, i + sides + 1], 1).reshape(-1)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", pos.astype(np.float32))
    geo.set_index(idx.astype(np.uint32))
    geo.compute_vertex_normals()
    return geo


def cps_radius(s):
    rib = _smooth(0.0, 0.08, 0.3 - np.abs((s % 0.6) - 0.3))           # a collar every 0.6 m
    stiff = np.clip(1.0 - s / 3.0, 0.0, 1.0) ** 1.5 * 0.38            # bend stiffener off the entry
    return CPS_R + 0.035 * rib + stiff



# --------------------------------------------------------------------------- #
#  Cameras (names are the contract)
# --------------------------------------------------------------------------- #
def pol(r, az_deg, y):
    a = math.radians(az_deg)
    return np.array([r * math.cos(a), y, r * math.sin(a)])


def cam_blade():
    root, mid, low = blade_point(1, 10.0), blade_point(1, 70.0), blade_point(1, 95.0)
    side = np.array([0.9, 0.0, -0.44])                 # off the leading edge, sun from camera left
    side /= np.linalg.norm(side)
    eye = blade_point(1, 100.0) + 12.0 * side + np.array([0.0, -3.0, 0.0])
    return eye, blade_point(1, 25.0), 50.0


CAMS = {
    #  name: (eye, target, vfov, underwater, exposure)
    "site_far": (lambda: (np.array([300.0, 20.0, 190.0]), np.array([0.0, 88.0, 0.0]), 45.0), False, 0.8),
    "tp_waterline": (lambda: (pol(30.0, 62.0, 2.6), np.array([0.0, 5.5, 3.5]), 48.0), False, 0.8),
    "nacelle": (lambda: (np.array([42.0, 171.0, 84.0]), np.array([3.0, 150.0, 0.0]), 34.0), False, 0.65),
    "blade": (cam_blade, False, 0.6),
    "under_pile": (lambda: (pol(9.5, -115.0, -17.5), pol(4.0, -125.0, -26.0), 70.0), True, 1.2),
    "seabed_cps": (lambda: (pol(10.5, -152.0, -27.4), pol(6.5, -134.0, -28.3), 72.0), True, 1.4),
    "pile_approach": (lambda: (pol(11.0, -110.0, -15.0), pol(5.9, -110.0, -13.3), 60.0), True, 1.4),
    "site_aerial": (lambda: (pol(250.0, 215.0, 120.0), np.array([0.0, 12.0, 0.0]), 40.0), False, 0.8),
    # "from above": 200 m up, 60 deg down (straight down the sea mirrors one patch of sky at 2 % Fresnel)
    "site_top": (lambda: (pol(115.0, 150.0, 200.0), np.array([0.0, 0.0, 0.0]), 50.0), False, 0.8),
    "sky_only": (lambda: (np.array([-150.0, 3.0, -150.0]),
                          np.array([-150.0, 3.0, -150.0]) + 100.0 * np.array([math.cos(math.radians(30)) * math.cos(math.radians(110)),
                                                                             0.5, math.cos(math.radians(30)) * math.sin(math.radians(110))]),
                          60.0), False, 0.8),
    "seabed_wide": (lambda: (pol(36.0, -122.0, -27.0), np.array([0.0, -25.5, 0.0]), 72.0), True, 1.6),
}


# --------------------------------------------------------------------------- #
#  The site
# --------------------------------------------------------------------------- #
OCEAN_RES = 1024
LAMP_I = 45.0                      # phase-1 work lamps (the robots bring theirs later)
FILL_I = 0.4
MOTE_CAP, MOTE_HALF, MOTE_TOP, MOTE_BOTTOM = 24_000, 8.0, -0.6, -DEPTH
MOTE_BASE = 0.055


def bed_mat_back():
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(0.28, 0.24, 0.18)
    m.roughness = 1.0
    return m


def build_site(renderer, murk=0.20, murk_colour=(0.14, 0.185, 0.155), foam=0.45, sheet=12000.0):
    """Build the whole site into a new scene; nothing is rendered here. Returns a namespace:
    scene, ocean, turbine, seabed, seabed_far, far, cps, cable_path, motes, lamps, fill, sky, uw_env, sun,
    seabed_height / seabed_parts (the ground truth), cameras (CAMS: name -> (fn, underwater, exposure)),
    blade_point, pol, spec, and the helpers
      set_lamps(eye, target, on)       the two underwater work lamps beside a camera
      apply_camera(camera, name)       framing, near/far, exposure, medium, lamps; returns underwater
      update(t, camera, under, dt)     per frame before render: sim_time, over/under split, snow
      settle_snow(camera, t)           steps the marine snow to a filled slab for a still."""
    import types
    if not tp.HAS_VULKAN:
        raise RuntimeError("the FFT ocean needs a Vulkan build of threepp (-DTHREEPP_WITH_VULKAN=ON)")
    if not os.path.exists(GLB):
        raise FileNotFoundError("turbine.glb is generated. Build it once, from this folder: "
                                "blender --background --factory-startup "
                                "--python build_turbine_blender.py -- --spec turbine_spec.json --out turbine.glb")
    S = types.SimpleNamespace(seabed_height=seabed_height, seabed_parts=seabed_parts, cameras=CAMS, spec=spec,
                              blade_point=blade_point, pol=pol)
    scene = tp.Scene()
    sky = sky_env(SUN_DIR)
    scene.background = sky
    scene.environment = sky
    uw_env = uw_env_tex()
    sun = tp.DirectionalLight(0xfff0dc, 2.4)
    sun.position.set(*(SUN_DIR * 600.0))
    sun.cast_shadow = True
    scene.add(sun)

    ocean_size = sheet                          # one sheet, packed toward the pile; the haze closes before its edge
    ocean = tp.Ocean(size=ocean_size, resolution=OCEAN_RES, wind_speed=7.0, wind_theta=math.pi + 0.25, choppiness=0.7,
                     fft_size=512, fetch=60000.0)
    ocean.params.tile_size_0 = 320.0
    ocean.params.tile_size_1 = 40.6
    ocean.params.tile_size_2 = 2.98
    ocean.warp.center_x = 0.0
    ocean.warp.center_z = 0.0
    ocean.warp.half_range = 0.5 * ocean_size
    # cubic warp x = h (a t + (1 - a) t^3): spacing h (a + 3 (1 - a) t^2) 2 / (N - 1); a gives 0.8 m at the pile.
    # The vertices carry cascades 0 + 1 (and their normals), so this spacing is where waves exist.
    h = 0.5 * ocean_size
    a = min(1.0, 0.8 * (OCEAN_RES - 1) / (2.0 * h))
    ocean.warp.coef_a = a
    tt = np.linspace(0.0, 1.0, 200001)
    xs = h * (a * tt + (1.0 - a) * tt ** 3)
    sp = h * (a + 3.0 * (1.0 - a) * tt ** 2) * 2.0 / (OCEAN_RES - 1)
    print("[site] sea mesh spacing (m from pile: m): "
          + ", ".join("%.0f: %.2f" % (d, np.interp(d, xs, sp)) for d in (0, 50, 100, 300, 1000, h)))
    fog = FOG_DENSITY                           # haze: the sheet edge (0.5 x sheet away) is gone in it
    scene.set_fog_exp2(tp.Color(*HAZE), fog)
    ocean.material.attenuation_color = tp.Color(0.11, 0.17, 0.15)
    ocean.material.specular_intensity = 0.7
    ocean.params.foam_amount = foam             # before the first frame: the default (size/300) seeds a foam buffer
    scene.add(ocean)
    # The murk is also the sea's body seen FROM ABOVE: the water shader refracts, probes the
    # bed to 6 / sigma and returns it through this murk, so sigma decides whether the bed
    # shows from the air and the murk colour is the colour of the sea's body.
    renderer.set_underwater_murk(murk, tp.Color(*murk_colour))

    turbine = tp.GLTFLoader().load(GLB).scene
    turbine.traverse(lambda o: (setattr(o, "cast_shadow", True), setattr(o, "receive_shadow", True))
                     if isinstance(o, tp.Mesh) else None)
    scene.add(turbine)
    turbine.get_object_by_name("marine_growth").traverse(_colour_growth)
    if "blade_detail" in spec and os.environ.get("TURBINE_NO_PAINT") != "1":
        add_blade_paint(scene)

    bed_mat = tp.MeshStandardMaterial()
    bed_mat.vertex_colors = True
    bed_mat.roughness = 1.0
    bed_mat.metalness = 0.0
    t0 = time.perf_counter()
    seabed = tp.Mesh(grid_mesh(np.linspace(-FINE_HALF, FINE_HALF, int(round(2 * FINE_HALF / FINE_STEP)) + 1),
                               seabed_colour, hole_box=CLOSE_BOX), bed_mat)
    # the cable-entry close-up: the SAME seabed_height sampled at 3 cm (the fine grid's 9 cm
    # rounds the fracture facets away), a hair lifted where it overlaps the grid's border cells
    x0, x1, z0, z1 = CLOSE_BOX
    close = tp.Mesh(grid_mesh(np.arange(x0 - 0.1, x1 + 0.1, CLOSE_STEP), seabed_colour,
                              gz=np.arange(z0 - 0.1, z1 + 0.1, CLOSE_STEP), lift=0.004), bed_mat)
    close.receive_shadow = True
    scene.add(close)
    seabed.receive_shadow = True
    scene.add(seabed)
    # the far field: the same function on a graded grid out to 1.5 km, band-limited (no
    # ripples), a hair lower where it underlaps the fine patch
    u = np.linspace(-1.0, 1.0, 481)
    gc = 1500.0 * (0.06 * u + 0.94 * u ** 3)
    seabed_far = tp.Mesh(grid_mesh(gc, seabed_colour, hole=FINE_HALF - 1.0, drop=0.06, ripples=False), bed_mat)
    seabed_far.receive_shadow = True
    scene.add(seabed_far)
    far_mat = tp.MeshStandardMaterial()
    far_mat.color = tp.Color(0.30, 0.33, 0.35)
    far_mat.roughness = 0.6
    far_mat.side = tp.Side.Double               # seen from below too: rays past the sheet edge must end in murk
    far = tp.Mesh(tp.RingGeometry(0.49 * ocean_size, 40000.0, 256), far_mat)   # beyond the sheet, in the haze
    far.rotate_x(-math.pi / 2)
    far.position.y = -0.05
    scene.add(far)
    # under water every ray must end on something: beyond the 1.5 km bed an eye-level sliver
    # between the bed's horizon and the surface's saw the background through the murk
    back = tp.Mesh(tp.PlaneGeometry(40000.0, 40000.0), bed_mat_back())
    back.rotate_x(-math.pi / 2)
    back.position.y = -DEPTH - 3.0             # below the scour pit, so it never cuts the bed
    scene.add(back)

    cable, cable_s = cable_path()
    cps_mat = tp.MeshStandardMaterial()
    cps_mat.color = tp.Color(0.22, 0.13, 0.04)                       # fouled orange polyurethane
    cps_mat.roughness = 0.85
    cps = tp.Mesh(tube(cable, cps_radius), cps_mat)
    cps.cast_shadow = True
    scene.add(cps)
    print(f"[site] seabed + CPS built in {time.perf_counter() - t0:.1f} s; CPS {cable_s[-1]:.1f} m, "
          f"bed at pile {float(seabed_height(PILE_R + 0.3, 0.0)):.2f}, pit bottom "
          f"{float(seabed_height(PIT_R * math.cos(CURRENT_AZ), PIT_R * math.sin(CURRENT_AZ))):.2f}")

    lamps = []
    for _ in range(2):
        sp = tp.SpotLight(tp.Color(1.0, 0.95, 0.86), 0.0, 60.0, math.radians(40.0), 0.85, 2.0)
        tg = tp.Group()
        scene.add(tg)
        sp.set_target(tg)
        scene.add(sp)
        lamps.append((sp, tg))

    # marine snow: a follow-the-camera slab of sinking, drifting billboards (the net-pen recipe)
    uc = tp.ParticleField.Config()
    uc.capacity = MOTE_CAP
    uc.ownership = tp.ParticleField.Ownership.Renderer
    uc.w_semantic = tp.ParticleField.WSemantic.Radius
    uc.uniform_radius = 0.008
    motes = tp.ParticleField.create(uc)
    motes.frustum_culled = False
    motes.set_billboard_repr(tp.Color(0.80, 0.88, 0.82), tp.Color(0.55, 0.68, 0.62), MOTE_BASE, 1.0)
    mb = motes.billboard_repr
    mb.lod_near = 0.0
    mb.lod_fade = 0.0
    mb.stretch_seconds = 0.0
    mb.softness = 0.85
    mb.fade_power = 0.0
    mb.size_taper = 0.0
    mb.near_fade = 1.2                          # no out-of-focus blob on the lens
    mb.bright_jitter = 0.75
    mb.glow = 0.0
    me = motes.emitter
    me.spawn_center = tp.Vector3(0.0, 0.5 * (MOTE_TOP + MOTE_BOTTOM), 0.0)
    me.spawn_half_extent = tp.Vector3(MOTE_HALF, 0.5 * (MOTE_TOP - MOTE_BOTTOM), MOTE_HALF)
    me.velocity = tp.Vector3(0.0, -0.02, 0.0)            # snow sinks
    me.speed_spread = 0.012
    me.wind = tp.Vector3(-0.08, 0.0, 0.015)              # with the tidal current (toward -X)
    me.drift_amplitude = 0.10
    me.drift_frequency = 0.13
    me.drift_scale = 3.5
    me.lifetime = 40.0
    me.duty_cycle = 1.0
    me.size = 0.008
    me.size_jitter = 0.70
    me.seed = 20260930
    me.follow = True
    me.follow_snap = 0.0
    motes.set_emitter(me)
    motes.set_emitter_time(0.0, 1.0 / 60.0)
    motes.set_live_count(MOTE_CAP)
    scene.add(motes)
    fill = tp.HemisphereLight(tp.Color(0.35, 0.50, 0.45), tp.Color(0.05, 0.06, 0.05), 0.0)   # underwater only
    scene.add(fill)
    for k in range(28):                         # a thin ring of wash where the swell meets the TP
        a = 2 * math.pi * k / 28
        ocean.add_foam_disturbance(5.75 * math.cos(a), 5.75 * math.sin(a), 0.55, 0.12)
    mote_t = [0.0]

    def set_lamps(eye, tgt, on):
        eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
        for k, (sp, tg) in enumerate(lamps):
            if on:
                d = tgt - eye
                d /= np.linalg.norm(d)
                side = np.cross(d, [0.0, 1.0, 0.0])
                side /= np.linalg.norm(side)
                p = eye + side * (1.6 if k else -1.6) + np.array([0.0, 1.2, 0.0])
                sp.position.set(*map(float, p))
                tg.position.set(*map(float, tgt + side * (2.0 if k else -2.0)))
                sp.intensity = LAMP_I
            else:
                sp.intensity = 0.0

    def set_medium(under, eye):
        ocean.params.foam_amount = 0.0 if under else foam
        fill.intensity = FILL_I if under else 0.0
        motes.visible = under
        scene.environment = uw_env if under else sky     # submerged metal must not mirror the sky
        motes.set_follow_center(tp.Vector3(float(eye[0]), float(eye[1]), float(eye[2])))
        motes.billboard_repr.intensity = MOTE_BASE * (1.0 + 0.9 * min(max(-eye[1], 0.0), 8.0) / 8.0)

    def apply_camera(camera, name):
        fn, under, expo = CAMS[name]
        under = bool(under)
        eye, tgt, fov = fn()
        camera.fov = fov
        camera.near, camera.far = (0.05, 20000.0) if under else (0.3, 20000.0)
        camera.update_projection_matrix()
        camera.position.set(*map(float, eye))
        camera.look_at(tp.Vector3(*map(float, tgt)))
        renderer.tone_mapping_exposure = expo
        set_medium(under, eye)
        set_lamps(eye, tgt, under)
        return under

    def update(t, camera, under=None, dt=1.0 / 60.0):
        """Per frame, before render: the sea clock, the over/under split, the marine snow."""
        cp = camera.position
        if under is None:
            under = cp.y < -0.5
        renderer.sim_time = t
        set_medium(under, (cp.x, cp.y, cp.z))
        if dt > 0.0:
            mote_t[0] += dt
            motes.set_emitter_time(mote_t[0], dt)
        # the per-pixel over/under split: the local wave above water, a guard plane when deep
        k = min(max((-cp.y - 0.5) / 1.0, 0.0), 1.0)
        renderer.set_fog_water_surface_y(float(ocean.sample_height(cp.x, cp.z)) * (1.0 - k) + 0.42 * k)

    def settle_snow(camera, t):
        """Run the marine snow for 30 s (60 x 0.5 s) so a still sees a filled slab."""
        for _ in range(60):
            mote_t[0] += 0.5
            motes.set_emitter_time(mote_t[0], 0.5)
            renderer.sim_time = t
            renderer.render(scene, camera)

    S.__dict__.update(scene=scene, ocean=ocean, turbine=turbine, seabed=seabed, seabed_close=close, fog=fog, seabed_far=seabed_far, far=far,
                      cps=cps, cable_path=cable, motes=motes, lamps=lamps, fill=fill, sky=sky, uw_env=uw_env,
                      sun=sun, set_lamps=set_lamps, apply_camera=apply_camera, update=update,
                      settle_snow=settle_snow)
    return S


def main():
    shot = cli_arg("--shot", "all", str)
    W, H = parse_size(cli_arg("--size", "1920x1080", str))
    t_shot = cli_arg("--time", T_SHOT, float)
    canvas = tp.Canvas("threepp - offshore turbine", width=W, height=H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 0.8
    renderer.sun_angular_radius = 0.6
    renderer.bloom_intensity = 0.06
    renderer.auto_exposure = False
    S = build_site(renderer, murk=cli_arg("--murk", 0.20, float),
                   murk_colour=[float(c) for c in cli_arg("--murk-colour", "0.14,0.185,0.155", str).split(",")],
                   foam=cli_arg("--foam", 0.45, float), sheet=cli_arg("--sheet", 12000.0, float))
    camera = tp.PerspectiveCamera(45.0, W / H, 0.1, 6000.0)

    def shoot(name, out):
        t0 = time.perf_counter()
        under = S.apply_camera(camera, name)
        S.update(t_shot, camera, under, 0.0)
        if under:                              # let the marine snow fill its slab around this camera
            S.settle_snow(camera, t_shot)
        for i in range(4):                     # a headless render needs a few frames to settle
            S.apply_camera(camera, name)
            S.update(t_shot - (3 - i) / 60.0, camera, under)
            renderer.render(S.scene, camera)
        renderer.sim_time = t_shot
        d = os.path.dirname(out) or "."
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "_tmp_" + os.path.basename(out))
        renderer.save_frame(S.scene, camera, tmp)
        os.replace(tmp, out)
        print(f"[site] {name}: wrote {out} in {time.perf_counter() - t0:.1f} s")

    renderer.sim_time = 0.0
    renderer.render(S.scene, camera)           # the ocean needs one frame before sample_height reads it
    if shot == "all":
        outdir = cli_arg("--outdir", out_dir("site"), str)
        tag = cli_arg("--tag", "v01", str)
        for name in cli_arg("--cams", ",".join(CAMS), str).split(","):
            out = os.path.join(outdir, f"{name}_{tag}.png")
            if os.path.exists(out):
                sys.exit(f"{out} exists; renders are never overwritten")
            shoot(name, out)
    else:
        if shot not in CAMS:
            sys.exit(f"unknown shot {shot!r}; one of {', '.join(CAMS)} or all")
        shoot(shot, cli_arg("--out", os.path.join(out_dir("site"), f"{shot}.png"), str))


if __name__ == "__main__":
    main()

"""The harbour's concrete: procedural quay-wall, deck and fender textures (numpy -> tp.data_texture)
and the block meshes that carry them (look pass L1 of plans/harbour-scene.md).

Every texel is placed by WORLD metres: a wall atlas runs u along the block's perimeter (metres)
and v up the wall (metres above the mean sea level), and the meshes' UVs are computed from their
own vertex positions, so the tidal bands sit at their true heights whatever the block's size:

- concrete with low-frequency tone, cast panels 4..6 m long with V-joints, a darker weathered
  cap beam along the top;
- the tidal zone (Ålesund's range ~1.5..2 m about the mean): wet dark concrete up to ~+0.8 m with
  drips above it, green-brown weed from ~+0.4 m down (densest at the bottom), a pale barnacle
  speckle band just above the weed;
- rust streaks running down from given points (bollards, ladder brackets).

Wet zones are smoother (roughness map) and the joints and barnacles carry a normal map.
"""
import math

import numpy as np
from PIL import Image

import threepp as tp


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _noise(rng, w, h, cx, cy):
    """Smooth value noise in [0, 1] on a (h, w) raster with cx x cy lattice cells (bicubic)."""
    small = rng.random((max(2, int(cy)) + 1, max(2, int(cx)) + 1)).astype(np.float32)
    return np.clip(np.asarray(Image.fromarray(small, mode="F").resize((w, h), Image.BICUBIC)), 0.0, 1.0)


def _fbm(rng, w, h, cx, cy, octaves=4):
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for k in range(octaves):
        out += amp * _noise(rng, w, h, cx * 2 ** k, cy * 2 ** k)
        tot += amp
        amp *= 0.5
    return out / tot


def _maps(alb, rough, hgt, du, dv, nstrength=1.0):
    """(albedo rgb linear, roughness, height m) -> three data textures (sRGB albedo, ORM, normal)."""
    srgb = np.where(alb <= 0.0031308, alb * 12.92, 1.055 * np.power(np.clip(alb, 0.0031308, 1.0), 1.0 / 2.4) - 0.055)
    a8 = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    orm = np.zeros(alb.shape[:2] + (3,), np.uint8)
    orm[..., 0] = 255
    orm[..., 1] = (np.clip(rough, 0.04, 1.0) * 255.0 + 0.5).astype(np.uint8)
    gy, gx = np.gradient(hgt.astype(np.float32))
    n = np.stack([-gx / du * nstrength, -gy / dv * nstrength, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n8 = (np.clip(n * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return tp.data_texture(a8, srgb=True), tp.data_texture(orm, srgb=False), tp.data_texture(n8, srgb=False)


CONCRETE = np.float32([0.37, 0.36, 0.335])       # dry harbour concrete, linear albedo
WEED_G = np.float32([0.105, 0.120, 0.040])       # green-brown weed (Fucus, sea lettuce)
WEED_B = np.float32([0.100, 0.078, 0.034])       # brown wrack
BARNACLE = np.float32([0.44, 0.42, 0.37])
RUST = np.float32([0.33, 0.15, 0.06])


def wall_textures(length, top, bottom, rust_s=(), seed=1, ppm=36.0, rows=384, cap=0.45):
    """The wall atlas: u = perimeter metres 0..length, v = height bottom..top (m above the mean
    sea level). rust_s: perimeter positions (m) a rust streak runs down from the cap."""
    rng = np.random.default_rng(seed)
    W = int(min(16384, math.ceil(length * ppm)))
    H = rows
    du, dv = length / W, (top - bottom) / H
    u = (np.arange(W, dtype=np.float32) + 0.5) * du
    y = bottom + (np.arange(H, dtype=np.float32) + 0.5) * dv
    U, Y = np.meshgrid(u, y)
    cxm = length / 7.0                                   # lattice cells: ~7 m
    cym = (top - bottom) / 7.0
    low = _fbm(rng, W, H, cxm, cym, 3)
    mid = _fbm(rng, W, H, length / 0.9, (top - bottom) / 0.9, 3)
    fine = _noise(rng, W, H, length / 0.06, (top - bottom) / 0.06)
    # cast panels 4..6 m long, each its own pour (tone)
    seams = np.cumsum(rng.uniform(4.0, 6.0, int(length / 4.0) + 2))
    seams = seams[seams < length - 0.5]
    pidx = np.searchsorted(seams, u)
    ptone = 1.0 + 0.07 * rng.standard_normal(len(seams) + 1).astype(np.float32)
    tone = (1.0 + 0.38 * (low - 0.5) + 0.16 * (mid - 0.5) + 0.06 * (fine - 0.5)) * ptone[pidx][None, :]
    # vertical drip staining from the cap (dark, stretched in y)
    drip = _noise(rng, W, H, length / 0.35, (top - bottom) / 2.5)
    stain = np.clip((drip - 0.45) * 2.0, 0.0, 1.0) * _smooth(top - 3.5, top - 0.3, Y)
    tone *= 1.0 - 0.30 * stain
    alb = CONCRETE[None, None, :] * tone[..., None]
    rough = np.full((H, W), 0.90, np.float32)
    hgt = 0.0015 * (fine - 0.5) + 0.002 * (mid - 0.5)
    # the cap beam: darker, browner, weathered; a horizontal pour joint under it
    capm = _smooth(top - cap - 0.02, top - cap + 0.02, Y)
    alb *= (1.0 - capm * (0.24 + 0.10 * (low - 0.5)))[..., None]
    alb[..., 2] *= 1.0 - 0.06 * capm
    lift = np.exp(-((Y - (top - cap)) / 0.012) ** 2)
    alb *= (1.0 - 0.35 * lift)[..., None]
    hgt -= 0.006 * lift
    # V-joints between the panels: a dark groove with a soft shadow
    ds = np.min(np.abs(u[None, :] - seams[:, None]), axis=0) if len(seams) else np.full(W, 9.0, np.float32)
    groove = np.exp(-(ds / 0.012) ** 2)
    alb *= (1.0 - 0.45 * groove - 0.10 * np.exp(-(ds / 0.05) ** 2))[None, :, None]
    hgt -= 0.010 * groove[None, :]
    # rust streaks from the cap down: narrow, streaky across, fading downward
    rs = np.zeros((H, W), np.float32)
    streak = _noise(rng, W, H, length / 0.05, (top - bottom) / 3.0)
    for s in rust_s:
        wd = rng.uniform(0.10, 0.22)
        ln = rng.uniform(1.2, 2.4)
        across = np.exp(-((u - s) / wd) ** 2)[None, :]
        down = np.clip(1.0 - (top - Y) / ln, 0.0, 1.0) ** 1.4 * (Y < top - 0.02)
        rs = np.maximum(rs, across * down * (0.45 + 0.75 * streak))
    rs = np.clip(rs, 0.0, 1.0)
    alb = alb * (1.0 - 0.75 * rs[..., None]) + RUST * (tone[..., None] * 0.9) * (0.75 * rs[..., None])
    # the tidal zone. Wet up to ~+0.8 m (ragged top, drips above), smoother
    wet_top = 0.80 + 0.12 * (_noise(rng, W, H, length / 1.5, 2) - 0.5) + 0.35 * np.clip(drip - 0.62, 0.0, 1.0)
    wet = 1.0 - _smooth(wet_top - 0.12, wet_top + 0.04, Y)
    alb = alb * (1.0 - 0.78 * wet)[..., None] + np.float32([0.006, 0.008, 0.004]) * wet[..., None]
    rough = rough * (1.0 - wet) + 0.60 * wet
    # weed: from ~+0.4 m (ragged, patchy) down, densest at the bottom
    patch = _fbm(rng, W, H, length / 1.2, (top - bottom) / 0.6, 3)
    rag = _noise(rng, W, H, length / 0.35, 2)            # ragged along the wall at ~0.3 m
    weed_top = 0.38 + 0.22 * (patch - 0.5) + 0.16 * (rag - 0.5)
    weed = (1.0 - _smooth(weed_top - 0.30, weed_top, Y)) * np.clip(0.75 + 0.9 * (patch - 0.35), 0.0, 1.0)
    weed = np.maximum(weed, 1.0 - _smooth(-0.15, 0.20, Y))
    hue = _noise(rng, W, H, length / 0.7, (top - bottom) / 0.4)
    wcol = WEED_G * hue[..., None] + WEED_B * (1.0 - hue[..., None])
    wcol = wcol * (0.75 + 0.5 * fine[..., None])
    alb = alb * (1.0 - 0.92 * weed[..., None]) + wcol * (0.92 * weed[..., None])
    rough = rough * (1.0 - weed) + 0.32 * weed
    hgt += 0.004 * weed * (hue - 0.5)
    # barnacles: a thin pale speckle band just above the weed
    band = np.exp(-((Y - (weed_top + 0.13)) / 0.07) ** 2) * (1.0 - weed)
    broken = np.clip(2.2 * (_noise(rng, W, H, length / 0.8, 2) - 0.3), 0.0, 1.0)   # a broken band
    dots = (rng.random((H, W)) < 0.24 * band * broken).astype(np.float32)
    dots = np.maximum(dots, 0.6 * np.roll(dots, 1, axis=1))
    alb = alb * (1.0 - 0.8 * dots[..., None]) + BARNACLE * (0.8 * dots[..., None])
    rough = rough * (1.0 - dots) + 0.8 * dots
    hgt += 0.006 * dots
    return _maps(alb, rough, hgt, du, dv), (W, H)


def deck_textures(lx, dz, edges=("+z",), rust_pts=(), seed=2, ppm=12.0, slab=(6.0, 4.0), edge_w=0.45):
    """A deck lx x dz m (u along local x, v along local z, v = 0 at -z): cast slabs with joints,
    tone, a darker weathered edge beam along the named edges, rust rings at rust_pts (local x, z)."""
    rng = np.random.default_rng(seed)
    W = int(min(8192, math.ceil(lx * ppm)))
    H = int(min(2048, max(16, math.ceil(dz * ppm))))
    du, dv = lx / W, dz / H
    x = (np.arange(W, dtype=np.float32) + 0.5) * du - 0.5 * lx
    z = (np.arange(H, dtype=np.float32) + 0.5) * dv - 0.5 * dz
    X, Z = np.meshgrid(x, z)
    low = _fbm(rng, W, H, lx / 9.0, dz / 9.0, 3)
    mid = _fbm(rng, W, H, lx / 1.2, dz / 1.2, 3)
    fine = _noise(rng, W, H, lx / 0.12, dz / 0.12)
    sx, sz = slab
    ix, iz = np.floor((X + 0.5 * lx) / sx), np.floor((Z + 0.5 * dz) / sz)
    ptone = 1.0 + 0.06 * np.sin(ix * 12.9898 + iz * 78.233 + seed) * 1.7
    tone = (1.0 + 0.20 * (low - 0.5) + 0.10 * (mid - 0.5) + 0.06 * (fine - 0.5)) * ptone
    alb = np.float32([0.47, 0.46, 0.44])[None, None, :] * tone[..., None]
    rough = np.full((H, W), 0.88, np.float32)
    hgt = 0.0015 * (fine - 0.5)
    jx = np.abs(((X + 0.5 * lx) / sx + 0.5) % 1.0 - 0.5) * sx
    jz = np.abs(((Z + 0.5 * dz) / sz + 0.5) % 1.0 - 0.5) * sz
    joint = np.maximum(np.exp(-(jx / 0.03) ** 2), np.exp(-(jz / 0.03) ** 2))
    alb *= (1.0 - 0.35 * joint)[..., None]
    hgt -= 0.006 * joint
    # oil / water stains: dark low-frequency blotches
    blot = np.clip((_fbm(rng, W, H, lx / 4.0, dz / 4.0, 3) - 0.58) * 4.0, 0.0, 1.0)
    alb *= (1.0 - 0.22 * blot)[..., None]
    rough = rough - 0.15 * blot
    d_edge = np.full((H, W), 1e9, np.float32)
    for e in edges:
        d = {"+z": 0.5 * dz - Z, "-z": Z + 0.5 * dz, "+x": 0.5 * lx - X, "-x": X + 0.5 * lx}[e]
        d_edge = np.minimum(d_edge, d)
    capm = 1.0 - _smooth(edge_w - 0.02, edge_w + 0.02, d_edge)
    alb *= (1.0 - capm * (0.26 + 0.12 * (low - 0.5)))[..., None]
    for (px, pz) in rust_pts:
        r = np.hypot(X - px, Z - pz)
        ring = np.exp(-((r - 0.25) / 0.12) ** 2) * (0.5 + 0.8 * fine) + 0.6 * np.exp(-(r / 0.5) ** 2) * mid
        ring = np.clip(ring, 0.0, 1.0)
        alb = alb * (1.0 - 0.6 * ring[..., None]) + RUST * tone[..., None] * (0.6 * ring[..., None])
    return _maps(alb, rough, hgt, du, dv), (W, H)


def fender_textures(seed=3, y0=-0.5, y1=2.1):
    """A cylinder fender (u around, v from its bottom y0 to its top y1, metres above the mean
    sea level): black rubber, pale scuffs and scrapes, a wet / weedy foot below ~+0.6 m."""
    rng = np.random.default_rng(seed)
    W, H = 256, 512
    v = (np.arange(H, dtype=np.float32) + 0.5) / H
    Y = (y0 + v * (y1 - y0))[:, None] * np.ones((1, W), np.float32)
    fine = _noise(rng, W, H, 40, 160)
    low = _fbm(rng, W, H, 3, 8, 3)
    alb = np.float32([0.030, 0.030, 0.031])[None, None, :] * (0.8 + 0.5 * low[..., None])
    rough = np.full((H, W), 0.78, np.float32)
    # scuffs: short horizontal scrapes, pale grey, mostly on the outboard half (u 0.25..0.75)
    sc = np.zeros((H, W), np.float32)
    for _ in range(36):
        cu, cv = rng.uniform(0.15, 0.85), rng.uniform(0.25, 0.95)
        lu, lv = rng.uniform(0.04, 0.18), rng.uniform(0.002, 0.008)
        uu = np.arange(W)[None, :] / W
        sc = np.maximum(sc, np.exp(-((uu - cu) / lu) ** 2) * np.exp(-((v[:, None] - cv) / lv) ** 2)
                        * rng.uniform(0.4, 1.0))
    sc *= 0.5 + 0.8 * fine
    alb = alb * (1.0 - 0.6 * sc[..., None]) + np.float32([0.10, 0.098, 0.095]) * (0.6 * sc[..., None])
    rough = rough + 0.12 * sc
    wet = 1.0 - _smooth(0.45, 0.75, Y)
    alb *= (1.0 - 0.3 * wet)[..., None]
    rough = rough * (1.0 - wet) + 0.4 * wet
    weed = (1.0 - _smooth(-0.05, 0.35, Y)) * np.clip(0.4 + low, 0.0, 1.0)
    alb = alb * (1.0 - 0.8 * weed[..., None]) + WEED_B * (0.8 * weed[..., None])
    hgt = 0.002 * sc + 0.001 * fine
    return _maps(alb, rough, hgt, 2 * math.pi * 0.18 / W, (y1 - y0) / H)


def textured_material(maps, normal_scale=1.0):
    alb, orm, nrm = maps
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(1.0, 1.0, 1.0)
    m.map = alb
    m.roughness = 1.0
    m.roughness_map = orm
    m.metalness = 0.0
    m.normal_map = nrm
    m.normal_scale = tp.Vector2(normal_scale, normal_scale)
    return m


def block_walls_geometry(lx, dz, top, bottom, face_s):
    """The four walls of a block centred on its local origin (x along, z across), y absolute.
    face_s: {"+z": (s0, s1), "+x": ..., "-z": ..., "-x": ...} perimeter metres at each face's
    start and end, walking it counter-clockwise seen from above (+z face from -x to +x, ...).
    UVs: u = s / face_s["len"], v = (y - bottom) / (top - bottom)."""
    hx, hz = 0.5 * lx, 0.5 * dz
    faces = {"+z": ((-hx, hz), (hx, hz), (0, 1)), "+x": ((hx, hz), (hx, -hz), (1, 0)),
             "-z": ((hx, -hz), (-hx, -hz), (0, -1)), "-x": ((-hx, -hz), (-hx, hz), (-1, 0))}
    P, N, UV, I = [], [], [], []
    plen = face_s["len"]
    for k, (a, b, n) in faces.items():
        if k not in face_s:
            continue
        s0, s1 = face_s[k]
        base = len(P)
        for (x, z), s in ((a, s0), (b, s1)):
            for yy in (bottom, top):
                P.append((x, yy, z))
                N.append((n[0], 0.0, n[1]))
                UV.append((s / plen, (yy - bottom) / (top - bottom)))
        # a0b, a0t, b0b, b0t -> outward-facing (counter-clockwise seen from outside)
        I += [base, base + 2, base + 1, base + 1, base + 2, base + 3]
    g = tp.BufferGeometry()
    g.set_attribute("position", np.asarray(P, np.float32))
    g.set_attribute("normal", np.asarray(N, np.float32))
    g.set_attribute("uv", np.asarray(UV, np.float32))
    g.set_index(np.asarray(I, np.uint32))
    return g


def deck_geometry(lx, dz, top):
    hx, hz = 0.5 * lx, 0.5 * dz
    P = np.float32([(-hx, top, -hz), (hx, top, -hz), (-hx, top, hz), (hx, top, hz)])
    N = np.float32([(0, 1, 0)] * 4)
    UV = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    g = tp.BufferGeometry()
    g.set_attribute("position", P)
    g.set_attribute("normal", N)
    g.set_attribute("uv", UV)
    g.set_index(np.uint32([0, 2, 1, 1, 2, 3]))
    return g


def quay_ladder(top, foot=-1.8, width=0.42, standoff=0.14):
    """A steel quay ladder in its own frame (x along the wall, z out of it, y absolute, the wall
    face at z = 0): two flat-bar stiles from the foot to the cap with hoop handles 0.9 m above the
    deck, round rungs every 0.3 m, wall brackets. Galvanised above the wet line, rusty and weedy
    below it."""
    g = tp.Group()
    galv = tp.MeshStandardMaterial()
    galv.color = tp.Color(0.30, 0.29, 0.27)
    galv.roughness, galv.metalness = 0.55, 0.6
    rusty = tp.MeshStandardMaterial()
    rusty.color = tp.Color(0.13, 0.08, 0.04)
    rusty.roughness, rusty.metalness = 0.6, 0.2
    wet_y = 0.7

    def bar(x0, y0, z0, x1, y1, z1, t, mat):
        L = math.dist((x0, y0, z0), (x1, y1, z1))
        m = tp.Mesh(tp.BoxGeometry(t, L, t), mat)
        m.position.set(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
        if abs(z1 - z0) > 1e-6 or abs(x1 - x0) > 1e-6:
            if abs(y1 - y0) < 1e-6:
                if abs(z1 - z0) > abs(x1 - x0):
                    m.rotation.x = math.pi / 2
                else:
                    m.rotation.z = math.pi / 2
        m.cast_shadow = True
        m.receive_shadow = True
        g.add(m)

    for sx in (-0.5 * width, 0.5 * width):
        bar(sx, foot, standoff, sx, wet_y, standoff, 0.05, rusty)
        bar(sx, wet_y, standoff, sx, top + 0.9, standoff, 0.05, galv)
        bar(sx, top + 0.9, standoff, sx, top + 0.9, -0.55, 0.045, galv)          # the hoop over the edge
        bar(sx, top, -0.55, sx, top + 0.9, -0.55, 0.045, galv)                   # down into the deck
        for yb in np.arange(foot + 0.4, top - 0.3, 1.5):
            bar(sx, float(yb), 0.0, sx, float(yb), standoff, 0.04, rusty if yb < wet_y else galv)
    for yr in np.arange(foot + 0.15, top - 0.05, 0.30):
        m = tp.Mesh(tp.CylinderGeometry(0.014, 0.014, width, 8), rusty if yr < wet_y else galv)
        m.rotation.z = math.pi / 2
        m.position.set(0.0, float(yr), standoff)
        m.cast_shadow = True
        g.add(m)
    return g


def edge_beams(edges, top, h=0.28, d=0.26, proud=0.10, out=0.05):
    """The dark timber edge beam along a quay's sea edges: a kerb `proud` m over the deck, standing
    `out` m off the wall face. edges: (x0, z0, x1, z1, outward nx, nz) in the block's local frame,
    each a straight face line. Returns a Group."""
    g = tp.Group()
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(0.085, 0.074, 0.062)
    m.roughness = 0.85
    for x0, z0, x1, z1, nx, nz in edges:
        L = math.hypot(x1 - x0, z1 - z0)
        b = tp.Mesh(tp.BoxGeometry(L, h, d), m)
        cx = 0.5 * (x0 + x1) + nx * (out - 0.5 * d)
        cz = 0.5 * (z0 + z1) + nz * (out - 0.5 * d)
        b.position.set(cx, top + proud - 0.5 * h, cz)
        b.rotation.y = -math.atan2(z1 - z0, x1 - x0)
        b.cast_shadow = True
        b.receive_shadow = True
        g.add(b)
    return g


def root_ramp_geometry(lx, dz, top, y_far, run=8.0, skirt_to=-1.0):
    """The mole's root: a strip from the deck's -x edge (at `top`) sloping `run` m inland down to
    y_far (one height per across-position, linspace over the deck width), so the deck meets the
    shore flush instead of standing proud of it; side skirts close it down to skirt_to. UVs mirror
    the deck's first `run` m (u = distance inland / lx)."""
    n = len(y_far)
    zs = np.linspace(-0.5 * dz, 0.5 * dz, n)
    x0, x1 = -0.5 * lx, -0.5 * lx - run
    P, N, UV, I = [], [], [], []
    for j, z in enumerate(zs):
        v = (z + 0.5 * dz) / dz
        for x, y, u in ((x0, top, 0.0), (x1, y_far[j], run / lx)):
            P.append((x, y, z))
            UV.append((u, v))
            dy = top - y_far[j]
            nn = np.array([-dy, run, 0.0])
            N.append(tuple(nn / np.linalg.norm(nn)))
    for j in range(n - 1):
        a, b, c, d = 2 * j, 2 * j + 1, 2 * j + 2, 2 * j + 3        # a near_j, b far_j, c near_j+1, d far_j+1
        I += [a, b, c, b, d, c]
    for side, j, nz in ((0, 0, -1.0), (1, n - 1, 1.0)):            # skirts under the two side edges
        base = len(P)
        for x, y in ((x0, top), (x1, y_far[j]), (x0, skirt_to), (x1, skirt_to)):
            P.append((x, y, zs[j]))
            N.append((0.0, 0.0, nz))
            UV.append(((x0 - x) / lx, 0.0))
        I += [base, base + 2, base + 1, base + 1, base + 2, base + 3] if nz < 0 else \
             [base, base + 1, base + 2, base + 1, base + 3, base + 2]
    g = tp.BufferGeometry()
    g.set_attribute("position", np.asarray(P, np.float32))
    g.set_attribute("normal", np.asarray(N, np.float32))
    g.set_attribute("uv", np.asarray(UV, np.float32))
    g.set_index(np.asarray(I, np.uint32))
    return g

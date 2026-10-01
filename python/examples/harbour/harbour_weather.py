"""Look pass L2 of plans/harbour-scene.md: weathering on the boats and the marks, applied at
RUNTIME to the loaded glbs (the builders and the committed .glb files stay clean).

Every weathered mesh is re-laid as a triangle soup and each triangle projected on the plane its
normal faces most (the sides on (x, y), the ends on (z, y), decks and caps on (x, z)), in the
asset's OWN frame (the vessel frame of its spec: x forward, y up from the baseline, z starboard;
a mark's frame from buoys_hydro.json), at a fixed number of texels per metre. A texel therefore
knows its height above the design waterline, so the grime sits at the true waterline whatever the
mesh, and its UVs are computed from positions (the glb's own UVs are not used). Each mesh gets
its own material (albedo + roughness maps over white); the mesh objects stay, so the label ids
(renderer.set_instance_id, keyed per mesh) are untouched.

- sjark: a thin, uneven brown-green scum line on and just above the boot-top, slime on the
  antifouling at the water, rust / dirt streaks down from the freeing ports, the bollard
  fairleads, the anchor hawse and the starboard net hauler's roller (narrow and strong at the
  source, fading downward, always darker than the paint), scuffs along the rubbing strake and dark
  fender marks at the bow, a dirtier matte working deck, a slight overall loss of gloss;
- Trollfjord (maintained: about 60 % of a sjark's amount): grime on the antifouling strip and
  the foot of the black, rust from the anchor pockets and the mooring openings in the red band,
  streaks under the lifeboat davits;
- marks: an algae band at and just above the waterline, sun-faded paint above it, seagull
  droppings (white splats on the up faces, streaks running down) on the topmark, the lantern and
  the top of the spar;
- floats: faded orange (white) with a grimy waterline.

weather(obj, ...) with amount 0 does nothing (the clean look). Seeds vary the amount per boat.
"""
import math

import numpy as np

import threepp as tp
from harbour_quay import RUST, WEED_B, WEED_G, _smooth

SCUM_G = np.float32([0.090, 0.095, 0.040])       # waterline scum: green-brown film
SCUM_B = np.float32([0.105, 0.075, 0.040])       # ... brown
DIRT = np.float32([0.120, 0.085, 0.050])         # dirty brown run-off (fish, rust, diesel)
RUST_DARK = np.float32([0.22, 0.085, 0.030])     # rust run-off on paint: orange-brown, darker than a light hull
DIRT_DARK = np.float32([0.11, 0.065, 0.035])     # scupper run-off: dark brown
SCUM_DG = np.float32([0.045, 0.050, 0.022])      # the waterline scum line: dark green-brown ...
SCUM_DB = np.float32([0.060, 0.045, 0.025])      # ... and dark brown
RUBBER_MARK = np.float32([0.09, 0.09, 0.085])    # fender / quay marks on a pale hull
DROPPINGS = np.float32([0.78, 0.77, 0.72])
SLIME = np.float32([0.11, 0.15, 0.05])           # fresh green slime at the top of an algae band


# --------------------------------------------------------------------------- #
#  World-anchored value noise (continuous across the atlas regions)
# --------------------------------------------------------------------------- #
def _hash(i, j, seed):
    h = np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453
    return h - np.floor(h)


def vnoise(x, y, seed=0.0):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    i, j = np.floor(x), np.floor(y)
    fx, fy = x - i, y - j
    ux, uy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    a, b = _hash(i, j, seed), _hash(i + 1, j, seed)
    c, d = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


def fbm(x, y, seed=0.0, octaves=3):
    out, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        out = out + amp * vnoise(np.asarray(x) * 2 ** k, np.asarray(y) * 2 ** k, seed + 17.0 * k)
        tot += amp
        amp *= 0.5
    return out / tot


def _mix(alb, col, m):
    m = np.clip(m, 0.0, 1.0)[..., None]
    return alb * (1.0 - m) + np.asarray(col, np.float64) * m


def _lum(c):
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


# --------------------------------------------------------------------------- #
#  The skin: a mesh re-laid with position-derived UVs and its own baked material
# --------------------------------------------------------------------------- #
def _rel_matrix(node, root):
    return np.linalg.inv(root.matrix_world.to_numpy().astype(np.float64)) @ node.matrix_world.to_numpy().astype(np.float64)


def _textures(alb, rough):
    alb = np.clip(alb, 0.0, 1.0)
    srgb = np.where(alb <= 0.0031308, alb * 12.92, 1.055 * np.power(np.maximum(alb, 0.0031308), 1.0 / 2.4) - 0.055)
    a8 = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    orm = np.zeros(alb.shape[:2] + (3,), np.uint8)
    orm[..., 0] = 255
    orm[..., 1] = (np.clip(rough, 0.04, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return tp.data_texture(a8, srgb=True), tp.data_texture(orm, srgb=False)


def skin(mesh, root, field, ppm, pad=3, max_px=8192, scale=None):
    """Re-lay `mesh` (a triangle soup with UVs from its positions in root's frame) and give it a
    material baked from field(region, X, Y, Z) -> (albedo linear (h, w, 3), roughness (h, w)),
    evaluated at every texel's point in root's frame. Regions: 0 port, 1 starboard (x, y),
    2 aft, 3 forward (z, y), 4 up, 5 down (x, z); scale: {region: factor on ppm} (default: the
    bottoms, never seen from above the water, at an eighth). Returns the atlas size."""
    scale = {5: 0.125} if scale is None else scale
    M = _rel_matrix(mesh, root)
    g = mesh.geometry
    A = {}
    for n in g.attribute_names():
        a = g.get_attribute(n)
        if a is not None:
            A[n] = np.asarray(a, np.float32)
    P = A["position"]
    idx = g.get_index()
    tri = (np.arange(len(P)) if idx is None else np.asarray(idx, np.int64)).reshape(-1, 3)
    R, t = M[:3, :3], M[:3, 3]
    V = (P.astype(np.float64) @ R.T + t)[tri]                       # (T, 3, 3) in root's frame
    fn = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    if "normal" in A:
        Nv = A["normal"].astype(np.float64) @ np.linalg.inv(R)
        s = np.einsum("ij,ij->i", fn, Nv[tri].sum(1))
        fn = fn * np.where(s < 0.0, -1.0, 1.0)[:, None]
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    reg = np.where(fn[:, 1] > 0.65, 4, np.where(fn[:, 1] < -0.65, 5, np.where(
        fn[:, 0] < -0.8, 2, np.where(fn[:, 0] > 0.8, 3, np.where(fn[:, 2] < 0.0, 0, 1)))))
    V = V.reshape(-1, 3)
    rv = np.repeat(reg, 3)
    proj = {0: (0, 1), 1: (0, 1), 2: (2, 1), 3: (2, 1), 4: (0, 2), 5: (0, 2)}
    AB = np.zeros((len(V), 2))
    for r, (ia, ib) in proj.items():
        k = rv == r
        AB[k, 0], AB[k, 1] = V[k, ia], V[k, ib]
    while True:
        sizes = []
        for r in range(6):
            k = rv == r
            if not k.any():
                continue
            lo, hi = AB[k].min(0), AB[k].max(0)
            pr = ppm * scale.get(r, 1.0)
            w = int(math.ceil((hi[0] - lo[0]) * pr)) + 2 * pad
            h = int(math.ceil((hi[1] - lo[1]) * pr)) + 2 * pad
            sizes.append((r, lo, w, h, V[k].mean(0), pr))
        # stack the regions down (wide regions: a hull side) or across (tall ones: a spar)
        down = (max(s[2] for s in sizes), sum(s[3] for s in sizes))
        across = (sum(s[2] for s in sizes), max(s[3] for s in sizes))
        W, H = down if max(down) <= max(across) else across
        if W <= max_px and H <= max_px:
            break
        ppm *= 0.95 * max_px / max(W, H)
    blocks, off = [], 0
    for r, lo, w, h, mean, pr in sizes:
        blocks.append((r, lo, w, h, (0, off) if (W, H) == down else (off, 0), mean, pr))
        off += h if (W, H) == down else w
    UV = np.zeros((len(V), 2), np.float32)
    alb = np.zeros((H, W, 3))
    rough = np.ones((H, W))
    for r, lo, w, h, (c0, r0), mean, pr in blocks:
        k = rv == r
        UV[k, 0] = (c0 + pad + (AB[k, 0] - lo[0]) * pr) / W
        UV[k, 1] = (r0 + pad + (AB[k, 1] - lo[1]) * pr) / H
        a = lo[0] + (np.arange(w) + 0.5 - pad) / pr
        b = lo[1] + (np.arange(h) + 0.5 - pad) / pr
        Ag, Bg = np.meshgrid(a, b)
        C = np.full_like(Ag, 0.0)
        if r in (0, 1):
            X, Y, Z = Ag, Bg, C + mean[2]
        elif r in (2, 3):
            X, Y, Z = C + mean[0], Bg, Ag
        else:
            X, Y, Z = Ag, C + mean[1], Bg
        al, ro = field(r, X, Y, Z)
        alb[r0:r0 + h, c0:c0 + w] = al
        rough[r0:r0 + h, c0:c0 + w] = ro
    # the soup: every attribute per corner, the UVs replaced
    for n, a in A.items():
        if n != "uv":
            g.set_attribute(n, np.ascontiguousarray(a[tri].reshape(-1, a.shape[1])))
    g.set_attribute("uv", UV)
    g.set_index(np.arange(len(V), dtype=np.uint32))
    old = mesh.material
    m = tp.MeshStandardMaterial()
    m.name = f"{old.name}_weathered"
    m.color = tp.Color(1.0, 1.0, 1.0)
    m.map, m.roughness_map = _textures(alb, rough)
    m.roughness = 1.0
    m.metalness = float(old.metalness)
    m.side = old.side
    mesh.set_material(m)
    return W, H


def _meshes(node):
    out = []
    node.traverse(lambda o: out.append(o) if isinstance(o, tp.Mesh) else None)
    return out


def _base(mesh):
    c = mesh.material.color
    return np.float64([c.r, c.g, c.b]), float(mesh.material.roughness)


def _frame_points(mesh, root):
    M = _rel_matrix(mesh, root)
    P = np.asarray(mesh.geometry.get_attribute("position"), np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


# --------------------------------------------------------------------------- #
#  Streaks and scuffs: features in the asset frame, rasterised into any mesh's field
# --------------------------------------------------------------------------- #
def _streaks(X, Y, side, S, seed):
    """Max mask of the streaks S (x0, y0, side, width, length, strength) on a side raster."""
    m = np.zeros_like(X)
    for k, (x0, y0, sd, w, L, kk) in enumerate(S):
        if sd != 0 and sd != side:
            continue
        t = (y0 - Y) / L
        live = (t >= -0.02) & (t <= 1.0)
        if not live.any():
            continue
        tt = np.clip(t, 0.0, 1.0)
        wid = w * (0.55 + 0.75 * tt)                                      # narrow at the source
        wob = 0.25 * w * (vnoise(Y / 0.4, k, seed) - 0.5)               # the run wanders a little
        across = np.exp(-((X - x0 - wob) / wid) ** 2)
        streak = 0.45 + 0.75 * vnoise(X / max(0.15 * w, 0.004), Y / (0.25 * L), seed + k)
        head = 1.0 + 0.8 * np.exp(-(t / 0.10) ** 2)                      # strong at the source
        m = np.maximum(m, kk * live * across * (1.0 - tt) ** 1.5 * streak * head)
    return np.clip(m, 0.0, 1.0)


def _scuffs(X, Y, side, S):
    """Max mask of short scrapes S (x, y, side, half-length, half-height, strength)."""
    m = np.zeros_like(X)
    if not S:
        return m
    xa, yb = X[0, :], Y[:, 0]
    for (x0, y0, sd, la, lb, kk) in S:
        if sd != side:
            continue
        i0, i1 = np.searchsorted(xa, [x0 - 3 * la, x0 + 3 * la])
        j0, j1 = np.searchsorted(yb, [y0 - 3 * lb, y0 + 3 * lb])
        if i1 <= i0 or j1 <= j0:
            continue
        sub = np.exp(-((xa[None, i0:i1] - x0) / la) ** 2 - ((yb[j0:j1, None] - y0) / lb) ** 2) * kk
        m[j0:j1, i0:i1] = np.maximum(m[j0:j1, i0:i1], sub)
    return np.clip(m, 0.0, 1.0)


def _side_of(r, Z):
    if r == 0:
        return -1
    if r == 1:
        return 1
    return 0


# --------------------------------------------------------------------------- #
#  The sjark
# --------------------------------------------------------------------------- #
SJARK_KINDS = {"hull_topside", "boot_top", "antifouling", "rub_black", "bulwark_paint", "bulwark_inner",
               "deck_grey", "cap_rail", "house_paint", "roof_white", "reg_mark"}


def weather_sjark(obj, spec, design, seed=0, amount=1.0, ppm=120.0):
    """obj: a loaded sjark glb at identity; spec / design: sjark_spec.json and its design
    condition (sjark_hydro.json). Returns a one-line summary."""
    if amount <= 0.0:
        return "clean"
    rng = np.random.default_rng(1000 + seed)
    amt = amount * rng.uniform(0.75, 1.25)
    root = obj.get_object_by_name("sjark") or obj
    wl0, trim = design["waterline_y_at_x0"], math.radians(design.get("trim_deg", 0.0))
    boot_top = spec["paint"]["boot_top_y"]
    meshes = _meshes(root)
    by_mat = {}
    for me in meshes:
        by_mat.setdefault(me.material.name, []).append(me)
    # the sheer (deck edge) and the cap rail along the length, per side, from the meshes
    def profile(mat, fn):
        pts = np.concatenate([_frame_points(me, root) for me in by_mat.get(mat, [])] or [np.zeros((0, 3))])
        out = {}
        for sd in (-1, 1):
            q = pts[np.sign(pts[:, 2]) == sd]
            xs = np.arange(-5.5, 5.75, 0.25)
            ys = np.array([fn(q[np.abs(q[:, 0] - x) < 0.2, 1]) if (np.abs(q[:, 0] - x) < 0.2).any() else np.nan for x in xs])
            ok = ~np.isnan(ys)
            out[sd] = (xs[ok], ys[ok])
        return out
    sheer = profile("hull_topside", np.max)
    cap = profile("cap_rail", np.max)
    sheer_y = lambda x, sd: float(np.interp(x, *sheer[sd]))
    cap_y = lambda x, sd: float(np.interp(x, *cap[sd]))
    # freeing ports: their bottom edges, from the decal meshes
    ports = []
    for me in by_mat.get("freeing_port", []):
        q = _frame_points(me, root)
        for x in spec["bulwark"]["freeing_ports"]["x"]:
            for sd in (-1, 1):
                k = (np.abs(q[:, 0] - x) < 0.3) & (np.sign(q[:, 2]) == sd)
                if k.any():
                    ports.append((x, float(q[k, 1].min()), sd))
    S_dirt, S_rust = [], []
    for x, y, sd in ports:                    # run-off from the deck: dirty brown, over rail and topside
        if rng.random() < 0.85:
            S_dirt.append((x + rng.uniform(-0.12, 0.12), y + 0.01, sd, rng.uniform(0.08, 0.13),
                           rng.uniform(1.0, 1.5), rng.uniform(0.9, 1.0) * min(1.2 * amt, 1.0)))
    for n in ("bollard_fwd_port", "bollard_aft_port", "bollard_fwd_stbd", "bollard_aft_stbd"):
        b = obj.get_object_by_name(n)
        if b is None:
            continue
        p = _rel_matrix(b, root)[:3, 3]
        sd = 1 if "stbd" in n else -1
        for dx in (-0.35, 0.35):              # the fairleads either side of the bollard
            if rng.random() < 0.7:
                x = p[0] + dx + rng.uniform(-0.08, 0.08)
                S_rust.append((x, cap_y(x, sd) - 0.04, sd, rng.uniform(0.025, 0.05), rng.uniform(0.7, 1.5),
                               rng.uniform(0.5, 0.9) * min(amt, 1.4)))
    for sd in (-1, 1):                        # the anchor hawse at the bow, rust down the flare
        x = 4.85 + rng.uniform(-0.1, 0.1)
        S_rust.append((x, cap_y(x, sd) - 0.06, sd, rng.uniform(0.035, 0.06), rng.uniform(1.3, 2.0), 0.85 * min(amt, 1.4)))
    # scrapes: along the rubbing strake (the whole length, denser at the bow), and on the topside
    # at the bow and under the starboard net hauler where the net comes over the rail
    scuffs = []
    for sd in (-1, 1):
        n_rub = int(60 * amt)
        for _ in range(n_rub):
            x = rng.choice([rng.uniform(-5.4, 5.4), rng.uniform(3.0, 5.4)], p=[0.6, 0.4])
            y = sheer_y(x, sd) - rng.uniform(0.0, 0.2)
            scuffs.append((x, y, sd, rng.uniform(0.04, 0.22), rng.uniform(0.004, 0.012), rng.uniform(0.4, 1.0)))
        for _ in range(int(26 * amt)):        # the bow, below the strake: fendering, landing
            x = rng.uniform(3.4, 5.35)
            y = sheer_y(x, sd) - rng.uniform(0.15, 0.95)
            scuffs.append((x, y, sd, rng.uniform(0.03, 0.16), rng.uniform(0.004, 0.012), rng.uniform(0.35, 0.9)))
    for _ in range(int(5 * amt) + 2):         # the hauler: dirty run-off from the roller down the side
        x = rng.uniform(-1.85, -0.65)
        S_dirt.append((x, cap_y(x, 1) - 0.03, 1, rng.uniform(0.03, 0.06), rng.uniform(0.6, 1.3),
                       rng.uniform(0.5, 0.85) * min(amt, 1.4)))
    F = dict(amt=amt, seed=float(seed) * 3.7 + 1.0, wl=lambda x: wl0 - np.asarray(x) * math.tan(trim),
             boot_top=boot_top, dirt=S_dirt, rust=S_rust, scuffs=scuffs, sheer=sheer_y)
    n, px = 0, 0
    for me in meshes:
        kind = me.material.name
        if kind not in SJARK_KINDS:
            continue
        c0, r0 = _base(me)
        if kind in ("house_paint", "roof_white", "reg_mark", "cap_rail"):
            # no texture: only the gloss goes (a working boat is not polished)
            if kind == "cap_rail":
                W, H = skin(me, root, _sjark_field(kind, F, c0, r0), ppm)
                px += W * H
            else:
                me.material.roughness = float(r0 + (0.55 - r0) * min(1.0, 0.6 * amt)) if r0 < 0.55 else r0
            n += 1
            continue
        W, H = skin(me, root, _sjark_field(kind, F, c0, r0), ppm)
        n += 1
        px += W * H
    return (f"weathered x{amt:.2f}: {n} meshes, {px / 1e6:.1f} Mpx, {len(S_dirt)} run-off + {len(S_rust)} rust "
            f"streaks, {len(scuffs)} scrapes")


def _sjark_field(kind, F, c0, r0):
    amt, seed = F["amt"], F["seed"]

    def field(r, X, Y, Z):
        sd = _side_of(r, Z)
        alb = np.broadcast_to(c0, X.shape + (3,)).copy()
        rough = np.full(X.shape, r0)
        lowf = fbm(X / 2.2, Y / 0.9 + Z / 2.2, seed + 1.0)
        # loss of gloss and a touch of chalking over the whole paint
        if kind in ("hull_topside", "bulwark_paint", "boot_top"):
            g = 0.45 + 0.12 * min(amt, 1.5)
            rough = np.maximum(rough, g + 0.10 * (lowf - 0.5))
            lum = alb @ np.float64([0.2126, 0.7152, 0.0722])
            alb = _mix(alb, lum[..., None] * 0.9 + 0.025, 0.07 * amt * (0.6 + 0.8 * lowf))
            alb *= (1.0 - 0.05 * amt * (lowf - 0.3))[..., None]
        if kind in ("hull_topside", "boot_top", "antifouling"):
            wl = F["wl"](X)
            bt = F["boot_top"]
            # the scum line: on the boot-top and a few cm above it, the top edge uneven
            top = bt + 0.05 + min(amt, 1.4) * (0.02 + 0.04 * vnoise(X / 1.1, sd * 7.0, seed)
                                               + 0.06 * vnoise(X / 0.22, sd * 13.0, seed) ** 3)
            band = 1.0 - _smooth(top - 0.025, top + 0.004, Y)
            band *= _smooth(wl - 0.14, wl - 0.02, Y)                     # above the slime
            patchy = 0.75 + 0.25 * fbm(X / 0.18, Y / 0.035, seed + 3.0)
            hue = vnoise(X / 0.6, Y / 0.15, seed + 5.0)
            col = (SCUM_DG * hue[..., None] + SCUM_DB * (1.0 - hue[..., None])) * (0.55 + 0.3 * patchy[..., None])
            # densest at the boot-top's lower half (the waterline itself)
            dens = np.where(Y < bt, 0.8 + 0.15 * (1.0 - _smooth(wl, bt, Y)), 0.9)
            m = np.clip(band * patchy * dens * min(1.2 * amt, 1.0), 0.0, 0.95)
            alb = _mix(alb, col, m)
            rough = rough * (1.0 - m) + 0.6 * m
            # green slime on the antifouling at the water
            if kind == "antifouling":
                sl = _smooth(wl - 0.30, wl - 0.04, Y) * (0.5 + 0.5 * fbm(X / 0.3, Y / 0.06, seed + 9.0))
                alb = _mix(alb, SLIME * 0.8, 0.7 * sl * min(amt, 1.3))
                rough = rough * (1.0 - sl) + 0.4 * sl
        if kind in ("hull_topside", "bulwark_paint", "rub_black", "boot_top", "cap_rail") and sd != 0:
            # rust and run-off streaks
            md = _streaks(X, Y, sd, F["dirt"], seed + 20.0)
            mr = _streaks(X, Y, sd, F["rust"], seed + 40.0)
            if kind in ("rub_black", "cap_rail"):
                dcol, rcol = DIRT, RUST
            else:                             # never lighter than the paint: scale to <= 70 % of its luminance
                dcol = DIRT_DARK * min(1.0, 0.5 * _lum(c0) / _lum(DIRT_DARK))
                rcol = RUST_DARK * min(1.0, 0.55 * _lum(c0) / _lum(RUST_DARK))
            alb = _mix(alb, dcol, 0.92 * md)
            alb = _mix(alb, rcol, 0.92 * mr)
            rough = rough + (0.75 - rough) * np.clip(md + mr, 0.0, 1.0) * 0.6
            # scrapes
            ms = _scuffs(X, Y, sd, F["scuffs"])
            if kind in ("rub_black", "cap_rail"):
                alb = _mix(alb, np.float64([0.20, 0.195, 0.185]), 0.65 * ms)
            else:                             # fender and quay marks: darker, never a pale tint
                alb = _mix(alb, RUBBER_MARK * min(1.0, 0.6 * _lum(c0) / _lum(RUBBER_MARK)), 0.45 * ms)
            rough = rough * (1.0 - ms) + 0.7 * ms
            # the hauler's worn patch on the starboard bulwark: paint rubbed thin under the rail
            if sd == 1 and kind in ("bulwark_paint", "hull_topside", "cap_rail"):
                wear = _smooth(-2.05, -1.75, X) * (1.0 - _smooth(-0.75, -0.45, X))
                wear = wear * (0.35 + 0.65 * fbm(X / 0.12, Y / 0.25, seed + 31.0))
                wear *= _smooth(F["sheer"](-1.25, 1) - 1.2, F["sheer"](-1.25, 1) + 0.6, Y)
                alb = _mix(alb, alb * 0.7 + DIRT_DARK * 0.3 * min(1.0, 0.7 * _lum(c0) / _lum(DIRT_DARK)),
                           0.35 * wear * min(amt, 1.3))
                rough = rough * (1.0 - wear) + 0.75 * wear
        if kind == "deck_grey":
            # a working deck: dark wet blotches, brown stains where the catch is handled (the
            # starboard working deck aft of the wheelhouse), scuffed paths, matte
            rough = np.full(X.shape, 0.9)
            alb *= (0.82 + 0.3 * fbm(X / 1.4, Z / 1.4, seed + 50.0))[..., None]
            blot = np.clip((fbm(X / 0.5, Z / 0.5, seed + 51.0) - 0.52) * 4.0, 0.0, 1.0)
            alb = _mix(alb, np.float64([0.06, 0.06, 0.055]), 0.55 * blot * min(amt, 1.3))
            rough = rough * (1.0 - blot) + 0.45 * blot
            work = _smooth(0.0, -0.8, X) * _smooth(-0.3, 0.6, Z)
            stain = work * np.clip((fbm(X / 0.35, Z / 0.35, seed + 52.0) - 0.45) * 3.0, 0.0, 1.0)
            alb = _mix(alb, np.float64([0.16, 0.085, 0.055]), 0.5 * stain * min(amt, 1.3))
        if kind == "bulwark_inner":
            # dirt splashed up from the deck
            foot = 1.0 - _smooth(2.25, 2.75, Y)
            alb = _mix(alb, alb * 0.55 + DIRT * 0.3, foot * (0.5 + 0.5 * fbm(X / 0.3, Y / 0.1, seed + 60.0)) * min(amt, 1.3))
            rough = np.maximum(rough, 0.7)
        return alb, rough
    return field


# --------------------------------------------------------------------------- #
#  Trollfjord: the same layers, kept subtle (a maintained ship)
# --------------------------------------------------------------------------- #
TF_KINDS = {"antifouling", "hull_black", "hull_red", "super_white"}


def weather_trollfjord(obj, spec, design, amount=1.0, ppm=24.0, seed=7):
    if amount <= 0.0:
        return "clean"
    rng = np.random.default_rng(2000 + seed)
    amt = 0.6 * amount
    root = obj.get_object_by_name("trollfjord") or obj
    wl0 = design["waterline_y_at_x0"]
    af_top = spec["paint"]["antifouling_top_y"]
    hd = spec["hull_details"]["anchor_pocket"]
    rust = []
    for sd in (-1, 1):
        # the anchor pockets: rust from the hawse down the black to the water
        for k in range(2):
            rust.append((hd["x"] + rng.uniform(-0.45, 0.45), hd["y"] - 0.5 * hd["ry"], sd, rng.uniform(0.45, 0.75),
                         rng.uniform(3.0, 3.4), 1.0))
        # mooring openings in the red band, aft and forward (fairleads at the bollard decks)
        for x in (-58.5, -62.0, -65.0, 58.5, 63.0):
            if rng.random() < 0.8:
                rust.append((x + rng.uniform(-0.4, 0.4), 12.75, sd, rng.uniform(0.10, 0.22), rng.uniform(2.0, 5.5),
                             rng.uniform(0.4, 0.85)))
    davits = []
    for b in spec["lifeboats"]["boats"]:
        for e in (-0.42, 0.42):
            x = b["x"] + e * b["length"]
            for sd in (-1, 1):
                davits.append((x + rng.uniform(-0.3, 0.3), spec["lifeboats"]["y_keel"] - 0.1, sd, rng.uniform(0.08, 0.16),
                               rng.uniform(1.5, 3.5), rng.uniform(0.5, 0.9)))
    F = dict(amt=amt, seed=11.0, wl=wl0, af_top=af_top, rust=rust, davits=davits)
    n, px = 0, 0
    for me in _meshes(root):
        kind = me.material.name
        if kind not in TF_KINDS:
            continue
        c0, r0 = _base(me)
        # the superstructure's roofs and the hull's bottom: never seen up close, a coarse skin
        W, H = skin(me, root, _tf_field(kind, F, c0, r0), ppm, scale={4: 0.1, 5: 0.06})
        n += 1
        px += W * H
    return f"weathered x{amt:.2f}: {n} meshes, {px / 1e6:.1f} Mpx, {len(rust)} rust + {len(davits)} davit streaks"


def _tf_field(kind, F, c0, r0):
    amt, seed, wl = F["amt"], F["seed"], F["wl"]

    def field(r, X, Y, Z):
        sd = _side_of(r, Z)
        alb = np.broadcast_to(c0, X.shape + (3,)).copy()
        rough = np.full(X.shape, r0)
        lowf = fbm(X / 12.0, Y / 3.0, seed)
        if kind in ("hull_black", "hull_red", "super_white"):
            rough = np.maximum(rough, 0.48 + 0.08 * (lowf - 0.5))
        if kind == "antifouling":
            # grime on the red strip between the water and the black: darker, browner, patchy
            g = _smooth(wl - 0.4, wl + 0.05, Y) * (0.5 + 0.5 * fbm(X / 1.5, Y / 0.15, seed + 1.0))
            alb = _mix(alb, np.float64([0.055, 0.035, 0.022]), 0.85 * g * min(1.0, 2.0 * amt))
            rough = rough * (1.0 - g) + 0.55 * g
        if kind == "hull_black":
            # the foot of the black: a faint brown-grey tide mark, uneven top
            top = F["af_top"] + 0.12 + 0.25 * vnoise(X / 6.0, sd * 3.0, seed) + 0.15 * vnoise(X / 1.2, sd * 5.0, seed) ** 2
            band = (1.0 - _smooth(top - 0.15, top, Y)) * (0.5 + 0.5 * fbm(X / 0.8, Y / 0.1, seed + 2.0))
            alb = _mix(alb, np.float64([0.045, 0.040, 0.028]), 0.8 * band * min(1.0, 2.0 * amt))
        if kind in ("hull_black", "hull_red") and sd != 0:
            mr = _streaks(X, Y, sd, F["rust"], seed + 3.0)
            col = RUST * (0.95 if kind == "hull_black" else 0.5)
            alb = _mix(alb, col, 0.9 * mr * min(1.0, 2.2 * amt))
        if kind == "super_white" and sd != 0:
            md = _streaks(X, Y, sd, F["davits"], seed + 4.0)
            alb = _mix(alb, RUST * 0.6 + DIRT * 0.4, 0.35 * md * min(1.0, 2.2 * amt))
        return alb, rough
    return field


# --------------------------------------------------------------------------- #
#  Marks and floats
# --------------------------------------------------------------------------- #
def weather_mark(obj, mark, hydro, seed=0, amount=1.0, ppm=420.0):
    """obj: a clone of a buoys.glb root; hydro: buoys_hydro.json (load_marks())."""
    if amount <= 0.0:
        return "clean"
    rng = np.random.default_rng(3000 + seed)
    amt = amount * rng.uniform(0.8, 1.2)
    wl = hydro["marks"][mark]["design"]["waterline_y"]
    meshes = _meshes(obj)
    pts = [_frame_points(me, obj) for me in meshes]
    tops = [p[:, 1].max() for p in pts]
    rads = [float(np.hypot(p[:, 0], p[:, 2]).max()) for p in pts]
    y_top = max(tops)
    floatk = mark.startswith("blaase") or mark == "mooring_buoy" or mark == "garnblaase"
    n = 0
    for me, ytm, rad in zip(meshes, tops, rads):
        kind = me.material.name
        if not (kind.startswith("paint_") or kind.startswith("float_") or kind in ("steel_black", "lantern_black")):
            continue
        c0, r0 = _base(me)
        # droppings: on the top of the mark (lantern, topmark, its post, the top of the spar)
        birds = (not floatk) and ytm > y_top - 0.75
        streaks = []
        if birds:
            for k in range(int(rng.integers(6, 12) * min(amt, 1.4))):
                streaks.append((rng.uniform(-0.85, 0.85) * rad, ytm - rng.uniform(0.0, 0.03), 0, rng.uniform(0.004, 0.012),
                                rng.uniform(0.06, 0.25 if ytm > y_top - 0.3 else 0.9), rng.uniform(0.6, 1.0)))
        sk = skin(me, obj, _mark_field(kind, dict(amt=amt, seed=seed * 5.3 + 2.0, wl=wl, top=ytm, y_top=y_top,
                                                  floatk=floatk, birds=birds, streaks=streaks), c0, r0),
                  ppm if not floatk else 300.0)
        n += 1
        del sk
    return f"weathered x{amt:.2f}: {n} meshes (waterline {wl:.2f} m)"


def _mark_field(kind, F, c0, r0):
    amt, seed, wl = F["amt"], F["seed"], F["wl"]

    def field(r, X, Y, Z):
        alb = np.broadcast_to(c0, X.shape + (3,)).copy()
        rough = np.full(X.shape, r0)
        A = X if r in (0, 1, 4, 5) else Z                                # across the face
        Bc = Z if r in (4, 5) else Y
        if kind.startswith("paint_") or kind.startswith("float_"):
            # sun-faded paint, more toward the top, chalky and matte
            fade = min(amt, 1.4) * (0.30 + 0.25 * _smooth(wl, F["y_top"], Y)) * (0.7 + 0.6 * fbm(A / 0.15 + r * 3.1, Y / 0.4, seed))
            if kind == "float_orange":
                faded = np.float64([0.78, 0.30, 0.13])
            else:
                lum = _lum(c0)
                faded = c0 * 0.62 + lum * 0.2 + 0.05
            alb = _mix(alb, faded, np.clip(fade * (0.9 if kind == "float_orange" else 0.6), 0.0, 0.8) * (Y > wl - 0.1))
            rough = np.maximum(rough, 0.62 + 0.1 * min(amt, 1.4))
            if F["floatk"]:
                # a grimy waterline: brown scum a few cm above the water, drips
                top = wl + 0.025 + 0.03 * vnoise(A / 0.04 + r * 7.0, 1.0, seed) + 0.03 * vnoise(A / 0.012 + r * 5.0, 2.0, seed) ** 4
                band = (1.0 - _smooth(top - 0.03, top, Y)) * _smooth(wl - 0.12, wl - 0.02, Y)
                band *= 0.5 + 0.5 * fbm(A / 0.03 + r * 2.0, Y / 0.01, seed + 1.0)
                alb = _mix(alb, SCUM_B * 0.9, 0.85 * band * min(amt, 1.3))
                below = 1.0 - _smooth(wl - 0.06, wl - 0.0, Y)
                alb = _mix(alb, alb * 0.45 + SCUM_G * 0.5, 0.75 * below)
                rough = rough * (1.0 - band) + 0.5 * band
            else:
                # algae: dense below the water, a ragged band up to ~0.15..0.4 m above it, bright slime at its top
                rag = 0.5 * vnoise(A / 0.03 + r * 11.0, 0.5, seed) + 0.5 * vnoise(A / 0.008 + r * 3.0, 1.5, seed)
                top = wl + (0.12 + 0.13 * rag) * min(amt, 1.4)
                weed = 1.0 - _smooth(top - 0.05, top, Y)
                patch = fbm(A / 0.02 + r * 4.0, Y / 0.05, seed + 2.0)
                weed *= np.clip(0.95 + 0.5 * (patch - 0.3), 0.0, 1.0)
                weed = np.maximum(weed, 1.0 - _smooth(wl - 0.35, wl - 0.1, Y))
                hue = vnoise(A / 0.02 + r * 6.0, Y / 0.04, seed + 3.0)
                wcol = WEED_G * hue[..., None] + WEED_B * (1.0 - hue[..., None])
                slime = np.exp(-((Y - (top - 0.04)) / 0.035) ** 2)
                wcol = wcol * (1.0 - 0.6 * slime[..., None]) + SLIME * (0.6 * slime[..., None])
                alb = _mix(alb, wcol * (0.8 + 0.4 * patch[..., None]), 0.95 * weed)
                rough = rough * (1.0 - weed) + 0.35 * weed
        if F["birds"]:
            # white splats on the faces that look up, streaks running down the sides
            if r == 4:
                sp = np.clip((fbm(X / 0.025, Z / 0.025, seed + 7.0) - 0.42) * 3.5, 0.0, 1.0) * min(amt, 1.3)
                alb = _mix(alb, DROPPINGS * (0.8 + 0.25 * fbm(X / 0.008, Z / 0.008, seed + 8.0))[..., None], 0.9 * sp)
                rough = rough * (1.0 - sp) + 0.85 * sp
            elif r != 5:
                ms = _streaks(A, Bc, 0, F["streaks"], seed + 9.0)
                cap = _smooth(F["top"] - 0.04, F["top"], Y)                 # the crusted top edge
                ms = np.maximum(ms, cap * np.clip((fbm(A / 0.01 + r, Y / 0.01, seed + 10.0) - 0.4) * 3.0, 0.0, 1.0) * 0.8)
                col = DROPPINGS * (0.75 + 0.3 * vnoise(A / 0.01, Y / 0.05, seed + 11.0))[..., None]
                alb = _mix(alb, col, 0.9 * ms * min(amt, 1.3))
                rough = rough * (1.0 - ms) + 0.85 * ms
        return alb, rough
    return field

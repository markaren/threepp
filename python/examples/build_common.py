"""What the build_*_blender.py generators share: the mesh accumulator and its primitives, the
hydrostatics, and the Blender export.

A generator builds its model with numpy in the frame of its spec (X forward, Y up, Z starboard,
metres), floats it if it is a boat (`hydrostatics`), and hands the parts to Blender
(`make_materials`, `new_object`, `export`) only to write the .glb. bpy and mathutils are imported
inside the functions that need them, so `--check` and `--hydro-only` run on plain Python + numpy.

A generator in a folder of examples/ imports it as the demos import `usv_rig`:

    sys.path.insert(0, os.path.dirname(HERE))          # examples/ (build_common)
    from build_common import Part, T, loft, hydrostatics, make_materials, new_object, export
"""
import math
import os

import numpy as np


# ---------------------------------------------------------------- transforms (vessel frame, numpy 4x4)
def T(x, y, z):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def R(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    M[i, i], M[i, j], M[j, i], M[j, j] = c, -s, s, c
    return M


def Z_TO_X():
    return R("Y", 90.0)


def Z_TO_Y():
    return R("X", -90.0)


def frame_from_forward_up(fwd, up):
    """4x4 rotation whose local -Z = fwd and local +Y = up (threepp camera convention)."""
    zb = -np.asarray(fwd, float)
    zb /= np.linalg.norm(zb)
    y = np.asarray(up, float)
    x = np.cross(y, zb)
    x /= np.linalg.norm(x)
    y = np.cross(zb, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = x, y, zb
    return M


def frame_y_toward(d):
    """4x4 rotation whose local +Y points along d."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(d, ref)
    x /= np.linalg.norm(x)
    zb = np.cross(x, d)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = x, d, zb
    return M


def pitched(look, pitch_down_deg):
    """Forward and up vectors of a sensor looking along the horizontal `look`, pitched down."""
    f = np.asarray(look, float)
    f[1] = 0.0
    f /= np.linalg.norm(f)
    a = math.radians(pitch_down_deg)
    fwd = f * math.cos(a) + np.array([0.0, -math.sin(a), 0.0])
    up = f * math.sin(a) + np.array([0.0, math.cos(a), 0.0])
    return fwd, up

def sensor_frame(pos, look, pitch_down=0.0):
    """Look along `look` (its own tilt kept), then pitch down by pitch_down more."""
    f = np.asarray(look, float)
    f = f / np.linalg.norm(f)
    if pitch_down:
        f, _ = pitched(f.tolist(), pitch_down + math.degrees(math.asin(-f[1])))
    return T(*pos) @ frame_from_forward_up(f, (0.0, 1.0, 0.0))

# ---------------------------------------------------------------- mesh accumulator
def _fan(V, f):
    return [(f[0], f[i], f[i + 1]) for i in range(1, len(f) - 1)]


def newell(V, f):
    n = np.zeros(3)
    for i in range(len(f)):
        a, b = V[f[i]], V[f[(i + 1) % len(f)]]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    return n


def signed_volume(V, faces):
    vol = 0.0
    for f in faces:
        for (a, b, c) in _fan(V, f):
            vol += float(np.dot(V[a], np.cross(V[b], V[c])))
    return vol / 6.0


class Part:
    """Polygons in the vessel frame with a material per face.

    closed=True primitives are oriented outward by their signed volume (so every
    mesh renders with backface culling); buoyant=True ones are also kept as
    triangles for the hydrostatics."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self.mats = []
        self.tris = []

    def add(self, verts, faces, mat, M=None, closed=True, buoyant=False, outward=None):
        V = np.asarray(verts, float).reshape(-1, 3)
        faces = [tuple(int(k) for k in f) for f in faces]
        if M is not None:
            V = V @ M[:3, :3].T + M[:3, 3]
        if closed:
            # A closed, consistently wound surface has face area vectors that sum to zero;
            # one inverted cap leaves twice its area behind (and renders as a hole).
            ns = [newell(V, f) for f in faces]
            tot, mag = np.linalg.norm(sum(ns)), sum(np.linalg.norm(n) for n in ns)
            assert tot <= 1e-9 + 1e-7 * mag, ("primitive not closed or not consistently wound", mat, tot, mag)
            if signed_volume(V, faces) < 0.0:
                faces = [tuple(reversed(f)) for f in faces]
        elif outward is not None:
            # open decal: make its area-weighted (Newell) normal agree with `outward`;
            # a single fan triangle can be degenerate and carry no sign
            if np.dot(sum(newell(V, f) for f in faces), outward) < 0.0:
                faces = [tuple(reversed(f)) for f in faces]
        base = len(self.verts)
        self.verts.extend(map(tuple, V))
        for f in faces:
            self.faces.append(tuple(base + k for k in f))
            self.mats.append(mat)
        if buoyant:
            self.tris.append(np.array([[V[a], V[b], V[c]] for f in faces for (a, b, c) in _fan(V, f)]))

    def add_multi(self, verts, faces, mats, M=None, closed=True, buoyant=False):
        """Like add, with one material name per face."""
        n0 = len(self.faces)
        self.add(verts, faces, "_", M=M, closed=closed, buoyant=buoyant)
        for i, m in enumerate(mats):
            self.mats[n0 + i] = m

# ---------------------------------------------------------------- primitives (outward by construction or by Part.add)
def box(sx, sy, sz):
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    return v, f


def box_between(p0, p1):
    """Axis-aligned box from corner p0 to corner p1, returned with its placement matrix."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s = np.abs(p1 - p0)
    v, f = box(*s)
    return v, f, T(*(0.5 * (p0 + p1)))


def revolve(profile, n):
    """Solid of revolution about +Z. profile = [(r, z), ...] from one end to the other.
    Points with r == 0 become poles; if an end has r > 0 it is capped flat."""
    prof = [(float(r), float(z)) for (r, z) in profile]
    v, f = [], []
    rings = []
    for (r, z) in prof:
        if r <= 1e-9:
            rings.append([len(v)])
            v.append((0.0, 0.0, z))
        else:
            idx = []
            for i in range(n):
                a = 2 * math.pi * i / n
                idx.append(len(v))
                v.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(idx)
    for ra, rb in zip(rings[:-1], rings[1:]):
        if len(ra) == 1 and len(rb) == 1:
            continue
        if len(ra) == 1:
            for i in range(n):
                f.append((ra[0], rb[(i + 1) % n], rb[i]))
        elif len(rb) == 1:
            for i in range(n):
                f.append((ra[i], ra[(i + 1) % n], rb[0]))
        else:
            for i in range(n):
                j = (i + 1) % n
                f.append((ra[i], ra[j], rb[j], rb[i]))
    for ring, first in ((rings[0], True), (rings[-1], False)):
        if len(ring) > 1:            # flat cap with its own vertices; the first one runs backwards
            b = len(v)
            v.extend(v[k] for k in ring)
            idx = range(n - 1, -1, -1) if first else range(n)
            f.append(tuple(b + i for i in idx))
    return v, f


def cylinder(r, z0, z1, n):
    return revolve([(r, z0), (r, z1)], n)


def frustum(r0, r1, z0, z1, n):
    return revolve([(r0, z0), (r1, z1)], n)


def strut(part, p0, p1, r, mat, n=16):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    v, f = cylinder(r, 0.0, float(np.linalg.norm(d)), n)
    part.add(v, f, mat, T(*p0) @ frame_y_toward(d) @ Z_TO_Y())


def extrude_xy(poly, z0, z1):
    """Prism along Z of a simple polygon in the XY plane. Caps are n-gons with their own vertices."""
    n = len(poly)
    v, f = [], []
    for (x, y) in poly:
        v += [(x, y, z0), (x, y, z1)]
    for i in range(n):
        j = (i + 1) % n
        f.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    b0 = len(v)
    v += [(x, y, z0) for (x, y) in poly]
    b1 = len(v)
    v += [(x, y, z1) for (x, y) in poly]
    f.append(tuple(b0 + i for i in reversed(range(n))))
    f.append(tuple(b1 + i for i in range(n)))
    return v, f


def loft(rings, caps=True):
    """Closed surface through rings of equal size (each a closed loop). Returns v, f and the
    face index range of the side quads; each side quad k of a ring pair is in segment k."""
    rings = [np.asarray(r, float) for r in rings]
    N = len(rings[0])
    v = [tuple(p) for r in rings for p in r]
    f = []
    for i in range(len(rings) - 1):
        for k in range(N):
            k1 = (k + 1) % N
            f.append((i * N + k, i * N + k1, (i + 1) * N + k1, (i + 1) * N + k))
    if caps:
        # The side quads walk ring 0 forwards and the last ring backwards, so the first
        # cap must run backwards and the last forwards for the caps to face out.
        for r, first in ((rings[0], True), (rings[-1], False)):
            b = len(v)
            v.extend(tuple(p) for p in r)
            f.append(tuple(b + k for k in (range(N - 1, -1, -1) if first else range(N))))
    return v, f

def annulus(r0, r1, z0, z1, n):
    """Closed ring (a thick-walled tube) about +Z from z0 to z1, inner radius r0, outer r1."""
    loops = [(r0, z0), (r1, z0), (r1, z1), (r0, z1)]
    v, f = [], []
    for (r, z) in loops:
        for i in range(n):
            a = 2 * math.pi * i / n
            v.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(4):
        a, b = k * n, ((k + 1) % 4) * n
        for i in range(n):
            j = (i + 1) % n
            f.append((a + i, a + j, b + j, b + i))
    return v, f

# ---------------------------------------------------------------- tables and outlines
def pchip(xs, ys):
    """Monotone cubic interpolant (Fritsch-Carlson) of a table; flat outside."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros_like(ys)
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = d[0], d[-1]
    if d[0] * m[0] <= 0:
        m[0] = 0.0
    if d[-1] * m[-1] <= 0:
        m[-1] = 0.0

    def f(x):
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            return float(ys[-1])
        i = int(np.searchsorted(xs, x) - 1)
        t = (x - xs[i]) / h[i]
        h00, h10 = (1 + 2 * t) * (1 - t) ** 2, t * (1 - t) ** 2
        h01, h11 = t * t * (3 - 2 * t), t * t * (t - 1)
        return float(h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])
    return f

def table(t):
    """pchip of a spec table [[x, y], ...]."""
    a = np.asarray(t, float)
    return pchip(a[:, 0], a[:, 1])

def clip_slab(poly, x0, x1):
    """The polygon [(x, y), ...] clipped to x0 <= x <= x1."""
    def clip(pts, keep, inter):
        out = []
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            if keep(p):
                out.append(p)
                if not keep(q):
                    out.append(inter(p, q))
            elif keep(q):
                out.append(inter(p, q))
        return out

    def ix(xc):
        return lambda p, q: (xc, p[1] + (q[1] - p[1]) * (xc - p[0]) / (q[0] - p[0]))
    pts = clip(poly, lambda p: p[0] >= x0, ix(x0))
    pts = clip(pts, lambda p: p[0] <= x1, ix(x1)) if pts else []
    out = []
    for p in pts:                                        # drop repeated corners
        if not out or abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1]) <= 1e-9:
        out.pop()
    return out

# ---------------------------------------------------------------- hydrostatics (numpy only)
class Hydro:
    """Winding-number column integration of closed, outward triangle meshes."""

    def __init__(self, tris, G):
        self.dx, self.dz, self.dy = G["dx"], G["dz"], G["dy"]
        (self.x0, x1), (self.z0, z1) = G.get("x_range", (-3.02, 3.02)), G.get("z_range", (-1.08, 1.08))
        self.nx = int(round((x1 - self.x0) / self.dx))
        self.nz = int(round((z1 - self.z0) / self.dz))
        self.y0, y1 = G["y_range"]
        self.ny = int(round((y1 - self.y0) / self.dy))
        self.xc = self.x0 + (np.arange(self.nx) + 0.5) * self.dx + 1.37e-4
        self.zc = self.z0 + (np.arange(self.nz) + 0.5) * self.dz + 0.73e-4
        D = np.zeros((self.nx, self.nz, self.ny + 2), np.float64)
        for tri in tris:
            self._raster(D, tri)
        self.closure = float(np.abs(D.sum(axis=2)).max())
        occ = np.clip(np.cumsum(D[:, :, :self.ny], axis=2), 0.0, 1.0)
        self.occ = occ
        yc = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        z = np.zeros((self.nx, self.nz, 1))
        self.cum = np.concatenate([z, np.cumsum(occ, axis=2) * self.dy], axis=2)
        self.cumy = np.concatenate([z, np.cumsum(occ * yc, axis=2) * self.dy], axis=2)
        low = np.argmax(occ > 0.5, axis=2).astype(float)
        self.has = occ.max(axis=2) > 0.5
        self.y_low = np.where(self.has, self.y0 + low * self.dy, np.inf)

    def _raster(self, D, tri):
        a, b, c = tri
        n = np.cross(b - a, c - a)
        if abs(n[1]) < 1e-12:
            return
        s = 1.0 if n[1] < 0.0 else -1.0             # facing down: entering the solid going up
        xmn, xmx = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
        zmn, zmx = min(a[2], b[2], c[2]), max(a[2], b[2], c[2])
        i0 = max(0, int(math.floor((xmn - self.x0) / self.dx - 0.5)))
        i1 = min(self.nx - 1, int(math.ceil((xmx - self.x0) / self.dx - 0.5)))
        j0 = max(0, int(math.floor((zmn - self.z0) / self.dz - 0.5)))
        j1 = min(self.nz - 1, int(math.ceil((zmx - self.z0) / self.dz - 0.5)))
        if i1 < i0 or j1 < j0:
            return
        X, Z = np.meshgrid(self.xc[i0:i1 + 1], self.zc[j0:j1 + 1], indexing="ij")
        det = (b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])
        u = ((b[0] - X) * (c[2] - Z) - (c[0] - X) * (b[2] - Z)) / det
        v = ((c[0] - X) * (a[2] - Z) - (a[0] - X) * (c[2] - Z)) / det
        w = 1.0 - u - v
        m = (u >= 0) & (v >= 0) & (w >= 0)
        if not m.any():
            return
        ii, jj = np.nonzero(m)
        y = u[m] * a[1] + v[m] * b[1] + w[m] * c[1]
        t = np.clip((y - self.y0) / self.dy, 0.0, self.ny + 0.999)
        k = np.floor(t).astype(int)
        fr = (k + 1) - t
        np.add.at(D, (ii + i0, jj + j0, k), s * fr)
        np.add.at(D, (ii + i0, jj + j0, k + 1), s * (1.0 - fr))

    def _at(self, arr, yw):
        """arr[i, j] interpolated at per-column height yw[i, j]."""
        t = np.clip((yw - self.y0) / self.dy, 0.0, self.ny - 1e-6)
        k = np.floor(t).astype(int)
        f = t - k
        lo = np.take_along_axis(arr, k[..., None], axis=2)[..., 0]
        hi = np.take_along_axis(arr, (k + 1)[..., None], axis=2)[..., 0]
        return lo + f * (hi - lo)

    def waterline(self, h, trim):
        yw = h - math.tan(trim) * self.xc
        return np.broadcast_to(yw[:, None], (self.nx, self.nz))

    def state(self, h, trim):
        yw = self.waterline(h, trim)
        vol_col = self._at(self.cum, yw)
        dA = self.dx * self.dz
        V = float(vol_col.sum() * dA)
        if V <= 0.0:
            return V, 0.0, 0.0
        xB = float((vol_col * self.xc[:, None]).sum() * dA / V)
        yB = float(self._at(self.cumy, yw).sum() * dA / V)
        return V, xB, yB

    def solve(self, mass, G, rho):
        """Heave and trim (bow up +) that float `mass` with its CoG at G = (x, y)."""
        Vt = mass / rho

        def h_for(trim):
            lo, hi = self.y0 + 0.01, self.y0 + self.ny * self.dy - 0.01
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self.state(mid, trim)[0] < Vt:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        def resid(trim):
            h = h_for(trim)
            V, xB, yB = self.state(h, trim)
            return (xB - G[0]) * math.cos(trim) - (yB - G[1]) * math.sin(trim), h
        lo, hi = math.radians(-8.0), math.radians(10.0)
        rlo = resid(lo)[0]
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            r = resid(mid)[0]
            if (r > 0) == (rlo > 0):
                lo, rlo = mid, r
            else:
                hi = mid
        trim = 0.5 * (lo + hi)
        return h_for(trim), trim

    def report(self, h, trim, mass, G, rho, g):
        V, xB, yB = self.state(h, trim)
        yw = self.waterline(h, trim)
        occ_w = self._at(np.concatenate([self.occ, self.occ[:, :, -1:]], axis=2), yw - 0.5 * self.dy)
        wp = occ_w > 0.5
        dA = self.dx * self.dz
        Awp = float(wp.sum() * dA)
        X = np.broadcast_to(self.xc[:, None], wp.shape)
        Z = np.broadcast_to(self.zc[None, :], wp.shape)
        lcf = float(X[wp].mean())
        IT = float((Z[wp] ** 2).sum() * dA)
        IL = float(((X[wp] - lcf) ** 2).sum() * dA)
        drafts = np.where(self.has & (yw > self.y_low), yw - self.y_low, 0.0)
        cols = np.nonzero(wp.any(axis=1))[0]
        half = [[round(float(self.xc[i]), 4), round(float(self.zc[wp[i]].max()), 4)] for i in cols]
        return {
            "mass": mass, "cog": [round(G[0], 4), round(G[1], 4), 0.0],
            "displacement_m3": round(V, 4),
            "waterline_y_at_x0": round(h, 4),
            "trim_deg": round(math.degrees(trim), 3),
            "trim_note": "positive = bow up (trim by the stern)",
            "draft_moulded_x0": round(h, 4),
            "draft_max": round(float(drafts.max()), 4),
            "lcb": round(xB, 4), "kb": round(yB, 4), "lcf": round(lcf, 4),
            "waterplane_area": round(Awp, 4),
            "lwl": round(float(self.xc[cols].max() - self.xc[cols].min() + self.dx), 4),
            "bwl": round(float(2 * max(z for (_, z) in half) + self.dz), 4),
            "gm_t": round(yB + IT / V - G[1], 4),
            "gm_l": round(yB + IL / V - G[1], 4),
            "tpc_kg_per_cm": round(rho * Awp * 0.01, 2),
            "waterline_half_breadth": half,
        }

    def bonjean(self, spec):
        B = spec["hydrostatics"]["bonjean"]
        sdx = B["station_dx"]
        h0, h1, dh = B["heights"]
        hs = np.round(np.arange(h0, h1 + 1e-9, dh), 4)
        xa, xb = B.get("x_range", (-3.0, 3.0))
        stations = np.round(np.arange(xa + 0.5 * sdx, xb, sdx), 4)
        stb = self.zc > 0.0
        area, zc_, yc_ = [], [], []
        for xs in stations:
            ii = np.nonzero(np.abs(self.xc - xs) <= 0.5 * sdx)[0]
            ra, rz, ry = [], [], []
            for hh in hs:
                yw = np.full((len(ii), int(stb.sum())), hh)
                vc = self._at(self.cum[ii][:, stb], yw)
                vy = self._at(self.cumy[ii][:, stb], yw)
                a = float(vc.sum() * self.dz / len(ii))
                ra.append(round(a, 5))
                rz.append(round(float((vc * self.zc[stb][None, :]).sum() * self.dz / len(ii) / a), 4) if a > 1e-7 else 0.0)
                ry.append(round(float(vy.sum() * self.dz / len(ii) / a), 4) if a > 1e-7 else 0.0)
            area.append(ra)
            zc_.append(rz)
            yc_.append(ry)
        return {
            "note": "Starboard half only (port is the mirror). area_half[s][k] = immersed area (m^2) of the "
                    "starboard half-section in the slab around station_x[s] when the water stands at height "
                    "heights[k] in the vessel frame; zc_half / yc_half = that area's centroid. A strip model "
                    "sums both halves with the local water height on each side of each station.",
            "station_x": stations.tolist(), "station_dx": sdx, "heights": hs.tolist(),
            "area_half": area, "zc_half": zc_, "yc_half": yc_,
        }

def mass_condition(spec, cond):
    M = spec["mass"]
    items = M["budget"] + M["conditions"][cond]["extra"]
    m = sum(b["mass"] for b in items)
    c = [sum(b["mass"] * b["centroid"][i] for b in items) / m for i in range(3)]
    return m, c

def mass_check(spec):
    """The mass budget against the spec's own totals (dry, com_dry). Returns the dry CoM."""
    M = spec["mass"]
    tot = sum(b["mass"] for b in M["budget"])
    assert abs(tot - M["dry"]) < 1e-6, ("budget does not sum to dry", tot)
    m, com = mass_condition(spec, "lightship")
    assert all(abs(com[i] - M["com_dry"][i]) < 0.005 for i in range(3)), ("com_dry stale", [round(c, 4) for c in com])
    return com

def hydrostatics(spec, tris, spec_name="mariner_spec.json", datum="baseline"):
    """<boat>_hydro.json, schema threepp.usv_hydro/1 (what usv_rig.StripHull floats a boat on):
    every mass condition of the spec solved for heave and trim, and the Bonjean tables. Every
    boat's generator calls this one. tris: the buoyant closed meshes' triangles; spec_name:
    the spec file, for the record; datum: what y = 0 is in the spec's frame."""
    HS = spec["hydrostatics"]
    hy = Hydro(tris, HS["grid"])
    assert hy.closure < 1e-6, ("buoyant meshes are not closed", hy.closure)
    out = {"schema": "threepp.usv_hydro/1", "spec": spec_name, "rho": HS["rho"], "g": HS["g"],
           "frame": f"vessel frame of the spec (X forward, Y up, Z starboard, {datum} y = 0)", "conditions": {}}
    for cond in spec["mass"]["conditions"]:
        m, c = mass_condition(spec, cond)
        h, trim = hy.solve(m, c, HS["rho"])
        out["conditions"][cond] = hy.report(h, trim, m, c, HS["rho"], HS["g"])
    out["design_condition"] = spec["mass"]["design_condition"]
    out["bonjean"] = hy.bonjean(spec)
    return out

def check_gates(spec, hydro):
    """The design condition against the spec's draft, trim and GM_T gates."""
    HS, P = spec["hydrostatics"], spec["principal"]
    d = hydro["conditions"][spec["mass"]["design_condition"]]
    errs = []
    if abs(d["draft_max"] - P["draft_reference"]) > HS["draft_tolerance"]:
        errs.append(f"draft {d['draft_max']:.3f} vs {P['draft_reference']} +- {HS['draft_tolerance']}")
    if abs(d["trim_deg"]) > HS["trim_limit_deg"]:
        errs.append(f"trim {d['trim_deg']:.2f} deg")
    lo, hi = HS["gm_t_range"]
    if not lo <= d["gm_t"] <= hi:
        errs.append(f"GM_T {d['gm_t']:.3f} outside {lo}..{hi}")
    assert not errs, ("hydrostatic gates failed", errs)

# ---------------------------------------------------------------- Blender side
def C_MAT():
    """Blender (Z-up) -> vessel/glTF (Y-up): gl = (x, z, -y)."""
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)


def to_bl(M_gl):
    from mathutils import Matrix
    C = C_MAT()
    return Matrix((np.linalg.inv(C) @ M_gl @ C).tolist())


def make_materials(spec):
    import bpy
    M = spec["materials"]
    mats = {}
    for name in sorted(k for k in M if k != "note"):
        d = M[name]
        m = bpy.data.materials.new(name)
        try:
            m.use_nodes = True
        except Exception:
            pass
        m.use_backface_culling = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        c = d["color"]
        bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1.0)
        bsdf.inputs["Metallic"].default_value = d["metallic"]
        bsdf.inputs["Roughness"].default_value = d["roughness"]
        if "emissive" in d:
            e = d["emissive"]
            bsdf.inputs["Emission Color"].default_value = (e[0], e[1], e[2], 1.0)
            bsdf.inputs["Emission Strength"].default_value = d["emissive_strength"]
        mats[name] = m
    return mats


def box_uvs(verts, faces, tile):
    """Per-corner box-projected UVs in metres / tile, from each face's Newell normal."""
    uvs = []
    V = verts
    for f in faces:
        nx = ny = nz = 0.0
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        ax = max((abs(nx), 0), (abs(ny), 1), (abs(nz), 2))[1]
        for k in f:
            x, y, z = V[k]
            uvs += [(z, y), (x, z), (x, y)][ax]
    return [c / tile for c in uvs]


def new_object(name, part, mats, parent, M_gl, props=None, sharp_deg=30.0, tile=0.5):
    import bpy
    coll = bpy.context.scene.collection
    if part is None:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = 0.1
    else:
        used = sorted(set(part.mats))
        slot = {m: i for i, m in enumerate(used)}
        me = bpy.data.meshes.new(name)
        verts = [(x, -z, y) for (x, y, z) in part.verts]        # vessel frame -> Blender Z-up
        me.from_pydata(verts, [], part.faces)
        me.polygons.foreach_set("material_index", [slot[m] for m in part.mats])
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", box_uvs(part.verts, part.faces, tile))
        for m in used:
            me.materials.append(mats[m])
        me.validate(verbose=False)
        me.update()
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        me.set_sharp_from_angle(angle=math.radians(sharp_deg))
        me.update()
        obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    obj.matrix_basis = to_bl(M_gl)
    for k, v in (props or {}).items():
        obj[k] = v
    return obj

def export(path_out, **gltf):
    """Write the scene as a .glb (through a temporary file, so a failed export leaves the old
    one). gltf: further exporter options. Returns the triangle count."""
    import bpy
    bpy.context.view_layer.update()
    tris_n = 0
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            tris_n += sum(len(p.vertices) - 2 for p in ob.data.polygons)
    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    tmp = path_out + ".tmp.glb"
    bpy.ops.export_scene.gltf(filepath=tmp, export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True, **gltf)
    os.replace(tmp, path_out)
    return tris_n

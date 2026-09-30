"""Build the Mariner USV from mariner_spec.json: mariner.glb (via Blender) and mariner_hydro.json.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_mariner_blender.py -- --spec mariner_spec.json --out mariner.glb

Hydrostatics only (plain Python + numpy, no Blender; writes mariner_hydro.json):
    python build_mariner_blender.py --hydro-only

All geometry is built with numpy in the spec's vessel frame (X forward, Y up,
Z starboard, metres) and converted to Blender's Z-up only when a mesh or a
transform is handed to bpy. The glTF exporter's Y-up conversion maps it straight
back, so every number in the spec is the number in the .glb.

Node hierarchy (all names exact):
    mariner                      root, identity, origin = centreline x baseline x mid-LOA
      hull                       canoe body (bottom, walls, topsides, deck, transom),
                                 keel shoe, gondola, stern platform, intake and tunnel mouths
      collar                     D-section fender with its rub strip
      deck_fittings              hatch lids, foredeck ribs, bow eye, stern boxes, cleats,
                                 orange livery panels, vents
      mast                       legs, rungs, shelf, boxes, posts, roof, arm, radome,
                                 camera housing, antennas, GNSS, nav and flood lights
      jet                        waterjet body at the transom (origin on the jet axis)
        jet_steering             steering nozzle, turns about local +Y
        jet_reverse_bucket       bucket, lowers by turning about local +Z
      jet_thrust, bow_thruster   empties, identity rotation (thrust axes in the extras)
      camera_main, radar, gnss_fore, gnss_aft, ais_vhf_antenna, lte_antenna_1,
      lte_antenna_2, imu, transducer, moonpool          sensor empties
      nav_light_port, nav_light_stbd, allround_light, floodlight_port, floodlight_stbd

Sensor empties look down local -Z with local +Y up (threepp camera convention).

The hydrostatics integrate the buoyant closed meshes column by column: every
triangle adds its winding (+1 entering from below, -1 leaving) to the column it
covers, so the collar, the hull and the appendages overlap without being counted
twice. The generator solves the floating position for each loading condition and
asserts the design draft against the brochure's 500 mm.
"""
import argparse
import json
import math
import os
import sys

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


# ---------------------------------------------------------------- hull lines
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


class Lines:
    """The canoe body's defining curves as functions of x.

    The bow is faceted, as on the real boat: the flat side ends at a chevron (its
    forward edges run from the tip up to the deck edge and down to the collar top),
    and ahead of it an upper and a lower facet meet along a crease that climbs from
    the tip to the nose. The orange livery covers exactly the flat side."""

    def __init__(self, spec):
        H, C = spec["hull"], spec["collar"]
        self.H, self.C = H, C
        kp = np.asarray(H["keel_profile"], float)
        self._keel = pchip(kp[:, 0], kp[:, 1])
        self.x_t, self.x_n, self.x_f = H["transom_x"], H["nose_x"], H["taper_start_x"]
        self.x_df = H["deck_taper_start_x"]
        self.x_tip, self.y_tip = H["chevron_tip"]
        self.y_ct0 = C["centre_y"] + C["b"]                  # collar top amidships
        self.y_top0 = H["deck_y"] - H["gunwale_round"]       # topside top amidships

    def t_bow(self, x):
        return min(1.0, max(0.0, (x - self.x_f) / (self.x_n - self.x_f)))

    def t_deck(self, x):
        return min(1.0, max(0.0, (x - self.x_df) / (self.x_n - self.x_df)))

    def flat_z(self, y):
        """|z| of the flat side (with its tumblehome) at height y."""
        s = min(1.0, max(0.0, (y - self.y_ct0) / (self.y_top0 - self.y_ct0)))
        return self.H["wall_halfwidth"] - self.H["tumblehome"] * s

    def topside(self, x):
        """Topside points (z, y) from the collar top up to the gunwale: collar top,
        lower chevron point, crease, upper chevron point, top of the side."""
        H = self.H
        Bn = H["nose_halfwidth"]
        p0 = (self.wall(x), self.collar_top(x))
        if x <= self.x_tip:
            p2 = (self.flat_z(self.y_tip), self.y_tip)
        else:
            s = (x - self.x_tip) / (self.x_n - self.x_tip)
            p2 = (self.flat_z(self.y_tip) + (Bn - self.flat_z(self.y_tip)) * s,
                  self.y_tip + (H["crease_nose_y"] - self.y_tip) * s)
        z_top0 = self.flat_z(self.y_top0)
        p4 = (z_top0 + (Bn - 0.01 - z_top0) * self.t_deck(x), self.deck(x) - H["gunwale_round"])

        def mid(p, q):
            y = 0.5 * (p[1] + q[1])
            return (0.5 * (p[0] + q[0]), y)
        if self.x_f <= x <= self.x_tip:
            y = self.y_ct0 + (x - self.x_f) / (self.x_tip - self.x_f) * (self.y_tip - self.y_ct0)
            p1 = (self.flat_z(y), y)
        else:
            p1 = mid(p0, p2)
        if self.x_df <= x <= self.x_tip:
            y = self.y_top0 + (x - self.x_df) / (self.x_tip - self.x_df) * (self.y_tip - self.y_top0)
            p3 = (self.flat_z(y), y)
        else:
            p3 = mid(p2, p4)
        return [p0, p1, p2, p3, p4]

    def keel(self, x):
        return self._keel(x)

    def wall(self, x):
        H = self.H
        return H["wall_halfwidth"] + (H["nose_halfwidth"] - H["wall_halfwidth"]) * self.t_bow(x)

    def collar_centre(self, x):
        C = self.C
        return C["centre_y"] + C["nose_rise"] * self.t_bow(x) ** C["rise_exponent"]

    def chine(self, x):
        return self.collar_centre(x) - self.C["b"]

    def collar_top(self, x):
        return self.collar_centre(x) + self.C["b"]

    def deck(self, x):
        H = self.H
        return H["deck_y"] + (H["deck_nose_y"] - H["deck_y"]) * self.t_deck(x)

    def deck_edge_z(self, x):
        return self.topside(x)[-1][0] - self.H["gunwale_round"]

    def deck_at(self, x, z):
        """Deck surface height (crowned) at (x, z)."""
        zd = self.deck_edge_z(x)
        q = min(1.0, abs(z) / zd)
        return self.deck(x) + self.H["deck_crown"] * (1.0 - q * q)

    def topside_z(self, x, y):
        """|z| of the topside surface at (x, y), between the collar top and the gunwale."""
        P = self.topside(x)
        ys = [p[1] for p in P]
        if y <= ys[0]:
            return P[0][0]
        for (za, ya), (zb, yb) in zip(P[:-1], P[1:]):
            if y <= yb:
                return za if yb - ya < 1e-9 else za + (zb - za) * (y - ya) / (yb - ya)
        return P[-1][0]

    def rail_width(self, x, rail):
        w = rail["width"]
        ramp = 0.15
        w *= min(1.0, max(0.0, (rail["x_end"] - x) / ramp))
        if "gap_x" in rail:
            g = abs(x - rail["gap_x"]) - rail["gap_half_length"]
            w *= min(1.0, max(0.0, g / 0.02))
        return max(w, 0.001)

    def bottom_slope(self, x):
        return (self.chine(x) - self.keel(x)) / self.wall(x)

    def section_half(self, x):
        """Starboard half-section from the keel (z = 0) to the deck centre (z = 0), as
        [(z, y, kind)], kind = the material of the segment that starts at this point."""
        H = self.H
        B, yk, yc = self.wall(x), self.keel(x), self.chine(x)
        s = (yc - yk) / B
        pts = []

        def on_chord(fr):
            return (fr * B, yk + fr * (yc - yk))
        rails = sorted(H["spray_rails"], key=lambda r: r["chord_fraction"])
        pts.append((*on_chord(0.0), "bottom"))
        pts.append((*on_chord(0.2), "bottom"))
        for ri, rail in enumerate(rails):
            fr = rail["chord_fraction"]
            az, ay = on_chord(fr)
            w = self.rail_width(x, rail)
            pts.append((az, ay, "bottom"))              # start of the rail's flat underside
            pts.append((az + w, ay, "bottom"))          # outer edge of the ledge (faces down)
            pts.append((az + w, ay + w * s, "bottom"))  # back on the chord
            nxt = 0.6 if ri == 0 else 0.9
            pts.append((*on_chord(nxt), "bottom"))
        pts.append((B, yc, "wall"))                     # chine; the collar covers the wall
        P = self.topside(x)
        for (z, y) in P[:-1]:
            pts.append((z, y, "topside"))
        r, yd = H["gunwale_round"], self.deck(x)
        cz, cy = P[-1][0] - r, yd - r
        for k, ang in enumerate((0.0, 30.0, 60.0, 90.0)):
            a = math.radians(ang)
            pts.append((cz + r * math.cos(a), cy + r * math.sin(a), "deck"))
        zd = cz
        for q in (0.66, 0.33):
            z = zd * q
            pts.append((z, yd + H["deck_crown"] * (1.0 - q * q), "deck"))
        pts.append((0.0, yd + H["deck_crown"], "deck"))
        return pts

    def ring(self, x):
        """Closed loop in 3D: starboard half keel -> deck centre, then port half back."""
        S = self.section_half(x)
        stb = [(x, y, z) for (z, y, _) in S]
        port = [(x, y, -z) for (z, y, _) in reversed(S[1:-1])]
        # Segment k runs from ring point k to k+1. The port half runs from the deck centre
        # back down to the keel, so its segment from mirror(S[i]) to mirror(S[i-1]) is the
        # mirror of starboard segment i-1 and takes that segment's kind.
        Sk = [k for (_, _, k) in S]
        port_kinds = [Sk[i - 1] for i in range(len(S) - 1, 0, -1)]
        return stb + port, Sk[:-1] + port_kinds


# ---------------------------------------------------------------- the hull
MAT_OF_KIND = {"bottom": "hull_pe_bottom", "wall": "hull_pe_bottom",
               "topside": "topside_black", "deck": "deck_black"}


def hull_stations(spec):
    H, L = spec["hull"], spec["hull"]["stations"]
    x0, x1 = H["transom_x"], H["nose_x"]
    xs = set(np.round(np.arange(x0, x1, L["dx"]), 6).tolist())
    for (a, b) in L["fine_ranges"]:
        xs |= set(np.round(np.arange(a, b, L["dx_fine"]), 6).tolist())
    xs |= {x0, x1, H["taper_start_x"], H["deck_taper_start_x"], H["chevron_tip"][0]}
    for rail in H["spray_rails"]:
        xs |= {rail["x_end"] - 0.15, rail["x_end"]}
        if "gap_x" in rail:
            g, hl = rail["gap_x"], rail["gap_half_length"]
            xs |= {g - hl - 0.02, g - hl, g + hl, g + hl + 0.02}
    # round, or 2.45 - 0.15 and 2.2 + 5 * 0.02 become two stations 4e-16 apart
    return sorted(set(round(x, 6) for x in xs if x0 <= x <= x1))


def build_hull(spec, L, part):
    xs = hull_stations(spec)
    rings, kinds = [], None
    for x in xs:
        r, k = L.ring(x)
        rings.append(r)
        kinds = k
    v, f = loft(rings)
    N = len(rings[0])
    mats = [MAT_OF_KIND[kinds[q % N]] for q in range(len(f) - 2)] + ["topside_black", "collar_pe_black"]
    part.add_multi(v, f, mats, buoyant=True)
    return xs


def build_appendages(spec, L, part):
    A = spec["appendages"]
    ks = A["keel_shoe"]
    # keel shoe + stem fin: a band under the keel curve (offset along its normal), then the
    # fin's own outer profile up the stem; the inner edge is buried 2 cm in the hull
    inner, outer = [], []
    for x in np.linspace(ks["x_aft"], L.x_n, 60):
        dy = (L.keel(x + 1e-3) - L.keel(x - 1e-3)) / 2e-3
        nrm = np.array([dy, -1.0])
        nrm /= np.linalg.norm(nrm)
        p = np.array([x, L.keel(x)])
        inner.append(tuple(p - 0.02 * nrm))
        if x <= ks["stem_from_x"]:
            t = min(1.0, max(0.0, (L.keel(x) - 0.02) / 0.4))
            outer.append(tuple(p + (ks["depth"] + (ks["stem_depth"] - ks["depth"]) * t) * nrm))
    outer += [tuple(q) for q in ks["stem_profile"]]
    inner.append((L.x_n - 0.02, ks["top_y"]))
    poly = outer + list(reversed(inner))
    hw = 0.5 * ks["thickness"]
    part.add(*extrude_xy(poly, -hw, hw), "collar_pe_black", buoyant=True)
    # gondola: wedge with a flat, faired into the bottom aft
    g = A["gondola"]
    poly = [(g["x_fwd"], 0.06), (g["x_fwd"], -g["depth"]), (g["x_flat_aft"], -g["depth"]), (g["x_aft"], 0.06)]
    part.add(*extrude_xy(poly, -0.5 * g["width"], 0.5 * g["width"]), "collar_pe_black", buoyant=True)
    # stern platform between the collar tails, over the jet
    sp = spec["deck"]["stern_platform"]
    B = spec["hull"]["wall_halfwidth"]
    v, f, M = box_between((sp["x_range"][0], sp["y_range"][0], -B), (sp["x_range"][1] + 0.01, sp["y_range"][1], B))
    part.add(v, f, "topside_black", M, buoyant=True)


def v_bottom_point(L, x, y, side):
    """Point on the straight V bottom at height y (between keel and first rail) and its outward normal."""
    s = L.bottom_slope(x)
    z = (y - L.keel(x)) / s
    n = np.array([0.0, -1.0, side * s])
    return np.array([x, y, side * z]), n / np.linalg.norm(n)


def build_bottom_openings(spec, L, part):
    """Bow-thruster tunnel mouths and the jet intake, as thin dark decals on the V bottom."""
    bt = spec["propulsion"]["bow_thruster"]
    cx, cy, _ = bt["position"]
    r = bt["tunnel_radius"]
    n = 32
    for side in (1, -1):
        p0, nrm0 = v_bottom_point(L, cx, cy, side)
        # the tunnel's dark mouth (a fan), then the steel lip round it
        v = [tuple(p0 + 0.002 * nrm0)]
        for i in range(n):
            a = 2 * math.pi * i / n
            p, nrm = v_bottom_point(L, cx + r * math.cos(a), cy + r * math.sin(a), side)
            v.append(tuple(p + 0.002 * nrm))
        part.add(v, [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)], "tunnel_dark", closed=False, outward=nrm0)
        v, f = [], []
        for i in range(n):
            a = 2 * math.pi * i / n
            for rr in (r, r + 0.012):
                p, nrm = v_bottom_point(L, cx + rr * math.cos(a), cy + rr * math.sin(a), side)
                v.append(tuple(p + 0.0025 * nrm))
        for i in range(n):
            j = (i + 1) % n
            f.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
        part.add(v, f, "steel", closed=False, outward=nrm0)
    it = spec["propulsion"]["waterjet"]["intake"]
    for side in (1, -1):
        v = []
        for (x, z) in ((it["x_range"][0], 0.005), (it["x_range"][1], 0.005),
                       (it["x_range"][1], it["halfwidth"]), (it["x_range"][0], it["halfwidth"])):
            y = L.keel(x) + z * L.bottom_slope(x)
            p, nrm = v_bottom_point(L, x, y, side)
            v.append(tuple(p + 0.003 * nrm))
        part.add(v, [(0, 1, 2, 3)], "tunnel_dark", closed=False, outward=nrm)


# ---------------------------------------------------------------- the collar
def collar_path(spec, L):
    """Plan path of the collar's inner (wall) line, starboard tail -> nose -> port tail,
    as [(x, z)], plus the fore-aft stations used for the tail taper."""
    C, H = spec["collar"], spec["hull"]
    B, Bn = H["wall_halfwidth"], H["nose_halfwidth"]
    tail = list(np.linspace(C["tail_x"], C["tail_start_x"], 9))
    side = tail + [C["tail_start_x"] + 0.1, C["rub_strip"]["x_aft"], H["transom_x"], 0.0, H["taper_start_x"]]
    side = sorted(set(round(x, 6) for x in side))
    stb = [(x, B) for x in side]
    nf = C["facet_segments"]
    for k in range(1, nf + 1):
        t = k / nf
        stb.append((H["taper_start_x"] + t * (H["nose_x"] - H["taper_start_x"]), B + t * (Bn - B)))
    nose = [(H["nose_x"], Bn * q) for q in (0.5, 0.0, -0.5)]
    port = [(x, -z) for (x, z) in reversed(stb)]
    return stb + nose + port


def path_normals(path):
    """Outward horizontal normals at path vertices, mitred (length 1/cos(half-turn))."""
    P = np.asarray(path, float)
    segn = []
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        d /= np.linalg.norm(d)
        segn.append(np.array([-d[1], d[0]]))      # (x, z): rotate d so +X on the stb side -> +Z
    out = []
    for i in range(len(P)):
        if i == 0:
            out.append(segn[0])
        elif i == len(P) - 1:
            out.append(segn[-1])
        else:
            s = segn[i - 1] + segn[i]
            out.append(2.0 * s / float(np.dot(s, s)))
    return out


def d_section(a, b, p, embed, k=20):
    """Closed D section (n outward, u up): the superellipse bulge plus the flat inner side."""
    pts = [(-embed, -b)]
    for i in range(k + 1):
        phi = -0.5 * math.pi + math.pi * i / k
        c, s = math.cos(phi), math.sin(phi)
        pts.append((a * abs(c) ** (2.0 / p), b * math.copysign(abs(s) ** (2.0 / p), s)))
    pts.append((-embed, b))
    return pts


def collar_rings(spec, L, path, section_fn):
    C = spec["collar"]
    Nv = path_normals(path)
    rings = []
    for (x, z), N in zip(path, Nv):
        bx = C["b"]
        if x < C["tail_start_x"]:
            q = (C["tail_start_x"] - x) / (C["tail_start_x"] - C["tail_x"])
            bx = C["tail_min_b"] + (C["b"] - C["tail_min_b"]) * math.sqrt(max(0.0, 1.0 - q * q))
        yc = L.chine(x) + bx if x < C["tail_start_x"] else L.collar_centre(x)
        ring = []
        for (n, u) in section_fn(bx):
            ring.append((x + n * N[0], yc + u, z + n * N[1]))
        rings.append(ring)
    return rings


def build_collar(spec, L, part):
    C = spec["collar"]
    path = collar_path(spec, L)
    rings = collar_rings(spec, L, path, lambda bx: [(n, u * bx / C["b"]) for (n, u) in
                                                   d_section(C["a"], C["b"], C["superellipse"], C["embed"])])
    part.add(*loft(rings), "collar_pe_black", buoyant=True)
    rs = C["rub_strip"]
    uu = rs["u"]
    ns = C["a"] * (1.0 - abs(uu / C["b"]) ** C["superellipse"]) ** (1.0 / C["superellipse"])
    sect = [(ns - 0.01, uu - 0.5 * rs["height"]), (ns + rs["proud"], uu - 0.5 * rs["height"]),
            (ns + rs["proud"], uu + 0.5 * rs["height"]), (ns - 0.01, uu + 0.5 * rs["height"])]
    rpath = [(x, z) for (x, z) in path if x >= rs["x_aft"] - 1e-9]
    rings = collar_rings(spec, L, rpath, lambda bx: sect)
    part.add(*loft(rings), "collar_pe_black")


# ---------------------------------------------------------------- deck fittings, livery
def clip_slab(poly, x0, x1):
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


def side_decal(part, L, poly, mat, off=0.004, extra_x=()):
    """Paint a side-view polygon (x, y) onto both topsides, following the facet corner."""
    xs = sorted(set([p[0] for p in poly] + list(extra_x)))
    for side in (1, -1):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            piece = clip_slab(poly, x0, x1)
            if len(piece) < 3:
                continue
            v = [(x, y, side * (L.topside_z(x, y) + off)) for (x, y) in piece]
            f = [tuple(range(len(v)))]
            part.add(v, f, mat, closed=False, outward=(0.0, 0.0, side))


def deck_strip(part, L, z, x0, x1, width, height, mat, n=12):
    """A rib that follows the (sloping, crowned) deck."""
    rings = []
    for x in np.linspace(x0, x1, n):
        y = L.deck_at(x, z)
        rings.append([(x, y - 0.01, z - 0.5 * width), (x, y - 0.01, z + 0.5 * width),
                      (x, y + height, z + 0.5 * width), (x, y + height, z - 0.5 * width)])
    part.add(*loft(rings), mat)


def build_deck_fittings(spec, L, part):
    D, LV = spec["deck"], spec["livery"]
    ht = D["hatches"]
    xsp = ht["x_splits"]
    for x0, x1 in zip(xsp[:-1], xsp[1:]):
        x0, x1 = x0 + 0.5 * ht["gap"], x1 - 0.5 * ht["gap"]
        for side in (1, -1):
            za, zb = 0.5 * ht["centre_gap"], ht["halfwidth"]
            zz = np.linspace(za, zb, 8)
            bot = [(L.deck_at(x0, z) - 0.01, z) for z in zz]
            top = [(L.deck_at(x0, z) + ht["thickness"], z) for z in zz]
            sect = bot + list(reversed(top))
            rings = [[(x, y, side * z) for (y, z) in sect] for x in (x0, x1)]
            part.add(*loft(rings), "hatch_grey")
            for k in range(ht["ribs"]):
                zc = za + (k + 0.5) * (zb - za) / ht["ribs"]
                y = L.deck_at(x0, zc) + ht["thickness"]
                v, f, M = box_between((x0 + 0.03, y - 0.002, side * zc - 0.5 * ht["rib_width"]),
                                      (x1 - 0.03, y + ht["rib_height"], side * zc + 0.5 * ht["rib_width"]))
                part.add(v, f, "hatch_grey", M)
    fr = D["foredeck_ribs"]
    for z in fr["z"]:
        x_end = fr["x_range"][1]
        while L.deck_edge_z(x_end) - 0.05 < z and x_end > fr["x_range"][0] + 0.1:
            x_end -= 0.01
        for side in (1, -1):
            deck_strip(part, L, side * z, fr["x_range"][0], x_end, fr["width"], fr["height"], "hatch_grey")
    be = D["bow_eye"]
    hw = 0.5 * be["thickness"]
    part.add(*extrude_xy([tuple(p) for p in be["profile_xy"]], -hw, hw), "collar_pe_black")
    hx, hy, hr = be["hole"]
    v, f = cylinder(hr, -hw - 0.002, hw + 0.002, 24)
    part.add(v, f, "tunnel_dark", T(hx, hy, 0.0))
    v, f = cylinder(0.012, -0.06, 0.06, 16)
    part.add(v, f, "steel", T(hx, hy - 0.01, 0.0))
    sb = D["stern_boxes"]
    for side in (1, -1):
        za, zb = sorted((side * sb["z_range"][0], side * sb["z_range"][1]))
        v, f, M = box_between((sb["x_range"][0], D["stern_platform"]["y_range"][1] - 0.005, za),
                              (sb["x_range"][1], sb["y_top"], zb))
        part.add(v, f, "topside_black", M)
    for (x, y, z) in D["cleats"]:
        v, f, M = box_between((x - 0.09, y, z - 0.015), (x + 0.09, y + 0.025, z + 0.015))
        part.add(v, f, "steel", M)
        for dx in (-0.05, 0.05):
            v, f, M = box_between((x + dx - 0.012, y - 0.005, z - 0.012), (x + dx + 0.012, y + 0.012, z + 0.012))
            part.add(v, f, "steel", M)
    # livery: the orange panel over the black topsides, then the vents in the notch
    fx = [L.x_f + k * 0.05 for k in range(1, 10)]
    side_decal(part, L, [tuple(p) for p in LV["orange_polygon_xy"]], "livery_orange", extra_x=[L.x_f] + fx)
    vs = LV["vent_slots"]
    for k in range(vs["count"]):
        xc = vs["x0"] - k * vs["pitch"]
        w, h = vs["size_xy"]
        side_decal(part, L, [(xc - 0.5 * w, vs["y"] - 0.5 * h), (xc + 0.5 * w, vs["y"] - 0.5 * h),
                             (xc + 0.5 * w, vs["y"] + 0.5 * h), (xc - 0.5 * w, vs["y"] + 0.5 * h)],
                   "vent_grey", off=0.006)
    lx, ly = LV["status_led"]
    side_decal(part, L, [(lx - 0.012, ly - 0.012), (lx + 0.012, ly - 0.012), (lx + 0.012, ly + 0.012),
                         (lx - 0.012, ly + 0.012)], "led_white", off=0.006)
    kx, ky, kr = LV["kill_switch"]
    for side in (1, -1):
        zs = L.topside_z(kx, ky)
        v, f = cylinder(kr, 0.0, 0.012, 20)
        part.add(v, f, "steel", T(kx, ky, side * zs) @ (np.eye(4) if side > 0 else R("Y", 180.0)))


# ---------------------------------------------------------------- the mast
def build_mast(spec, part):
    MA, SE, LI = spec["mast"], spec["sensors"], spec["lights"]
    lg = MA["legs"]
    for side in (1, -1):
        legs = []
        for key in ("fore", "aft"):
            p0 = np.array(lg[key]["deck"], float) * [1, 1, side]
            p1 = np.array(lg[key]["shelf"], float) * [1, 1, side]
            strut(part, p0, p1 + [0, 0.02, 0], lg["radius"], "mast_powder_black")
            fx, fy, fz = lg["foot_plate"]
            v, f, M = box_between(p0 - [0.5 * fx, 0.0, 0.5 * fz], p0 + [0.5 * fx, fy, 0.5 * fz])
            part.add(v, f, "mast_powder_black", M)
            legs.append((p0, p1))
        pts = []
        for (p0, p1) in legs:
            t = (lg["rung_y"] - p0[1]) / (p1[1] - p0[1])
            pts.append(p0 + t * (p1 - p0))
        strut(part, pts[0], pts[1], 0.022, "mast_powder_black", 12)
    sh = MA["shelf"]
    v, f, M = box_between((sh["x_range"][0], sh["y_range"][0], -sh["halfwidth"]),
                          (sh["x_range"][1], sh["y_range"][1], sh["halfwidth"]))
    part.add(v, f, "mast_powder_black", M)
    ub = MA["under_box"]
    v, f, M = box_between((ub["x_range"][0], ub["y_range"][0], -ub["halfwidth"]),
                          (ub["x_range"][1], ub["y_range"][1] + 0.005, ub["halfwidth"]))
    part.add(v, f, "mast_powder_black", M)
    ab = MA["aux_box"]
    v, f, M = box_between((ab["x_range"][0], ab["y_range"][0], -ab["halfwidth"]),
                          (ab["x_range"][1] + 0.005, ab["y_range"][1] + 0.01, ab["halfwidth"]))
    part.add(v, f, "livery_orange", M)
    fl = MA["floodlights"]
    for side in (1, -1):
        za, zb = sorted((side * fl["z_range"][0], side * fl["z_range"][1]))
        v, f, M = box_between((fl["x_range"][0], fl["y_range"][0], za), (fl["x_range"][1], fl["y_range"][1] + 0.003, zb))
        part.add(v, f, "floodlight_lens", M)
    po = MA["posts"]
    ro = MA["roof"]
    for x in po["x"]:
        for side in (1, -1):
            v, f = cylinder(po["radius"], sh["y_range"][1] - 0.005, ro["y_range"][0] + 0.005, 16)
            part.add(v, f, "mast_powder_black", T(x, 0.0, side * po["z"]) @ Z_TO_Y())
    v, f, M = box_between((ro["x_range"][0], ro["y_range"][0], -ro["halfwidth"]),
                          (ro["x_range"][1], ro["y_range"][1], ro["halfwidth"]))
    part.add(v, f, "mast_powder_black", M)
    # low pyramid cap
    cx = 0.5 * (ro["x_range"][0] + ro["x_range"][1])
    bx, bz = ro["cap_size_xz"]
    y0, y1 = ro["y_range"][1] - 0.002, ro["y_range"][1] + ro["cap_height"]
    tx, tz = 0.35 * bx, 0.3 * bz
    v = [(cx - bx / 2, y0, -bz / 2), (cx + bx / 2, y0, -bz / 2), (cx + bx / 2, y0, bz / 2), (cx - bx / 2, y0, bz / 2),
         (cx - tx / 2, y1, -tz / 2), (cx + tx / 2, y1, -tz / 2), (cx + tx / 2, y1, tz / 2), (cx - tx / 2, y1, tz / 2)]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    part.add(v, f, "mast_powder_black")
    ar = MA["arm"]
    v, f, M = box_between((ar["x_range"][0], ar["y_range"][0], -ar["halfwidth"]),
                          (ar["x_range"][1], ar["y_range"][1], ar["halfwidth"]))
    part.add(v, f, "mast_powder_black", M)
    # radome: blue base band, grey body, low dome
    rd = MA["radome"]
    rx, rz = rd["centre_xz"]
    ya, yb, rr = rd["y_range"][0], rd["y_range"][1], rd["radius"]
    yband = ya + rd["band_height"]
    v, f = cylinder(rr + 0.001, ya, yband, 64)
    part.add(v, f, "radome_band_blue", T(rx, 0.0, rz) @ Z_TO_Y())
    prof = [(rr, yband - 0.001), (rr, yb - 0.04), (rr - 0.006, yb - 0.022), (rr - 0.03, yb - 0.008),
            (rr - 0.08, yb - 0.001), (0.0, yb)]
    v, f = revolve(prof, 64)
    part.add(v, f, "radome_grey", T(rx, 0.0, rz) @ Z_TO_Y())
    # GNSS antennas on the arm ends
    for name in ("gnss_fore", "gnss_aft"):
        g = SE[name]
        gx, gy, gz = g["position"]
        top = gy
        ybase = top - g["disc_height"]
        v, f = cylinder(0.012, ar["y_range"][1] - 0.003, ybase + 0.002, 12)
        part.add(v, f, "steel", T(gx, 0.0, gz) @ Z_TO_Y())
        R0 = g["disc_radius"]
        prof = [(0.0, ybase), (R0, ybase), (R0, ybase + 0.01), (0.9 * R0, ybase + 0.03),
                (0.65 * R0, ybase + 0.047), (0.0, top)]
        v, f = revolve(prof, 32)
        part.add(v, f, "gnss_cream", T(gx, 0.0, gz) @ Z_TO_Y())
    # main camera on its pole
    cam = SE["camera_main"]
    hc, hs = np.array(cam["housing"]["centre"], float), cam["housing"]["size"]
    ybot = hc[1] - 0.5 * hs[1]
    v, f = frustum(0.05, cam["housing"]["pole_radius"], sh["y_range"][1] - 0.002, sh["y_range"][1] + 0.08, 20)
    part.add(v, f, "camera_white", T(hc[0], 0.0, hc[2]) @ Z_TO_Y())
    v, f = cylinder(cam["housing"]["pole_radius"], sh["y_range"][1] + 0.07, ybot + 0.002, 20)
    part.add(v, f, "camera_white", T(hc[0], 0.0, hc[2]) @ Z_TO_Y())
    v, f = box(*hs)
    part.add(v, f, "camera_white", T(*hc))
    lp = np.array(cam["position"], float)
    v, f, M = box_between((hc[0] + 0.5 * hs[0] - 0.002, lp[1] - 0.018, hc[2] - 0.045),
                          (hc[0] + 0.5 * hs[0] + 0.004, lp[1] + 0.018, hc[2] + 0.045))
    part.add(v, f, "camera_window", M)
    v, f = cylinder(0.014, 0.0, 0.004, 24)
    part.add(v, f, "lens_glass", T(hc[0] + 0.5 * hs[0] + 0.003, lp[1], lp[2]) @ Z_TO_X())
    # antennas
    av = SE["ais_vhf_antenna"]
    bxa, bya, bza = av["base"]
    v, f = cylinder(0.02, bya - 0.002, bya + 0.1, 16)
    part.add(v, f, "antenna_white", T(bxa, 0.0, bza) @ Z_TO_Y())
    v, f = revolve([(0.0, bya + 0.099), (av["radius"], bya + 0.1), (0.5 * av["radius"], bya + av["length"]),
                    (0.0, bya + av["length"] + 0.004)], 8)
    part.add(v, f, "antenna_white", T(bxa, 0.0, bza) @ Z_TO_Y())
    l1 = SE["lte_antenna_1"]
    bx1, by1, bz1 = l1["base"]
    v, f = revolve([(0.0, by1 - 0.002), (l1["radius"], by1 - 0.002), (l1["radius"], by1 + l1["length"] - 0.01),
                    (0.6 * l1["radius"], by1 + l1["length"] - 0.002), (0.0, by1 + l1["length"])], 16)
    part.add(v, f, "antenna_white", T(bx1, 0.0, bz1) @ Z_TO_Y())
    l2 = SE["lte_antenna_2"]
    x2, y2, z2 = l2["position"]
    v, f = revolve([(0.0, y2 - 0.002), (l2["radius"], y2 - 0.002), (l2["radius"] * 0.9, y2 + 0.01),
                    (l2["radius"] * 0.55, y2 + 0.019), (0.0, y2 + l2["height"])], 24)
    part.add(v, f, "mast_powder_black", T(x2, 0.0, z2) @ Z_TO_Y())
    # lights
    for name, mat in (("nav_light_stbd", "nav_green"), ("nav_light_port", "nav_red")):
        lx, ly, lz = LI[name]["position"]
        side = 1 if lz > 0 else -1
        v, f, M = box_between((lx - 0.035, ly - 0.03, lz - side * 0.03), (lx + 0.035, ly + 0.03, lz + side * 0.005))
        part.add(v, f, "mast_powder_black", M)
        v, f, M = box_between((lx - 0.028, ly - 0.022, lz + side * 0.004), (lx + 0.03, ly + 0.022, lz + side * 0.014))
        part.add(v, f, mat, M)
    ax, ay, az = LI["allround_light"]["position"]
    v, f = cylinder(0.02, y1 - 0.002, ay + 0.03, 20)
    part.add(v, f, "nav_white", T(ax, 0.0, az) @ Z_TO_Y())
    v, f = cylinder(0.024, ay + 0.03, ay + 0.04, 20)
    part.add(v, f, "mast_powder_black", T(ax, 0.0, az) @ Z_TO_Y())


# ---------------------------------------------------------------- the waterjet
def build_jet(spec):
    """Three parts in their own node frames: body (jet), nozzle (jet_steering), bucket."""
    J = spec["propulsion"]["waterjet"]
    ox, oy = J["flange"]["x"], J["axis_y"]
    body, steer, bucket = Part(), Part(), Part()
    about_x = Z_TO_X()                                   # revolve about local +X

    def lx(x):
        return x - ox
    fl = J["flange"]
    v, f = cylinder(fl["radius"], lx(fl["x"]) - fl["thickness"], lx(fl["x"]) + 0.004, 32)
    body.add(v, f, "jet_grey", about_x)
    b0, b1 = J["body"]["x_range"]
    r0, r1 = J["body"]["radius"]
    v, f = frustum(r0, r1, lx(b0), lx(b1) - fl["thickness"] + 0.002, 32)
    body.add(v, f, "jet_grey", about_x)
    t0, t1 = J["tailpipe"]["x_range"]
    q0, q1 = J["tailpipe"]["radius"]
    v, f = revolve([(0.0, lx(t0) - 0.001), (q0, lx(t0)), (q1, lx(t1)), (r0 * 0.98, lx(t1) + 0.02), (0.0, lx(t1) + 0.02)], 32)
    body.add(v, f, "jet_grey", about_x)
    for side in (1, -1):                                # steering rams (fixed, cosmetic)
        strut(body, (lx(b0) + 0.08, 0.02, side * 0.14), (lx(J["steering"]["pivot_x"]) + 0.03, 0.0, side * 0.1),
              0.014, "steel", 12)
    st = J["steering"]
    s0, s1 = st["radius"]
    v, f = revolve([(0.0, 0.0), (s0, 0.0), (s1, -st["x_len"]), (0.0, -st["x_len"])], 32)
    steer.add(v, f, "jet_grey", about_x)
    for side in (1, -1):
        v, f, M = box_between((-0.06, -0.02, side * s0 - 0.012), (-0.01, 0.02, side * (s0 + 0.03)))
        steer.add(v, f, "jet_grey", M)
    rb = J["reverse_bucket"]
    px, py = rb["pivot"]
    # Bucket frame: origin at the hinge. The bowl is authored LOWERED (a cylinder sector
    # wrapped round the back of the nozzle exit) and then turned back up by the full
    # travel, so node rotation 0 is the raised rest pose and +limit lowers it again.
    ex = J["steering"]["pivot_x"] - st["x_len"]
    c = np.array([ex + rb["lowered_centre_dx"] - px, oy - py])
    a0, a1 = [math.radians(a) for a in rb["lowered_arc_deg"]]
    hw = 0.5 * rb["width"]
    up = R("Z", -rb["limit_deg"][1])
    rings = []
    for k in range(13):
        a = a0 + (a1 - a0) * k / 12
        d = np.array([math.cos(a), math.sin(a)])
        pi, po = c + rb["radius"] * d, c + (rb["radius"] + 0.01) * d
        rings.append([(pi[0], pi[1], -hw), (po[0], po[1], -hw), (po[0], po[1], hw), (pi[0], pi[1], hw)])
    bucket.add(*loft(rings), "jet_bucket_orange", up)
    top = c + (rb["radius"] + 0.005) * np.array([math.cos(a0), math.sin(a0)])
    top = up[:2, :2] @ top                              # the bowl's top edge, at rest
    for side in (1, -1):                                # arms from the hinge to the bowl's top edge
        strut(bucket, (0.0, 0.0, side * (hw - 0.01)), (top[0], top[1], side * (hw - 0.01)), 0.012, "jet_bucket_orange", 12)
    v, f = cylinder(0.016, -hw - 0.015, hw + 0.015, 16)
    bucket.add(v, f, "steel")
    return body, steer, bucket


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


def check_spec(spec):
    M = spec["mass"]
    tot = sum(b["mass"] for b in M["budget"])
    assert abs(tot - M["dry"]) < 1e-9, tot
    m, com = mass_condition(spec, "lightship")
    assert all(abs(com[i] - M["com_dry"][i]) < 0.005 for i in range(3)), com
    C, H = spec["collar"], spec["hull"]
    assert abs(2 * (H["wall_halfwidth"] + C["a"]) - spec["principal"]["beam"]) < 1e-6
    L = Lines(spec)
    for x in np.linspace(H["transom_x"], H["nose_x"], 200):
        assert L.keel(x) < L.chine(x) - 0.05, ("keel above chine", x)
        ys = [p[1] for p in L.topside(x)]
        assert all(b >= a - 1e-9 for a, b in zip(ys[:-1], ys[1:])), ("topside not rising", x, ys)
    return com


def hydrostatics(spec, tris):
    HS = spec["hydrostatics"]
    hy = Hydro(tris, HS["grid"])
    assert hy.closure < 1e-6, ("buoyant meshes are not closed", hy.closure)
    out = {"schema": "threepp.usv_hydro/1", "spec": "mariner_spec.json", "rho": HS["rho"], "g": HS["g"],
           "frame": "vessel frame of the spec (X forward, Y up, Z starboard, baseline y = 0)", "conditions": {}}
    for cond in spec["mass"]["conditions"]:
        m, c = mass_condition(spec, cond)
        h, trim = hy.solve(m, c, HS["rho"])
        out["conditions"][cond] = hy.report(h, trim, m, c, HS["rho"], HS["g"])
    out["design_condition"] = spec["mass"]["design_condition"]
    out["bonjean"] = hy.bonjean(spec)
    return out


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


def sensor_frame(pos, look, pitch_down=0.0):
    """Look along `look` (its own tilt kept), then pitch down by pitch_down more."""
    f = np.asarray(look, float)
    f = f / np.linalg.norm(f)
    if pitch_down:
        f, _ = pitched(f.tolist(), pitch_down + math.degrees(math.asin(-f[1])))
    return T(*pos) @ frame_from_forward_up(f, (0.0, 1.0, 0.0))


def build_geometry(spec):
    """All parts in the vessel frame (numpy), plus the buoyant triangles."""
    L = Lines(spec)
    hull, collar, fit, mast = Part(), Part(), Part(), Part()
    build_hull(spec, L, hull)
    build_appendages(spec, L, hull)
    build_bottom_openings(spec, L, hull)
    build_collar(spec, L, collar)
    build_deck_fittings(spec, L, fit)
    build_mast(spec, mast)
    tris = np.concatenate(hull.tris + collar.tris)
    return {"hull": hull, "collar": collar, "deck_fittings": fit, "mast": mast}, tris


def build_blender(spec, parts, hydro):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    d = hydro["conditions"][hydro["design_condition"]]
    m_dry, com = mass_condition(spec, "lightship")
    root = new_object("mariner", None, mats, None, I, props={
        "frame": "X forward, Y up, Z starboard; origin = centreline x baseline x mid-LOA",
        "spec": "mariner_spec.json", "hydro": "mariner_hydro.json",
        "mass_dry": m_dry, "com_dry": [round(c, 4) for c in com],
        "design_waterline_y": d["waterline_y_at_x0"], "design_trim_deg": d["trim_deg"]})
    new_object("hull", parts["hull"], mats, root, I, sharp_deg=20.0)
    new_object("collar", parts["collar"], mats, root, I, sharp_deg=40.0)
    new_object("deck_fittings", parts["deck_fittings"], mats, root, I)
    new_object("mast", parts["mast"], mats, root, I)

    J = spec["propulsion"]["waterjet"]
    body, steer, bucket = build_jet(spec)
    ox, oy = J["flange"]["x"], J["axis_y"]
    jet = new_object("jet", body, mats, root, T(ox, oy, 0.0))
    st = J["steering"]
    new_object("jet_steering", steer, mats, jet, T(st["pivot_x"] - ox, 0.0, 0.0),
               props={"axis": "+Y", "limit_deg": st["limit_deg"], "positive": "nozzle exit swings to starboard, the stern is pushed to port, "
                                                   "the boat turns to starboard"})
    rb = J["reverse_bucket"]
    new_object("jet_reverse_bucket", bucket, mats, jet, T(rb["pivot"][0] - ox, rb["pivot"][1] - oy, 0.0),
               props={"axis": "+Z", "limit_deg": rb["limit_deg"], "rest": "raised (ahead)"})
    new_object("jet_thrust", None, mats, root, T(*J["thrust_point"]),
               props={"thrust_axis": "+X", "bollard_thrust_n": J["bollard_thrust_n"],
                      "steered_by": "jet_steering", "assumed": J["assumed"]})
    bt = spec["propulsion"]["bow_thruster"]
    new_object("bow_thruster", None, mats, root, T(*bt["position"]),
               props={"thrust_axis": bt["thrust_axis"], "thrust_n": bt["thrust_n"],
                      "tunnel_radius": bt["tunnel_radius"]})

    SE = spec["sensors"]
    cam = SE["camera_main"]
    new_object("camera_main", None, mats, root, sensor_frame(cam["position"], cam["look"], cam["pitch_down_deg"]),
               props={"convention": "look -Z, up +Y", "hfov_deg": cam["hfov_deg"],
                      "resolution": cam["resolution"], "fps": cam["fps"]})
    rd = SE["radar"]
    new_object("radar", None, mats, root, sensor_frame(rd["position"], (1, 0, 0)),
               props={k: rd[k] for k in ("spin_axis", "rpm", "h_beam_deg", "v_beam_deg", "range_max")})
    for name in ("gnss_fore", "gnss_aft"):
        g = SE[name]
        p = list(g["position"])
        p[1] -= 0.5 * g["disc_height"]
        new_object(name, None, mats, root, T(*p), props={"role": "phase centre", "baseline": SE["gnss_baseline"]})
    av = SE["ais_vhf_antenna"]
    new_object("ais_vhf_antenna", None, mats, root, T(*av["base"]), props={"length": av["length"]})
    new_object("lte_antenna_1", None, mats, root, T(*SE["lte_antenna_1"]["base"]))
    new_object("lte_antenna_2", None, mats, root, T(*SE["lte_antenna_2"]["position"]))
    new_object("imu", None, mats, root, T(*SE["imu"]["position"]))
    td = SE["transducer"]
    new_object("transducer", None, mats, root,
               T(*td["position"]) @ frame_from_forward_up(td["look"], (1.0, 0.0, 0.0)),
               props={"convention": "look -Z, up +Y"})
    mp = SE["moonpool"]
    new_object("moonpool", None, mats, root, T(*mp["position"]), props={"diameter": mp["diameter"]})
    LI = spec["lights"]
    for name, d_ in LI.items():
        look = d_.get("look", (1.0, 0.0, 0.0))
        props = {"convention": "look -Z, up +Y"}
        props.update({k: v for k, v in d_.items() if k in ("colour", "cone_deg")})
        new_object(name, None, mats, root, sensor_frame(d_["position"], look), props=props)
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="mariner_spec.json")
    ap.add_argument("--out", default="mariner.glb")
    ap.add_argument("--hydro-out", default="mariner_hydro.json")
    ap.add_argument("--hydro-only", action="store_true")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def path(p):
        return p if os.path.isabs(p) else os.path.join(here, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    com = check_spec(spec)
    print(f"[mariner] spec ok; dry CoM from budget = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")

    parts, tris = build_geometry(spec)
    body, steer, bucket = build_jet(spec)
    used = set(m for p in list(parts.values()) + [body, steer, bucket] for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    hydro = hydrostatics(spec, tris)
    for cond, r in hydro["conditions"].items():
        print(f"[mariner] {cond:9s} {r['mass']:6.0f} kg  waterline y {r['waterline_y_at_x0']:.3f}  "
              f"max draft {r['draft_max']:.3f}  trim {r['trim_deg']:+.2f} deg  LCB {r['lcb']:+.3f}  "
              f"GM_T {r['gm_t']:.2f}  GM_L {r['gm_l']:.1f}  LWL {r['lwl']:.2f}  BWL {r['bwl']:.2f}  "
              f"Awp {r['waterplane_area']:.2f}")
    with open(path(a.hydro_out), "w", encoding="utf-8") as fh:
        json.dump(hydro, fh, indent=1)
    print(f"[mariner] wrote {path(a.hydro_out)}")
    HS, P = spec["hydrostatics"], spec["principal"]
    d = hydro["conditions"][spec["mass"]["design_condition"]]
    assert abs(d["draft_max"] - P["draft_brochure"]) <= HS["draft_tolerance"], d["draft_max"]
    assert abs(d["trim_deg"]) <= HS["trim_limit_deg"], d["trim_deg"]
    assert d["gm_t"] > 0.3, d["gm_t"]
    if a.hydro_only:
        return

    import bpy
    build_blender(spec, parts, hydro)
    bpy.context.view_layer.update()
    tris_n, zmax, zmin = 0, -1e9, 1e9
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            tris_n += sum(len(p.vertices) - 2 for p in ob.data.polygons)
            mw = ob.matrix_world
            zs = [(mw @ v.co).z for v in ob.data.vertices]
            zmax, zmin = max(zmax, max(zs)), min(zmin, min(zs))
    bpy.ops.export_scene.gltf(filepath=path(a.out), export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True)
    print(f"[mariner] wrote {path(a.out)}  triangles={tris_n}  y range [{zmin:.3f}, {zmax:.3f}]")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

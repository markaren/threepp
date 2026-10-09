"""Build the Foxtech Babyshark 260 VTOL from babyshark_spec.json: babyshark.glb (via Blender).

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_babyshark_blender.py -- --spec babyshark_spec.json --out babyshark.glb

Geometry only (plain Python + numpy, no Blender; prints the checks and fails on a gate):
    python build_babyshark_blender.py --check
    python build_babyshark_blender.py --check --write-areas    # and the measured projected areas into the spec

The mesh accumulator, the primitives and the Blender export are build_common's (../build_common.py,
shared by every generator); this script adds the aircraft. All geometry is built with numpy in the
spec's layout frame (X forward, Y up, Z starboard, metres, origin at the centre of gravity) and
converted to Blender's Z-up only when a mesh or a transform is handed to bpy. Every number is the
spec's: `layout` for the lifting surfaces, the rotor axes and the feet (the thesis author's
measurements, met exactly), `model` for the shapes fitted by eye to the thesis's renders, `materials`.

The wing is one closed loft from tip to tip through the pod: a section at each station, its leading
and trailing edge a monotone cubic through the layout's sections; where an aileron is, the section
stops at the hinge line and the aileron is the rest of it, lofted on its own node. The tail is one
closed loft along the inverted V, from one boom over the centre piece to the other, cut the same way
for the ruddervators.

Node hierarchy (all names exact):
    babyshark                    root, identity, origin = the centre of gravity
      airframe                   an empty; the fixed meshes are its children:
        fuselage, wing, tail, booms, motors, gear, fittings
      aileron_left, aileron_right
                                 on the hinge lines: local +Z along the hinge toward starboard; a positive
                                 turn about local +Z puts the trailing edge down on either side
      ruddervator_left, ruddervator_right
                                 on the tail panels' hinge lines: local +Z along the hinge toward starboard
                                 (down the starboard panel, up the port one); a positive turn about local
                                 +Z moves the trailing edge toward the panel's lower, inner face
      lift_fr, lift_fl, lift_rl, lift_rr
                                 at the lift rotors' disc centres, above the booms; hub and two blades;
                                 spins about local +Y (`spin` +1 = counter-clockwise seen from above)
      pusher                     at the pusher's disc centre, spins about local +X (positive: clockwise
                                 seen from behind); hub and two blades
      imu, gnss, pitot_tip, camera_nadir, camera_fpv        sensor empties

A hinge node's local +Y has no X part, so its rest rotation has no turn about its own +Z and a
scene may set rotation.z alone. The cameras look down local -Z with local +Y up (threepp camera
convention).
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # examples/ (build_common)
from build_common import (Part, R, T, Z_TO_Y, box, export, frame_from_forward_up, frame_y_toward, loft,  # noqa: E402
                          make_materials, new_object, newell, pchip, revolve, signed_volume, strut, table)

I4 = np.eye(4)
LIFT = ("fr", "fl", "rl", "rr")
WHITE = "airframe_white"


# ---------------------------------------------------------------- primitives
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def toward(origin, d):
    """Placement that puts a solid of revolution's +Z along d at origin."""
    return T(*origin) @ frame_y_toward(unit(d)) @ Z_TO_Y()


def spun(part, profile, mat, origin, d, n=32, M=None):
    v, f = revolve(profile, n)
    A = toward(origin, d)
    part.add(v, f, mat, A if M is None else M @ A)
    return np.asarray(v, float) @ A[:3, :3].T + A[:3, 3]


def superellipse(hw, hu, hl, n, eu, el):
    """Closed loop in (y, z) about the origin, counter-clockwise from +z: half-width hw, hu up and hl
    down, a superellipse of exponent eu above the widest point and el below it."""
    pts = []
    for k in range(n):
        q = 2.0 * math.pi * k / n
        c, s = math.cos(q), math.sin(q)
        e = eu if s >= 0.0 else el
        pts.append(((hu if s >= 0.0 else hl) * math.copysign(abs(s) ** (2.0 / e), s), hw * math.copysign(abs(c) ** (2.0 / e), c)))
    return pts


def fillet(points, r, seg=5):
    """The polyline with each inner corner replaced by an arc of radius r, tangent to both legs."""
    P = [np.asarray(p, float) for p in points]
    out = [P[0]]
    for a, b, c in zip(P[:-2], P[1:-1], P[2:]):
        u, v = unit(a - b), unit(c - b)
        half = 0.5 * math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0)))
        d = min(r / math.tan(half), 0.45 * min(np.linalg.norm(a - b), np.linalg.norm(c - b)))
        rr = d * math.tan(half)
        centre = b + unit(u + v) * (rr / math.sin(half))
        p0, p1 = b + u * d - centre, b + v * d - centre
        a0 = math.atan2(p0[1], p0[0])
        da = (math.atan2(p1[1], p1[0]) - a0 + math.pi) % (2.0 * math.pi) - math.pi
        out += [centre + rr * np.array([math.cos(a0 + da * k / seg), math.sin(a0 + da * k / seg)]) for k in range(seg + 1)]
    out.append(P[-1])
    return out


def mitred(points):
    """[(point, normal, scale)] along a polyline in (y, z): the normal turns the tangent a quarter turn
    (up where the line runs toward +z), bisects a corner and is scaled so a strip keeps its thickness."""
    P = [np.asarray(p, float) for p in points]
    ns = [np.array([t[1], -t[0]]) for t in (unit(b - a) for a, b in zip(P[:-1], P[1:]))]
    out = []
    for i, p in enumerate(P):
        a, b = ns[max(i - 1, 0)], ns[min(i, len(ns) - 1)]
        n = unit(a + b)
        out.append((p, n, 1.0 / float(np.dot(n, a))))
    return out


def lens(part, at, d, barrel_r, glass_r, proud, n=28):
    """A window: a bezel standing `proud` of the surface at `at`, a cone inside it down to the domed glass."""
    gi, d0 = 0.72 * glass_r, proud - 0.30 * glass_r
    spun(part, [(barrel_r, -0.004), (barrel_r, proud - 0.0008), (barrel_r - 0.0008, proud), (glass_r, proud), (gi, d0)], "trim_black", at, d, n)
    spun(part, [(0.985 * gi, d0 - 0.0015), (0.985 * gi, d0 + 0.0003), (0.62 * gi, d0 + 0.10 * glass_r), (0.0, d0 + 0.14 * glass_r)],
         "lens_glass", at, d, n)


def merged(*parts):
    out = Part()
    for p in parts:
        b = len(out.verts)
        out.verts.extend(p.verts)
        out.faces.extend(tuple(b + k for k in f) for f in p.faces)
        out.mats.extend(p.mats)
    return out


def hinge_frame(p_from, p_to):
    """4x4 node matrix: origin p_from, local +Z along p_from -> p_to, local +Y across it with no X part
    (the rest rotation then has no turn about its own +Z: Rx Ry only)."""
    z = unit(np.asarray(p_to, float) - np.asarray(p_from, float))
    y = unit((0.0, z[2], -z[1]))
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = np.cross(y, z), y, z, p_from
    return M


# ---------------------------------------------------------------- the airfoils
def thickness(x):
    """NACA four-digit half-thickness per unit thickness ratio, open trailing edge (0.0105 t at x = 1)."""
    return 5.0 * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def camber(x, m, p):
    """NACA four-digit camber line: its height m at the chord fraction p."""
    return m / (p * p) * (2.0 * p * x - x * x) if x < p else m / ((1.0 - p) ** 2) * (1.0 - 2.0 * p + 2.0 * p * x - x * x)


def chord_points(x0, x1, n):
    """n + 1 chord fractions from x0 to x1, closer together toward x0."""
    return [x0 + (x1 - x0) * (1.0 - math.cos(0.5 * math.pi * i / n)) for i in range(n + 1)]


def around(xs, upper_all=False):
    """A section's loop as (chord fraction, upper?): the upper surface from the last fraction to the
    first, the lower back (from the first again if upper_all, for a surface that starts at a hinge)."""
    return [(x, True) for x in reversed(xs)] + [(x, False) for x in (xs if upper_all else xs[1:])]


# ---------------------------------------------------------------- the wing
class Wing:
    """layout.wing: the planform through its sections and the section over it, as functions of the
    distance z from the centreline."""

    N_CHORD = 22                      # chordwise intervals per surface

    def __init__(self, spec):
        L, S = spec["layout"]["wing"], spec["model"]["wing"]
        sec = np.asarray(L["sections"], float)                   # [X of the leading edge, Z, chord]
        self.sec, self.y, self.zt = sec, L["y"], float(sec[-1, 1])
        self._le, self._te = pchip(sec[:, 1], sec[:, 0]), pchip(sec[:, 1], sec[:, 0] - sec[:, 2])
        (x0, z0, _), (x1, z1, _) = sec[-2], sec[-1]               # the leading edge's last stretch ends steeper: a round tip
        self._tip = (z0, z1, x0, x1, (self._le(z0) - self._le(z0 - 1e-6)) / 1e-6, S["tip_slope"] * (x1 - x0) / (z1 - z0))
        self.tr, self.m, self.p = S["thickness_ratio"], S["camber"], S["camber_at"]
        self.tip_round, self.gap = S["tip_round"], S["aileron_gap"]
        A = L["aileron"]
        (self.a_z0, self.a_z1), (f0, f1) = A["z"], A["hinge_chord_fraction"]
        self.h0 = self.x_le(self.a_z0) - f0 * self.chord(self.a_z0)
        self.h1 = self.x_le(self.a_z1) - f1 * self.chord(self.a_z1)

    def x_le(self, z):
        z = min(abs(z), self.zt)
        z0, z1, x0, x1, m0, m1 = self._tip
        if z <= z0:
            return self._le(z)
        h, t = z1 - z0, (z - z0) / (z1 - z0)                      # cubic Hermite, as pchip's
        return ((1.0 + 2.0 * t) * (1.0 - t) ** 2 * x0 + t * (1.0 - t) ** 2 * h * m0 + t * t * (3.0 - 2.0 * t) * x1
                + t * t * (t - 1.0) * h * m1)

    def x_te(self, z):
        return self._te(min(abs(z), self.zt))

    def chord(self, z):
        return self.x_le(z) - self.x_te(z)

    def hinge_x(self, z):
        """X of the aileron's straight hinge line at z."""
        return self.h0 + (self.h1 - self.h0) * (z - self.a_z0) / (self.a_z1 - self.a_z0)

    def x_cut(self, z):
        return (self.x_le(z) - self.hinge_x(z)) / self.chord(z)

    def tip_factor(self, z):
        u = (z - (self.zt - self.tip_round)) / self.tip_round
        return 1.0 if u <= 0.0 else max(0.16, math.sqrt(max(0.0, 1.0 - u * u)))

    def surface(self, z, x, upper, k=1.0):
        """(X, Y) of the upper or lower surface at the chord fraction x; k scales the thickness."""
        c = self.chord(z)
        yt = thickness(x) * self.tr * c * k
        return self.x_le(z) - x * c, self.y + camber(x, self.m, self.p) * c + (yt if upper else -yt)

    def stations(self):
        """[(z, cut at the hinge?)] from the centreline to the tip."""
        zt, tr = self.zt, self.tip_round
        z0c, z1c = round(self.a_z0 - self.gap, 6), round(self.a_z1 + self.gap, 6)
        zs = set(np.round(np.arange(0.0, zt - 0.16, 0.045), 6).tolist()) | set(np.round(np.arange(zt - 0.16, zt - tr, 0.008), 6).tolist())
        zs |= set(np.round(self.sec[:, 1], 6).tolist())
        cut = set(np.round(np.linspace(z0c, z1c, 15), 6).tolist()) | {z for z in zs if z0c < z < z1c}
        zs = {z for z in zs if not (z0c - 0.004 <= z <= z1c + 0.004) and z < zt - tr - 0.004}
        zs |= {round(z0c - 0.003, 6), round(z1c + 0.003, 6)}
        zs |= {round(zt - tr * (1.0 - math.sin(0.5 * math.pi * k / 6)), 6) for k in range(7)}
        return [(z, z in cut) for z in sorted(zs | cut)]

    def ring(self, z, cut, side):
        xs = chord_points(0.0, self.x_cut(z) if cut else 1.0, self.N_CHORD)
        k = self.tip_factor(z)
        return [self.surface(z, x, up, k) + (side * z,) for x, up in around(xs)]


def build_wing(wing):
    st = wing.stations()
    rings = [wing.ring(z, c, -1.0) for (z, c) in reversed(st[1:])] + [wing.ring(z, c, 1.0) for (z, c) in st]
    part = Part()
    v, f = loft(rings)
    part.add(v, f, WHITE)
    return part


def build_aileron(wing, side):
    """The aileron on one side: the mesh in its node's frame, and the node's matrix."""
    zs = set(np.round(np.linspace(wing.a_z0, wing.a_z1, 15), 6).tolist()) | {float(z) for z in wing.sec[:, 1] if wing.a_z0 < z < wing.a_z1}
    n = wing.N_CHORD // 2
    rings, hinge = [], []
    for z in sorted(zs):
        xc = wing.x_cut(z)
        rings.append([wing.surface(z, x, up) + (side * z,) for x, up in around(chord_points(xc, 1.0, n), True)])
        hinge.append(wing.surface(z, xc, True, k=0.0) + (side * z,))          # on the camber line
    a, b = (hinge[0], hinge[-1]) if side > 0 else (hinge[-1], hinge[0])
    M = hinge_frame(a, b)
    part = Part()
    v, f = loft(rings)
    part.add(v, f, WHITE, np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the tail
class Tail:
    """layout.tail: the inverted V as a path of its leading edge in (Y, Z), the section across it."""

    N_CHORD = 14

    def __init__(self, spec):
        L, S = spec["layout"]["tail"], spec["model"]["tail"]
        self.L, self.S = L, S
        self.x_le, self.c, self.inc = L["x_leading_edge"], L["chord"], math.radians(L["incidence_deg"])
        self.foot, self.top = np.array(L["panel_foot"], float), np.array(L["panel_top"], float)      # (Y, Z), starboard
        self.length = float(np.linalg.norm(self.top - self.foot))
        self.hinge = L["hinge_chord_fraction"]
        self.rv0, self.rv1 = S["ruddervator_span"]
        self.tr = 0.01 * float(L["airfoil"].split()[-1][-2:])     # "NACA 0010": 10 %

    def at(self, f, side=1.0):
        """(Y, Z) of the leading edge the fraction f up the panel from its foot."""
        return (self.foot + f * (self.top - self.foot)) * (1.0, side)

    def normal(self, side=1.0):
        """The panel's outer normal in (Y, Z): up and outboard."""
        t = unit(self.top - self.foot)
        return np.array([-t[1], t[0]]) * (1.0, side)

    def path(self):
        """[((Y, Z), cut at the hinge?)] from the port panel's end in its boom over the centre piece to
        the starboard panel's."""
        S, L = self.S, self.L
        g, e = S["gap"] / self.length, 0.003 / self.length
        f0, f1 = self.rv0 - g, self.rv1 + g
        fs = [(-S["foot_extension"] / self.length, False), (0.0, False), (f0 - e, False)]
        fs += [(f, True) for f in np.linspace(f0, f1, 9)] + [(f1 + e, False), (1.0, False)]
        half = [(self.at(f), c) for f, c in fs]                   # the starboard panel, foot to top
        # the centre piece: straight from each panel's top to centre_y on the centreline
        assert abs(self.top[1] - L["centre_half_width"]) < 1e-9, "the panels' tops are not the centre piece's ends"
        port = [(p * (1.0, -1.0), c) for p, c in half]
        return port + [(np.array([L["centre_y"], 0.0]), False)] + half[::-1]

    def offset(self, x, t):
        """(along X, along the normal) of the point at the chord fraction x and the height t off the
        chord line, the section set at the incidence about its leading edge (leading edge up)."""
        ci, si = math.cos(self.inc), math.sin(self.inc)
        return -x * self.c * ci - t * si, -x * self.c * si + t * ci

    def ring(self, p, n, k, xs, upper_all=False):
        pts = []
        for x, up in around(xs, upper_all):
            t = thickness(x) * self.tr * self.c * (1.0 if up else -1.0)
            a, b = self.offset(x, t)
            pts.append((self.x_le + a, p[0] + k * b * n[0], p[1] + k * b * n[1]))
        return pts


def build_tail(tail):
    """The fixed tail, and the index of each panel end's leading-edge vertex (for the gates)."""
    path = tail.path()
    st = mitred([p for p, _ in path])
    rings = [tail.ring(p, n, k, chord_points(0.0, tail.hinge if cut else 1.0, tail.N_CHORD)) for (p, n, k), (_, cut) in zip(st, path)]
    part = Part()
    v, f = loft(rings)
    part.add(v, f, WHITE)
    N = len(rings[0])
    marks = {}
    for i, (p, _) in enumerate(path):
        for key, q in (("foot", tail.foot), ("top", tail.top)):
            for side, s in (("port", -1.0), ("starboard", 1.0)):
                if np.allclose(p, q * (1.0, s), atol=1e-9):
                    marks[f"{key} {side}"] = i * N + tail.N_CHORD
    return part, marks


def build_ruddervator(tail, side):
    """The ruddervator on one panel: the mesh in its node's frame, and the node's matrix."""
    n_out = tail.normal(side)
    xs = chord_points(tail.hinge, 1.0, tail.N_CHORD // 2)
    rings, hinge = [], []
    for f in np.linspace(tail.rv0, tail.rv1, 9):
        p = tail.at(f, side)
        rings.append(tail.ring(p, n_out, 1.0, xs, True))
        a, b = tail.offset(tail.hinge, 0.0)
        hinge.append((tail.x_le + a, p[0] + b * n_out[0], p[1] + b * n_out[1]))
    a, b = (hinge[-1], hinge[0]) if side > 0 else (hinge[0], hinge[-1])      # toward starboard
    M = hinge_frame(a, b)
    part = Part()
    v, f = loft(rings)
    part.add(v, f, WHITE, np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the pod
class Pod:
    """model.fuselage: the pod's outlines over X and the section between them."""

    def __init__(self, spec):
        F = spec["model"]["fuselage"]
        self.F = F
        self.xn, self.yn, self.xe = F["nose_x"], F["nose_y"], F["end_x"]
        self._w, self._t, self._b = table(F["half_width"]), table(F["top"]), table(F["bottom"])

    def _nose(self, x, a):
        d = self.xn - x
        return 1.0 if d >= a else math.sqrt(max(0.0, 1.0 - ((a - d) / a) ** 2))

    def half_width(self, x):
        return self._w(x) * self._nose(x, self.F["nose_length"]["width"])

    def top(self, x):
        return self.yn + (self._t(x) - self.yn) * self._nose(x, self.F["nose_length"]["top"])

    def bottom(self, x):
        return self.yn + (self._b(x) - self.yn) * self._nose(x, self.F["nose_length"]["bottom"])

    def ring(self, x, n=40):
        F = self.F
        t, b, w = self.top(x), self.bottom(x), self.half_width(x)
        x0, x1 = F["tail_blend_x"]
        u = min(1.0, max(0.0, (x0 - x) / (x0 - x1)))                # 0 ahead of the blend, 1 at the end: a circle there
        u = u * u * (3.0 - 2.0 * u)
        yw = t - (F["widest_at"] + (0.5 - F["widest_at"]) * u) * (t - b)
        eu, el = (e + (2.0 - e) * u for e in F["exponent"])
        return [(x, yw + y, z) for (y, z) in superellipse(w, t - yw, yw - b, n, eu, el)]

    def stations(self):
        a = self.F["nose_length"]["width"]
        nose = [self.xn - a * (1.0 - math.cos(q)) for q in [0.035, 0.09, 0.17] + np.linspace(0.26, 0.5 * math.pi, 12).tolist()]
        return nose + np.linspace(self.xn - a, self.xe, 26)[1:].tolist()


def build_fuselage(pod):
    part = Part()
    v, f = loft([pod.ring(x) for x in pod.stations()])
    part.add(v, f, WHITE)
    return part


# ---------------------------------------------------------------- the booms and their motors
def build_boom(spec, wing, B, Mo, cans, side):
    """One boom: the moulded arm under the wing, the saddle between them, the tube to the tail, and the
    two lift motors on the arm's pads."""
    S, Lay = spec["model"], spec["layout"]
    Bm, LM = S["boom"], S["lift_motor"]
    z = side * Lay["boom"]["z"]
    fa = np.asarray(Bm["fairing"], float)                       # [X, half width, half height, centre Y]
    w_, h_, y_ = pchip(fa[:, 0], fa[:, 1]), pchip(fa[:, 0], fa[:, 2]), pchip(fa[:, 0], fa[:, 3])
    x0, x1, a = float(fa[0, 0]), float(fa[-1, 0]), Bm["fairing_nose"]
    qs = [0.5 * math.pi * (1.0 - k / 8.0) for k in range(1, 8)] + [0.09, 0.035]
    xs = np.linspace(x0, x1 - a, 44).tolist() + [x1 - a * (1.0 - math.cos(q)) for q in qs]
    rings = []
    for x in xs:
        d = x1 - x
        e = 1.0 if d >= a else math.sqrt(max(0.0, 1.0 - ((a - d) / a) ** 2))
        ex = Bm["fairing_exponent"]
        rings.append([(x, y_(x) + y, z + zz) for (y, zz) in superellipse(w_(x) * e, h_(x) * e, h_(x) * e, 24, ex, ex)])
    v, f = loft(rings)
    B.add(v, f, WHITE)
    # the saddle: from the arm's middle up into the wing's underside, along the chord
    zc = abs(z)
    rings = []
    for u in np.linspace(0.03, 0.97, 17):
        X, lo = wing.surface(zc, u, False)
        _, up = wing.surface(zc, u, True)
        top, bot = min(lo + 0.003, 0.5 * (lo + up)), y_(X)
        hw = max(0.003, Bm["saddle_half_width"] * max(0.0, 1.0 - (2.0 * u - 1.0) ** 2) ** 0.3)
        rings.append([(X, 0.5 * (top + bot) + y, z + zz) for (y, zz) in superellipse(hw, 0.5 * (top - bot), 0.5 * (top - bot), 12, 3.0, 3.0)])
    v, f = loft(rings)
    B.add(v, f, WHITE)
    # the tube: a white sleeve out of the arm, bare carbon, the white fitting the tail panel stands on
    ty, rt, rs = Bm["tube_y"], Bm["tube_radius"], Bm["sleeve_radius"]
    aft = (-1.0, 0.0, 0.0)
    L = Bm["sleeve_x"][0] - Bm["sleeve_x"][1]
    spun(B, [(rs - 0.002, 0.0), (rs, 0.003), (rs, L - 0.004), (rt, L)], WHITE, (Bm["sleeve_x"][0], ty, z), aft, 24)
    strut(B, (Bm["carbon_x"][0], ty, z), (Bm["carbon_x"][1], ty, z), rt, "carbon", 24)
    L = Bm["tail_fitting_x"][0] - Bm["tail_fitting_x"][1]
    spun(B, [(rt, 0.0), (rs, 0.004), (rs, L - 0.010), (0.8 * rs, L - 0.004), (0.45 * rs, L - 0.001), (0.0, L)], WHITE,
         (Bm["tail_fitting_x"][0], ty, z), aft, 24)
    # the lift motors: a base on the pad, the can, the shaft up to the propeller's hub
    for name in ("f", "r"):
        key = name + ("r" if side > 0 else "l")
        p = np.array(Lay["lift_rotors"][key]["pos"], float)
        at = (p[0], 0.0, p[2])
        assert abs(y_(p[0]) + h_(p[0]) - LM["base_y"][0]) < 0.004, ("lift motor off its pad", key)
        rb, rc, (c0, c1) = LM["base_radius"], LM["can_radius"], LM["can_y"]
        spun(Mo, [(rb, LM["base_y"][0]), (rb, LM["base_y"][1])], "motor_black", at, (0, 1, 0), 32)
        cans[key] = spun(Mo, [(rc - 0.0015, c0), (rc, c0 + 0.0015), (rc, c1 - 0.003), (rc - 0.002, c1 - 0.0008), (0.45 * rc, c1)],
                         "motor_black", at, (0, 1, 0), 48)
        hub0 = p[1] + S["lift_propeller"]["hub_y"][0]
        spun(Mo, [(LM["shaft_radius"], c1 - 0.001), (LM["shaft_radius"], hub0 + 0.001)], "motor_steel", at, (0, 1, 0), 16)


def build_pusher_motor(spec, Mo):
    PM, PP = spec["model"]["pusher_motor"], spec["model"]["pusher_propeller"]
    px, py, pz = spec["layout"]["pusher"]["pos"]
    r, (m0, m1) = PM["radius"], PM["x"]
    L = m0 - m1
    spun(Mo, [(r - 0.003, -0.002), (r, 0.002), (r, L - 0.004), (r - 0.0025, L - 0.0008), (0.45 * r, L)], "motor_black", (m0, py, pz), (-1, 0, 0), 48)
    hub0 = px + PP["hub_x"][1]                                   # the hub's forward face
    spun(Mo, [(PM["shaft_radius"], 0.0), (PM["shaft_radius"], m1 + 0.001 - hub0 + 0.001)], "motor_steel", (m1 + 0.001, py, pz), (-1, 0, 0), 16)


# ---------------------------------------------------------------- the landing gear
def build_gear(spec, pod):
    """The two frames, and each pad's vertices (front starboard, front port, rear port, rear starboard)."""
    G = spec["model"]["gear"]
    part, pads = Part(), {}
    ky, kz = G["knee"]
    hw, ht = 0.5 * G["strip_width"], 0.5 * G["strip_thickness"]
    half = [(G["belly_y"], G["belly_half_width"]), (ky, kz), (ky, G["foot_end_z"]), (ky + G["lip"][0], G["foot_end_z"] + G["lip"][1])]
    path = fillet([(y, -zz) for (y, zz) in reversed(half)] + half, G["bend_radius"])
    sec = [(-ht, -hw + 0.001), (-ht + 0.001, -hw), (ht - 0.001, -hw), (ht, -hw + 0.001), (ht, hw - 0.001), (ht - 0.001, hw), (-ht + 0.001, hw), (-ht, hw - 0.001)]
    pz0, pz1 = G["pad"]["z"]
    for i, x in enumerate(G["x"]):
        assert abs(pod.bottom(x) - (G["belly_y"] + ht)) < 0.004, ("gear frame off the belly", x)
        v, f = loft([[(x + b, p[0] + k * a * n[0], p[1] + k * a * n[1]) for (a, b) in sec] for (p, n, k) in mitred(path)])
        part.add(v, f, WHITE)
        for s in (1.0, -1.0):
            v, f = box(G["strip_width"], G["pad"]["thickness"], pz1 - pz0)
            Mp = T(x, ky - ht - 0.5 * G["pad"]["thickness"], s * 0.5 * (pz0 + pz1))
            part.add(v, f, "rubber", Mp)
            pads[(i, s)] = np.asarray(v, float) + Mp[:3, 3]
    return part, [pads[k] for k in ((0, 1.0), (0, -1.0), (1, -1.0), (1, 1.0))]


# ---------------------------------------------------------------- the fittings
def look_frame(c):
    return T(*c["position"]) @ frame_from_forward_up(unit(c["look"]), c["up"])


def build_fittings(spec, pod):
    S = spec["model"]
    SE = S["sensors"]
    F = Part()
    pt = S["pitot"]                                              # out of the nose, on its axis
    x0, x1 = pt["x"]
    spun(F, [(pt["radius"], 0.0), (pt["radius"], x1 - x0 - 0.006), (0.55 * pt["radius"], x1 - x0)], "pitot_steel", (x0, pod.yn, 0.0), (1, 0, 0), 12)
    gn = S["gnss"]                                               # a puck on the pod's back
    gp = SE["gnss"]["position"]
    assert gp[1] - gn["height"] < pod.top(gn["x"]) < gp[1], ("GNSS puck off the pod's back", pod.top(gn["x"]))
    r, h = gn["radius"], gn["height"]
    spun(F, [(r, -h - 0.006), (r, -0.004), (r - 0.004, 0.0), (0.0, 0.0)], "trim_black", gp, (0, 1, 0), 28)
    cn = SE["camera_nadir"]["position"]                          # the windows: in the belly, in the nose
    assert abs(pod.bottom(cn[0]) - cn[1]) < 0.002, ("nadir window off the belly", pod.bottom(cn[0]))
    lens(F, cn, (0, -1, 0), 0.022, 0.017, 0.003)
    cf = SE["camera_fpv"]["position"]
    assert abs(pod.bottom(cf[0]) - cf[1]) < 0.002, ("forward window off the nose", pod.bottom(cf[0]))
    dx = 0.0005
    tangent = unit((2.0 * dx, pod.bottom(cf[0] + dx) - pod.bottom(cf[0] - dx), 0.0))
    lens(F, cf, (tangent[1], -tangent[0], 0.0), 0.0125, 0.009, 0.0025)
    return F


# ---------------------------------------------------------------- the propellers
def blade_sections(P, Rt, n=12, m=27):
    """[(r, twist, outline)] along a blade from root_r to the tip: the outline is (u, w) in metres, u
    along the chord from the trailing to the leading edge, w across it."""
    chord, r0 = table(P["chord"]), P["root_r"]
    out = []
    for f in sorted(set(np.round(np.linspace(0.0, 1.0, m), 5).tolist()) | {0.97, 0.99}):
        r = r0 + (Rt - r0) * f
        c, th = chord(f), P["thickness"][0] + (P["thickness"][1] - P["thickness"][0]) * f
        beta = min(math.atan(P["pitch_m"] / (2.0 * math.pi * r)), math.radians(P["max_twist_deg"]))
        out.append((r, beta, [(0.5 * math.cos(q) * c, 0.5 * math.sin(q) * (1.0 + 0.35 * math.cos(q)) * th)
                              for q in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)]))
    return out


def build_lift_propeller(spec, spin):
    """(hub, blades) in a lift node's frame: the disc centre at the origin, the axis local +Y, the blades
    along local +-X. `spin` +1 turns counter-clockwise seen from above, so the blade on +X moves toward
    -Z and carries its leading edge there, raised above its trailing edge: thrust up."""
    P = spec["model"]["lift_propeller"]
    hub, blades = Part(), Part()
    rings = [[(r, u * math.sin(b) + w * math.cos(b), spin * (-u * math.cos(b) + w * math.sin(b))) for (u, w) in sec]
             for (r, b, sec) in blade_sections(P, 0.5 * spec["actuators"]["D_mr"])]
    v, f = loft(rings)
    for ang in (0.0, 180.0):
        blades.add(v, f, "prop_carbon", R("Y", ang))
    h0, h1 = P["hub_y"]
    hr = P["hub_radius"]
    spun(hub, [(0.7 * hr, h0), (hr, h0 + 0.002), (hr, h1 - 0.002), (0.7 * hr, h1), (0.0, h1 + 0.0006)], "motor_black", (0, 0, 0), (0, 1, 0), 28)
    for e in (-1.0, 1.0):                                        # the two screws of the propeller's clamp
        spun(hub, [(0.0022, h1 - 0.001), (0.0022, h1 + 0.0016), (0.0, h1 + 0.0019)], "motor_steel", (0.0, 0.0, e * 0.008), (0, 1, 0), 10)
    return hub, blades


def build_pusher_propeller(spec):
    """(hub, blades) in the pusher node's frame: the disc centre at the origin, the axis local +X, the
    blades along local +-Y. A positive turn about +X moves the blade on +Y toward +Z (clockwise seen
    from behind); it carries its leading edge there, ahead of its trailing edge: thrust forward."""
    P = spec["model"]["pusher_propeller"]
    hub, blades = Part(), Part()
    rings = [[(u * math.sin(b) + w * math.cos(b), r, u * math.cos(b) - w * math.sin(b)) for (u, w) in sec]
             for (r, b, sec) in blade_sections(P, 0.5 * spec["actuators"]["D_fw"])]
    v, f = loft(rings)
    for ang in (0.0, 180.0):
        blades.add(v, f, "prop_wood", R("X", ang))
    h0, h1 = P["hub_x"]
    hr = P["hub_radius"]
    spun(hub, [(0.75 * hr, h0), (hr, h0 + 0.003), (hr, h1 - 0.003), (0.75 * hr, h1)], "prop_wood", (0, 0, 0), (1, 0, 0), 28)
    spun(hub, [(0.62 * hr, 0.0), (0.62 * hr, 0.002), (0.34 * hr, 0.0025), (0.34 * hr, 0.008), (0.2 * hr, 0.011), (0.0, 0.0115)], "motor_steel",
         (h0, 0, 0), (-1, 0, 0), 20)                             # the washer and the nut, aft
    return hub, blades


# ---------------------------------------------------------------- the nodes
def build_nodes(spec):
    """[(name, parent, part or None, local matrix, props, sharp angle)] in creation order, and what the
    checks need."""
    S, Lay = spec["model"], spec["layout"]
    wing, tail, pod = Wing(spec), Tail(spec), Pod(spec)
    booms, motors, cans = Part(), Part(), {}
    for side in (1.0, -1.0):
        build_boom(spec, wing, booms, motors, cans, side)
    build_pusher_motor(spec, motors)
    tail_part, marks = build_tail(tail)
    gear, pads = build_gear(spec, pod)
    nodes = [("babyshark", None, None, I4, {
        "frame": "X forward, Y up, Z starboard; origin = the centre of gravity (the thesis's body origin)",
        "spec": "babyshark_spec.json", "body_axes": "body (x fwd, y right, z down) = (X, Z, -Y)",
        "mass": spec["physical"]["mass"], "model": spec["reference"]["thesis"]["url"]}, 32.0)]
    nodes.append(("airframe", "babyshark", None, I4, {}, 32.0))
    for name, part, sharp in (("fuselage", build_fuselage(pod), 40.0), ("wing", build_wing(wing), 40.0), ("tail", tail_part, 40.0),
                              ("booms", booms, 35.0), ("motors", motors, 32.0), ("gear", gear, 32.0), ("fittings", build_fittings(spec, pod), 32.0)):
        nodes.append((name, "airframe", part, I4, {}, sharp))
    A = spec["actuators"]
    for name, side in (("aileron_left", -1.0), ("aileron_right", 1.0)):
        part, M = build_aileron(wing, side)
        nodes.append((name, "babyshark", part, M, {"axis": "+Z (hinge, toward starboard)", "limit_deg": A["delta_a_max_deg"],
                                                    "positive": "trailing edge down"}, 40.0))
    for name, side in (("ruddervator_left", -1.0), ("ruddervator_right", 1.0)):
        part, M = build_ruddervator(tail, side)
        nodes.append((name, "babyshark", part, M, {"axis": "+Z (hinge, toward starboard)",
                                                    "positive": "trailing edge toward the panel's lower, inner face"}, 40.0))
    hubs, blades = {}, {}
    for k in LIFT:
        spin = Lay["lift_rotors"][k]["spin"]
        hubs[k], blades[k] = build_lift_propeller(spec, float(spin))
        nodes.append(("lift_" + k, "babyshark", merged(hubs[k], blades[k]), T(*Lay["lift_rotors"][k]["pos"]),
                      {"axis": "+Y", "spin": spin, "positive": "counter-clockwise seen from above", "diameter": A["D_mr"]}, 32.0))
    hubs["pusher"], blades["pusher"] = build_pusher_propeller(spec)
    nodes.append(("pusher", "babyshark", merged(hubs["pusher"], blades["pusher"]), T(*Lay["pusher"]["pos"]),
                  {"axis": "+X", "positive": "spin vector along body +x, clockwise seen from behind", "diameter": A["D_fw"]}, 32.0))
    SE = S["sensors"]
    for k in ("imu", "gnss", "pitot_tip"):
        nodes.append((k, "babyshark", None, T(*SE[k]["position"]), {}, 32.0))
    for k in ("camera_nadir", "camera_fpv"):
        nodes.append((k, "babyshark", None, look_frame(SE[k]), {"convention": "look -Z, up +Y", "assumed": SE[k]["note"]}, 32.0))
    return nodes, {"wing": wing, "tail": tail, "pod": pod, "cans": cans, "hubs": hubs, "blades": blades, "marks": marks, "pads": pads}


def world_matrices(nodes):
    W = {}
    for name, parent, _, M, _, _ in nodes:
        W[name] = M if parent is None else W[parent] @ M
    return W


# ---------------------------------------------------------------- checks
def triangles(part, M):
    V = np.asarray(part.verts, float) @ M[:3, :3].T + M[:3, 3]
    idx = [(f[0], f[i], f[i + 1]) for f in part.faces for i in range(1, len(f) - 1)]
    return V[np.asarray(idx, int)]


def projected_area(tris, drop, cell):
    """Area of the union of the triangles seen along axis `drop`, and its centroid on the two other
    axes: the cell centres of a grid that are inside any of them."""
    ax = [i for i in range(3) if i != drop]
    P = tris[:, :, ax]
    lo = P.min(axis=(0, 1)) - cell
    nx, ny = np.ceil((P.max(axis=(0, 1)) + cell - lo) / cell).astype(int)
    G = np.zeros((nx + 1, ny + 1), bool)
    Q = (P - lo) / cell - 0.5
    mn, mx = np.floor(Q.min(axis=1)).astype(int), np.ceil(Q.max(axis=1)).astype(int)
    for q, (i0, j0), (i1, j1) in zip(Q, mn, mx):
        (ax_, ay_), (bx, by), (cx, cy) = q
        det = (bx - ax_) * (cy - ay_) - (cx - ax_) * (by - ay_)
        if abs(det) < 1e-9:
            continue
        X, Y = np.arange(i0, i1 + 1)[:, None], np.arange(j0, j1 + 1)[None, :]
        u = ((bx - X) * (cy - Y) - (cx - X) * (by - Y)) / det
        v = ((cx - X) * (ay_ - Y) - (ax_ - X) * (cy - Y)) / det
        G[i0:i1 + 1, j0:j1 + 1] |= (u >= 0) & (v >= 0) & (u + v <= 1)
    ii, jj = np.nonzero(G)
    c = lo + (np.array([ii.mean(), jj.mean()]) + 0.5) * cell
    return float(G.sum()) * cell * cell, (float(c[0]), float(c[1]))


def rim_gap(centre, radius, tris, ax):
    """Least distance, seen along the axis not in `ax`, from a disc's rim to the triangles' edges
    (negative: they overlap)."""
    P = tris[:, :, ax]
    A, B = P.reshape(-1, 2), P[:, [1, 2, 0], :].reshape(-1, 2)
    c, d = np.asarray(centre, float), B - A
    L2 = (d * d).sum(axis=1)
    t = np.clip(((c - A) * d).sum(axis=1) / np.where(L2 > 0.0, L2, 1.0), 0.0, 1.0)
    return float(np.sqrt((((A + t[:, None] * d) - c) ** 2).sum(axis=1)).min() - radius)


def check(spec, nodes, extra):
    S, Lay, A = spec["model"], spec["layout"], spec["actuators"]
    G = S["gates"]
    W = world_matrices(nodes)
    parts = {name: part for name, _, part, _, _, _ in nodes if part is not None}
    tri = {name: triangles(part, W[name]) for name, part in parts.items()}
    print("[babyshark] nodes:")
    for name, parent, part, _, _, _ in nodes:
        kind = "empty" if part is None else f"mesh, {sum(len(f) - 2 for f in part.faces)} triangles, " + ", ".join(sorted(set(part.mats)))
        print(f"[babyshark]   {name:<18} < {parent or '-':<10} {kind}")
    tris_n = sum(len(t) for t in tri.values())
    errs = []

    def gate(label, value, ref, tol):
        ok = abs(value - ref) <= tol
        print(f"[babyshark] gate {label:<34} {value:9.4f}  ({ref} +- {tol})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(label)

    def least(label, value, floor):
        ok = value >= floor
        print(f"[babyshark] gate {label:<34} {value:9.4f}  (at least {floor})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(label)

    # the wing
    wing, tail = extra["wing"], extra["tail"]
    wz = tri["wing"][:, :, 2]
    gate("wingspan", float(wz.max() - wz.min()), *G["span"])
    # the rotors: on their axes, the right size, turning the right way
    Rl, Rp = 0.5 * A["D_mr"], 0.5 * A["D_fw"]
    for k in LIFT:
        p, M, can = Lay["lift_rotors"][k], W["lift_" + k], extra["cans"][k].mean(axis=0)
        off = max(abs(M[0, 3] - p["pos"][0]), abs(M[2, 3] - p["pos"][2]), abs(can[0] - p["pos"][0]), abs(can[2] - p["pos"][2]))
        gate(f"lift_{k} node and motor off its axis", off, 0.0, G["lift_axis"])
        assert abs(M[1, 3] - p["pos"][1]) < 1e-9, ("lift node off its disc height", k)
        V = np.asarray(extra["blades"][k].verts)
        gate(f"lift_{k} tip circle", 2.0 * float(np.hypot(V[:, 0], V[:, 2]).max()), *G["lift_tip_circle"])
        sec = V[(V[:, 0] > 0.10) & (V[:, 0] < 0.14)]             # the blade on +X climbs toward its leading edge
        hi, lo = sec[np.argmax(sec[:, 1])], sec[np.argmin(sec[:, 1])]
        assert hi[2] * p["spin"] < 0.0 < lo[2] * p["spin"] and hi[1] - lo[1] > 0.002, ("lift blade pitched the wrong way", k)
    V = np.asarray(extra["blades"]["pusher"].verts)
    gate("pusher tip circle", 2.0 * float(np.hypot(V[:, 1], V[:, 2]).max()), *G["pusher_tip_circle"])
    sec = V[(V[:, 1] > 0.09) & (V[:, 1] < 0.13)]                 # the blade on +Y leads toward +Z, its leading edge forward
    fw, bk = sec[np.argmax(sec[:, 0])], sec[np.argmin(sec[:, 0])]
    assert fw[2] > 0.0 > bk[2] and fw[0] - bk[0] > 0.004, "pusher blade pitched the wrong way"
    assert np.allclose(W["pusher"][:3, 3], Lay["pusher"]["pos"])
    # the discs' clearances
    plan = (0, 2)
    lifting = np.concatenate([tri["wing"], tri["aileron_left"], tri["aileron_right"]])
    tailing = np.concatenate([tri["tail"], tri["ruddervator_left"], tri["ruddervator_right"]])
    cs = {k: np.array(Lay["lift_rotors"][k]["pos"], float) for k in LIFT}
    least("lift discs to the wing, from above", min(rim_gap(cs[k][[0, 2]], Rl, lifting, plan) for k in LIFT), G["clearance"])
    least("lift discs to the tail, from above", min(rim_gap(cs[k][[0, 2]], Rl, tailing, plan) for k in LIFT), G["clearance"])
    least("lift discs to each other", min(float(np.linalg.norm(cs[a] - cs[b])) for a in LIFT for b in LIFT if a < b) - 2.0 * Rl, G["clearance"])
    pod_gap = min(rim_gap(cs[k][[0, 2]], Rl, tri["fuselage"], plan) for k in LIFT)
    pp = np.array(Lay["pusher"]["pos"], float)
    least("pusher disc to the booms and tail", rim_gap(pp[[1, 2]], Rp, np.concatenate([tri["booms"], tailing]), (1, 2)), G["clearance"])
    gaps = []
    for k in ("rl", "rr"):                                       # along the line the two discs' planes share
        a2, b2 = Rl ** 2 - (pp[0] - cs[k][0]) ** 2, Rp ** 2 - (cs[k][1] - pp[1]) ** 2
        gaps.append(abs(cs[k][2] - pp[2]) - math.sqrt(max(a2, 0.0)) - math.sqrt(max(b2, 0.0)))
    least("pusher disc to the rear lift discs", min(gaps), G["clearance"])
    feet = np.array(Lay["gear"]["feet"], float)
    least("pusher disc above the feet's plane", pp[1] - Rp - float(feet[:, 1].max()), G["clearance"])
    # the tail on the layout
    V = np.asarray(parts["tail"].verts)
    for key, idx in sorted(extra["marks"].items()):
        ref = np.array(Lay["tail"]["panel_" + key.split()[0]], float) * (1.0, -1.0 if "port" in key else 1.0)
        gate(f"tail panel {key} (Y, Z)", float(np.abs(V[idx][1:] - ref).max()), 0.0, G["tail_ends"])
    assert len(extra["marks"]) == 4, extra["marks"]
    gate("tail leading edge X", float(V[:, 0].max()), Lay["tail"]["x_leading_edge"], G["tail_leading_edge"])
    centre = V[np.abs(V[:, 2]) < 1e-9]
    cy = float(centre[np.argmax(centre[:, 0])][1])              # the leading edge on the centreline
    # the feet
    fixed = np.concatenate([tri[k] for k in ("fuselage", "wing", "tail", "booms", "motors", "gear", "fittings")])
    lowest = float(fixed[:, :, 1].min())
    for i, pad in enumerate(extra["pads"]):
        under = np.array([pad[:, 0].mean(), pad[:, 1].min(), pad[:, 2].mean()])
        gate(f"foot {i + 1} underside off the layout", float(np.abs(under - feet[i]).max()), 0.0, G["feet"])
    assert lowest > float(feet[:, 1].min()) - 1e-6, ("something hangs below the feet", lowest)
    # the surfaces turn the way the nodes say
    turn = R("Z", math.degrees(0.3))[:3, :3]
    moved = {}
    for name in ("aileron_left", "aileron_right", "ruddervator_left", "ruddervator_right"):
        V = np.asarray(parts[name].verts)
        M = W[name]
        assert abs(M[0, 1]) < 1e-12 and M[2, 2] > 0.5, ("hinge node: local +Y has an X part, or +Z is not toward starboard", name)
        te = V[np.argsort(-np.hypot(V[:, 0], V[:, 1]))[:8]]       # the vertices farthest from the hinge
        d = ((te @ turn.T - te) @ M[:3, :3].T).mean(axis=0)
        if name.startswith("aileron"):
            moved[name] = float(d[1])
            assert d[1] < -0.005, ("aileron: +Z does not put the trailing edge down", name, d)
        else:
            n = tail.normal(-1.0 if name.endswith("left") else 1.0)
            moved[name] = float(d[1] * n[0] + d[2] * n[1])
            assert moved[name] < -0.005 and d[2] * (1.0 if name.endswith("left") else -1.0) > 0.0, ("ruddervator: +Z does not move the trailing edge into the tent", name, d)
    # every part closed and wound outward (each primitive was checked as it was added, by Part.add)
    vols = {}
    for name, part in parts.items():
        Vp = np.asarray(part.verts)
        ns = [newell(Vp, f) for f in part.faces]
        tot, mag = float(np.linalg.norm(sum(ns))), float(sum(np.linalg.norm(n) for n in ns))
        vols[name] = signed_volume(Vp, part.faces)
        assert tot <= 1e-9 + 1e-6 * mag and vols[name] > 0.0, ("part not closed or wound inward", name, tot, mag, vols[name])
    # what is printed, not gated
    cell = 0.002
    sec = wing.sec
    S_layout = 2.0 * float(sum(0.5 * (a[2] + b[2]) * (b[1] - a[1]) for a, b in zip(sec[:-1], sec[1:])))
    S_wing, c_wing = projected_area(lifting, 1, cell)
    S_tail, c_tail = projected_area(tailing, 1, cell)
    S_both, _ = projected_area(np.concatenate([lifting, tailing]), 1, cell)
    S_out = projected_area(np.concatenate([lifting, tailing, tri["fuselage"]]), 1, cell)[0] - projected_area(tri["fuselage"], 1, cell)[0]
    solid = [tri[k] for k in tri if not (k.startswith("lift_") or k == "pusher")]      # everything but the propeller blades
    solid += [triangles(extra["hubs"][k], W["lift_" + k]) for k in LIFT] + [triangles(extra["hubs"]["pusher"], W["pusher"])]
    solid = np.concatenate(solid)
    body = np.concatenate([tri[k] for k in tri if k != "fittings" and not (k.startswith("lift_") or k == "pusher")])
    length = float(solid[:, :, 0].max() - solid[:, :, 0].min())
    tail_top = float(tri["tail"][:, :, 1].max())
    pod = extra["pod"]
    print(f"[babyshark] gates' numbers: lift discs clear the pod by {pod_gap:.3f} m seen from above; the pusher's disc is "
          f"{float(tri['fuselage'][:, :, 0].min()) - pp[0]:.3f} m behind the pod's end and {float(lifting[:, :, 0].min()) - pp[0]:.3f} m behind the wing")
    print(f"[babyshark] tail: centre piece at Y {cy:.4f} on the centreline (layout {Lay['tail']['centre_y']}), top of the tail Y {tail_top:.4f}")
    print(f"[babyshark] surfaces, +0.3 rad about local +Z: ailerons' trailing edges {1000 * moved['aileron_left']:+.1f} and "
          f"{1000 * moved['aileron_right']:+.1f} mm in Y (down); ruddervators' {1000 * moved['ruddervator_left']:+.1f} and "
          f"{1000 * moved['ruddervator_right']:+.1f} mm along the panels' outer normals (toward the inner face)")
    print(f"[babyshark] wing planform {S_wing:.4f} m^2 on the mesh, centroid X {c_wing[0]:+.3f} (the layout's sections joined by straight "
          f"lines: {S_layout:.4f}); tail {S_tail:.4f}, centroid X {c_tail[0]:+.3f}; wing + tail {S_both:.4f}, {S_out:.4f} of it outside the pod (thesis S {spec['physical']['S']})")
    print(f"[babyshark] length {length:.4f} m with the pitot tube, {float(body[:, :, 0].max() - body[:, :, 0].min()):.4f} m without "
          f"(pod {pod.xn - pod.xe:.3f}); height {tail_top - lowest:.4f} m from the feet's underside (Y {lowest:.4f}) to the top of the tail; "
          f"pod {2.0 * max(pod.half_width(x) for x in pod.stations()):.3f} wide, {max(pod.top(x) - pod.bottom(x) for x in pod.stations()):.3f} high")
    print(f"[babyshark] triangles {tris_n}; every part closed and wound outward (Newell sum, signed volume): "
          + ", ".join(f"{k} {1e3 * v:.2f} l" for k, v in vols.items() if k in ("fuselage", "wing", "tail", "booms")))
    areas, cps = {}, {}
    for key, drop in (("front", 0), ("side", 2), ("top", 1)):
        areas[key], cps[key] = projected_area(solid, drop, cell)
    bl = spec["envelope"]["bluff"]
    print(f"[babyshark] projected areas without the blades, {1000 * cell:.0f} mm grid: front {areas['front']:.4f}, side {areas['side']:.4f}, "
          f"top {areas['top']:.4f} m^2 (spec: front {bl['area_front']}, side {bl['area_side']}, top {bl['area_top']})")
    print(f"[babyshark] their centroids: front (Y {cps['front'][0]:+.3f}, Z {cps['front'][1]:+.3f}), side (X {cps['side'][0]:+.3f}, "
          f"Y {cps['side'][1]:+.3f}), top (X {cps['top'][0]:+.3f}, Z {cps['top'][1]:+.3f})")
    assert not errs, ("gates failed", errs)
    return areas, tris_n


def write_areas(spec_path, areas):
    """The three measured numbers into the spec's envelope.bluff, in place: nothing else of the file is touched."""
    with open(spec_path, "r", encoding="utf-8", newline="") as fh:
        txt = fh.read()
    new = txt
    for key in ("front", "side", "top"):
        new, n = re.subn(r'"area_%s": [0-9.]+' % key, '"area_%s": %.4f' % (key, areas[key]), new, count=1)
        assert n == 1, f"area_{key} not found in the spec"
    if new != txt:
        with open(spec_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    print(f"[babyshark] projected areas written to {os.path.basename(spec_path)}")


# ---------------------------------------------------------------- Blender side
def build_blender(spec, nodes):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    obj = {}
    for name, parent, part, M, props, sharp in nodes:
        obj[name] = new_object(name, part, mats, obj.get(parent), M, props=props, sharp_deg=sharp)
    return obj["babyshark"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="babyshark_spec.json")
    ap.add_argument("--out", default="babyshark.glb")
    ap.add_argument("--check", action="store_true", help="build the geometry and print the checks, no Blender")
    ap.add_argument("--write-areas", action="store_true", help="write the measured projected areas into the spec")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    nodes, extra = build_nodes(spec)
    used = set(m for _, _, p, _, _, _ in nodes if p is not None for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    areas, _ = check(spec, nodes, extra)
    if a.write_areas:
        write_areas(path(a.spec), areas)
    if a.check:
        return
    build_blender(spec, nodes)
    tris_n = export(path(a.out))
    print(f"[babyshark] wrote {path(a.out)}  triangles={tris_n}  bytes={os.path.getsize(path(a.out))}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

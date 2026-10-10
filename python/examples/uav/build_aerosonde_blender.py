"""Build the Aerosonde from aerosonde_spec.json: aerosonde.glb (via Blender).

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_aerosonde_blender.py -- --spec aerosonde_spec.json --out aerosonde.glb

Geometry only (plain Python + numpy, no Blender; prints the checks and fails on a gate):
    python build_aerosonde_blender.py --check

The mesh accumulator, the primitives and the Blender export are build_common's (../build_common.py,
shared by every generator); this script adds the aircraft. All geometry is built with numpy in the
spec's model frame (X forward, Y up, Z starboard, metres, origin at the centre of gravity) and
converted to Blender's Z-up only when a mesh or a transform is handed to bpy. Every number is the
spec's: `physical` for the wing's span, area and mean chord and `propeller.D` for the propeller's
diameter (the textbook's, met exactly), `geometry` for everything else (ASSUMED, fitted to one
photograph as described in words and to the quoted length and height), `materials`.

The wing is one closed loft from tip to tip over the pod: a section at each station between the
spec's sections; where an aileron is, the section stops at the hinge line and the aileron is the
rest of it, lofted on its own node. The tail is one closed loft along the inverted V, from one boom
over the rounded apex to the other, cut the same way for the ruddervators.

Node hierarchy (all names exact):
    aerosonde                    root, identity, origin = the centre of gravity
      airframe                   an empty; the fixed meshes are its children:
        fuselage, wing, tail, booms, engine, fittings
      aileron_left, aileron_right
                                 on the hinge lines: local +Z along the hinge toward starboard; a positive
                                 turn about local +Z puts the trailing edge down on either side
      ruddervator_left, ruddervator_right
                                 on the tail panels' hinge lines: local +Z along the hinge toward starboard
                                 (down the starboard panel, up the port one); a positive turn about local
                                 +Z moves the trailing edge toward the panel's lower, inner face
      propeller                  at the disc centre, spins about local +X (positive: clockwise seen from
                                 behind); two blades and the spinner over their hub
      imu, gnss, pitot_tip, camera_nadir, camera_fpv        sensor empties

A hinge node's local +Y has no X part, so its rest rotation has no turn about its own +Z and a
scene may set rotation.z alone. The cameras look down local -Z with local +Y up (threepp camera
convention).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # examples/ (build_common)
from build_common import (Part, R, T, Z_TO_Y, box, export, frame_from_forward_up, frame_y_toward, loft,  # noqa: E402
                          make_materials, new_object, newell, revolve, signed_volume, strut, table)

I4 = np.eye(4)
SURFACES = ("aileron_left", "aileron_right", "ruddervator_left", "ruddervator_right")


# ---------------------------------------------------------------- primitives
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def toward(origin, d):
    """Placement that puts a solid of revolution's +Z along d at origin."""
    return T(*origin) @ frame_y_toward(unit(d)) @ Z_TO_Y()


def spun(part, profile, mat, origin, d, n=32):
    v, f = revolve(profile, n)
    part.add(v, f, mat, toward(origin, d))


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


def fillet(points, r, seg=6):
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
    """geometry.wing: the planform through its sections (straight lines between them), the dihedral
    and the section over it, as functions of the distance z from the centreline."""

    N_CHORD = 22                      # chordwise intervals per surface

    def __init__(self, spec):
        W = spec["geometry"]["wing"]
        sec = np.asarray(W["sections"], float)                   # [X of the leading edge, Z, chord]
        self.sec, self.y, self.zt = sec, W["y"], float(sec[-1, 1])
        self.tr, self.m, self.p = W["thickness_ratio"], W["camber"], W["camber_at"]
        self.tip_round = W["tip_round"]
        self.d_from, self.d_slope = W["dihedral_from_z"], math.tan(math.radians(W["dihedral_deg"]))
        A = W["aileron"]
        (self.a_z0, self.a_z1), self.hinge, self.gap = A["z"], A["hinge_chord_fraction"], A["gap"]

    def x_le(self, z):
        return float(np.interp(abs(z), self.sec[:, 1], self.sec[:, 0]))

    def chord(self, z):
        return float(np.interp(abs(z), self.sec[:, 1], self.sec[:, 2]))

    def rise(self, z):
        return max(0.0, abs(z) - self.d_from) * self.d_slope

    def tip_factor(self, z):
        u = (z - (self.zt - self.tip_round)) / self.tip_round
        return 1.0 if u <= 0.0 else max(0.16, math.sqrt(max(0.0, 1.0 - u * u)))

    def surface(self, z, x, upper, k=1.0):
        """(X, Y) of the upper or lower surface at the chord fraction x; k scales the thickness."""
        c = self.chord(z)
        yt = thickness(x) * self.tr * c * k
        return self.x_le(z) - x * c, self.y + self.rise(z) + camber(x, self.m, self.p) * c + (yt if upper else -yt)

    def stations(self):
        """[(z, cut at the hinge?)] from the centreline to the tip."""
        zt, tr = self.zt, self.tip_round
        z0c, z1c = round(self.a_z0 - self.gap, 6), round(self.a_z1 + self.gap, 6)
        zs = set(np.round(np.arange(0.0, zt - tr, 0.07), 6).tolist()) | set(np.round(self.sec[:, 1], 6).tolist())
        cut = set(np.round(np.linspace(z0c, z1c, 9), 6).tolist())
        zs = {z for z in zs if not (z0c - 0.004 <= z <= z1c + 0.004) and z < zt - tr - 0.004}
        zs |= {round(z0c - 0.003, 6), round(z1c + 0.003, 6)}
        zs |= {round(zt - tr * (1.0 - math.sin(0.5 * math.pi * k / 6)), 6) for k in range(7)}
        return [(z, z in cut) for z in sorted(zs | cut)]

    def ring(self, z, cut, side):
        xs = chord_points(0.0, self.hinge if cut else 1.0, self.N_CHORD)
        k = self.tip_factor(z)
        return [self.surface(z, x, up, k) + (side * z,) for x, up in around(xs)]


def build_wing(wing):
    st = wing.stations()
    rings = [wing.ring(z, c, -1.0) for (z, c) in reversed(st[1:])] + [wing.ring(z, c, 1.0) for (z, c) in st]
    part = Part()
    v, f = loft(rings)
    part.add(v, f, "wing_white")
    return part


def build_aileron(wing, side):
    """The aileron on one side: the mesh in its node's frame, and the node's matrix."""
    n = wing.N_CHORD // 2
    rings, hinge = [], []
    for z in np.linspace(wing.a_z0, wing.a_z1, 9):
        rings.append([wing.surface(z, x, up) + (side * z,) for x, up in around(chord_points(wing.hinge, 1.0, n), True)])
        hinge.append(wing.surface(z, wing.hinge, True, k=0.0) + (side * z,))          # on the camber line
    a, b = (hinge[0], hinge[-1]) if side > 0 else (hinge[-1], hinge[0])              # toward starboard
    M = hinge_frame(a, b)
    part = Part()
    v, f = loft(rings)
    part.add(v, f, "wing_white", np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the tail
class Tail:
    """geometry.tail: the inverted V as a path of its leading edge in (Y, Z), the section across it."""

    N_CHORD = 14

    def __init__(self, spec):
        L = spec["geometry"]["tail"]
        self.L = L
        self.x_le, self.c, self.inc = L["x_leading_edge"], L["chord"], math.radians(L["incidence_deg"])
        self.foot, self.top = np.array(L["panel_foot"], float), np.array([L["apex_y"], 0.0])       # (Y, Z), starboard
        self.length = float(np.linalg.norm(self.top - self.foot))
        self.hinge, self.tr = L["hinge_chord_fraction"], L["thickness_ratio"]
        self.rv0, self.rv1 = L["ruddervator_span"]

    def at(self, f, side=1.0):
        """(Y, Z) of the leading edge the fraction f up the panel from its foot."""
        return (self.foot + f * (self.top - self.foot)) * (1.0, side)

    def normal(self, side=1.0):
        """The panel's outer normal in (Y, Z): up and outboard."""
        t = unit(self.top - self.foot)
        return np.array([-t[1], t[0]]) * (1.0, side)

    def path(self):
        """[((Y, Z), cut at the hinge?)] from the port panel's end in its boom over the rounded apex to
        the starboard panel's."""
        L = self.L
        g, e = L["gap"] / self.length, 0.003 / self.length
        f0, f1 = self.rv0 - g, self.rv1 + g
        fs = [(-L["foot_extension"] / self.length, False), (0.0, False), (f0 - e, False)]
        fs += [(f, True) for f in np.linspace(f0, f1, 9)] + [(f1 + e, False)]
        half = [(self.at(f), c) for f, c in fs]                   # the starboard panel, foot to its last section
        last = half[-1][0]
        arc = fillet([last * (1.0, -1.0), self.top, last], L["apex_radius"])[1:-1]
        port = [(p * (1.0, -1.0), c) for p, c in half]
        return port + [(p, False) for p in arc] + half[::-1]

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
    path = tail.path()
    st = mitred([p for p, _ in path])
    rings = [tail.ring(p, n, k, chord_points(0.0, tail.hinge if cut else 1.0, tail.N_CHORD)) for (p, n, k), (_, cut) in zip(st, path)]
    part = Part()
    v, f = loft(rings)
    part.add(v, f, "tail_black")
    return part


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
    part.add(v, f, "tail_black", np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the pod
class Pod:
    """geometry.pod: a body of revolution about the propeller's axis, its radius over X."""

    def __init__(self, spec):
        F = spec["geometry"]["pod"]
        self.F = F
        self.xn, self.xe, self.y = F["nose_x"], F["end_x"], F["axis_y"]
        self._r = table(F["radius"])

    def radius(self, x):
        a, d = self.F["nose_length"], self.xn - x
        return self._r(x) * (1.0 if d >= a else math.sqrt(max(0.0, 1.0 - ((a - d) / a) ** 2)))

    def top(self, x):
        return self.y + self.radius(x)

    def bottom(self, x):
        return self.y - self.radius(x)

    def ring(self, x, n=44):
        r = self.radius(x)
        return [(x, self.y + r * math.sin(2.0 * math.pi * k / n), r * math.cos(2.0 * math.pi * k / n)) for k in range(n)]

    def stations(self):
        """X from the nose to the end, the colour split among them."""
        a = self.F["nose_length"]
        xs = [self.xn - a * (1.0 - math.cos(q)) for q in [0.035, 0.09, 0.17] + np.linspace(0.26, 0.5 * math.pi, 14).tolist()]
        xs += np.linspace(self.xn - a, self.xe, 22)[1:].tolist()
        s = self.F["split_x"]
        return sorted({round(x, 6) for x in xs if abs(x - s) > 0.004} | {s}, reverse=True)


def build_fuselage(spec, pod, wing):
    """The pod in its two colours, and the pylon the wing stands on."""
    F = spec["geometry"]["pod"]
    part = Part()
    xs = pod.stations()
    rings = [pod.ring(x) for x in xs]
    v, f = loft(rings)
    N = len(rings[0])
    mats = []
    for a, b in zip(xs[:-1], xs[1:]):
        mats += ["pod_orange" if 0.5 * (a + b) > F["split_x"] else "pod_white"] * N
    part.add_multi(v, f, mats + ["pod_orange", "pod_white"])                # the two caps: the nose's tip, the end
    # the pylon: under the wing's chord, then down from the trailing edge to the pod's end; a point at each end
    P = F["pylon"]
    c, xl = wing.chord(0.0), wing.x_le(0.0)
    xa, xt, xb = xl - 0.03 * c, xl - c, pod.xe + P["end_margin"]
    y_te = wing.surface(0.0, 1.0, False)[1]
    rings = []
    for t in np.linspace(0.0, 1.0, 29):
        X = xa + (xb - xa) * t
        if X >= xt:
            u = (xl - X) / c
            lo, up = wing.surface(0.0, u, False)[1], wing.surface(0.0, u, True)[1]
            top = min(lo + 0.003, 0.5 * (lo + up))
        else:
            top = y_te + (pod.top(xb) + 0.001 - y_te) * (xt - X) / (xt - xb)
        w = max(0.0015, P["half_width"] * math.sin(math.pi * t) ** 0.6)
        r = pod.radius(X)
        bot = min(pod.y + math.sqrt(max(r * r - w * w, 0.0)) - 0.004, top - 0.004)       # sunk into the pod's back
        rings.append([(X, 0.5 * (top + bot) + y, z) for (y, z) in superellipse(w, 0.5 * (top - bot), 0.5 * (top - bot), 20, P["exponent"], P["exponent"])])
    v, f = loft(rings)
    part.add(v, f, "pod_white")
    return part


# ---------------------------------------------------------------- the booms
def build_booms(spec, wing):
    """The two tubes with their round ends, and the saddle between each and the wing."""
    B = spec["geometry"]["boom"]
    part = Part()
    (x0, x1), r, y = B["x"], B["radius"], B["y"]
    L = x0 - x1
    q = [0.5 * math.pi * k / 5 for k in range(6)]
    cap = [(r * math.sin(a), r * (1.0 - math.cos(a))) for a in q]                      # a round end over one radius
    prof = cap + [(rr, L - d) for (rr, d) in reversed(cap)]
    for side in (1.0, -1.0):
        z = side * B["z"]
        spun(part, prof, "carbon", (x0, y, z), (-1, 0, 0), 20)
        rings = []
        for u in np.linspace(0.16, 0.97, 15):
            X, lo = wing.surface(B["z"], u, False)
            _, up = wing.surface(B["z"], u, True)
            top, bot = min(lo + 0.003, 0.5 * (lo + up)), y
            hw = max(0.003, B["saddle_half_width"] * max(0.0, 1.0 - (2.0 * u - 1.0) ** 2) ** 0.3)
            rings.append([(X, 0.5 * (top + bot) + yy, z + zz) for (yy, zz) in superellipse(hw, 0.5 * (top - bot), 0.5 * (top - bot), 12, 3.0, 3.0)])
        v, f = loft(rings)
        part.add(v, f, "carbon")
    return part


# ---------------------------------------------------------------- the engine
def build_engine(spec, pod):
    """A single-cylinder engine on the pod's end: mount, crankcase, bearing housing, shaft, an upright
    finned cylinder with its head and rocker box, an intake stub, an exhaust pipe and its silencer."""
    G = spec["geometry"]
    E, P = G["engine"], G["propeller"]
    part = Part()
    y, aft, up = pod.y, (-1, 0, 0), (0, 1, 0)
    mr, mt = E["mount"]["radius"], E["mount"]["thickness"]
    (c0, c1), rc, hx, rh = E["crankcase_x"], E["crankcase_radius"], E["housing_x"], E["housing_radius"]
    assert abs(pod.xe - mt - c0) < 1e-9, "the mount plate does not reach the crankcase"
    spun(part, [(mr, 0.0), (mr, mt - 0.001), (mr - 0.001, mt)], "engine_alu", (pod.xe + 0.001, y, 0.0), aft, 36)
    L, Lh = c0 - c1, c0 - hx
    spun(part, [(rc - 0.004, -0.001), (rc, 0.003), (rc, L - 0.008), (rc - 0.006, L - 0.001), (rh + 0.004, L + 0.003), (rh, L + 0.008),
                (rh, Lh - 0.002), (rh - 0.002, Lh)], "engine_alu", (c0, y, 0.0), aft, 36)
    hub0 = P["disc_x"] + P["spinner_x"][0]                       # the spinner's forward face
    spun(part, [(E["shaft_radius"], 0.0), (E["shaft_radius"], hx - 0.001 - hub0 + 0.002)], "engine_steel", (hx - 0.001, y, 0.0), aft, 16)
    C = E["cylinder"]
    (y0, y1), rb, rf, nf, ft = C["y"], C["barrel_radius"], C["fin_radius"], C["fins"], C["fin_thickness"]
    pitch = (y1 - y0 - ft) / (nf - 1)
    prof = [(rb + 0.004, y0 - 0.012), (rb + 0.004, y0 - 0.004), (rb, y0 - 0.002)]
    for i in range(nf):
        a = y0 + i * pitch
        prof += [(rb, a), (rf, a), (rf, a + ft), (rb, a + ft)]
    hr, hy = C["head_radius"], C["head_y"]
    prof += [(rb, y1 + 0.002), (hr, y1 + 0.002), (hr, hy - 0.002), (hr - 0.002, hy)]
    spun(part, prof, "engine_alu", (C["x"], y, 0.0), up, 36)
    bx, by, bz = C["rocker_box"]
    v, f = box(bx, by, bz)
    part.add(v, f, "engine_alu", T(C["x"], y + hy + 0.5 * by - 0.001, 0.0))
    I = E["intake"]
    strut(part, I["from"], I["to"], I["radius"], "engine_alu", 20)
    X = E["exhaust"]
    strut(part, X["pipe"][0], X["pipe"][1], X["pipe_radius"], "engine_dark", 16)
    s0, s1 = (np.array(p, float) for p in X["silencer"])
    rs, Ls = X["silencer_radius"], float(np.linalg.norm(s1 - s0))
    spun(part, [(0.5 * rs, 0.0), (rs, 0.004), (rs, Ls - 0.010), (0.45 * rs, Ls - 0.004), (0.3 * rs, Ls)], "engine_dark", s0, s1 - s0, 24)
    return part


# ---------------------------------------------------------------- the fittings
def look_frame(c):
    return T(*c["position"]) @ frame_from_forward_up(unit(c["look"]), c["up"])


def build_fittings(spec, pod):
    G = spec["geometry"]
    SE = G["sensors"]
    F = Part()
    pt = G["pitot"]                                              # out of the nose, on its axis
    x0, x1 = pt["x"]
    spun(F, [(pt["radius"], 0.0), (pt["radius"], x1 - x0 - 0.006), (0.55 * pt["radius"], x1 - x0)], "pitot_steel", (x0, pod.y, 0.0), (1, 0, 0), 12)
    assert abs(SE["pitot_tip"]["position"][0] - x1) < 1e-9, "the pitot's empty is not at its tip"
    gn = G["gnss"]                                               # a puck on the pod's back
    gp = SE["gnss"]["position"]
    assert gp[1] - gn["height"] - 0.002 < pod.top(gn["x"]) <= gp[1], ("GNSS puck off the pod's back", pod.top(gn["x"]))
    r, h = gn["radius"], gn["height"]
    spun(F, [(r, -h - 0.008), (r, -0.004), (r - 0.004, 0.0), (0.0, 0.0)], "trim_black", gp, (0, 1, 0), 28)
    cn = SE["camera_nadir"]["position"]                          # the windows: in the belly, in the nose's underside
    assert abs(pod.bottom(cn[0]) - cn[1]) < 0.002, ("nadir window off the belly", pod.bottom(cn[0]))
    lens(F, cn, (0, -1, 0), 0.022, 0.017, 0.003)
    cf = SE["camera_fpv"]["position"]
    assert abs(pod.bottom(cf[0]) - cf[1]) < 0.002, ("forward window off the nose", pod.bottom(cf[0]))
    dx = 0.0005
    tangent = unit((2.0 * dx, pod.bottom(cf[0] + dx) - pod.bottom(cf[0] - dx), 0.0))
    lens(F, cf, (tangent[1], -tangent[0], 0.0), 0.0125, 0.009, 0.0025)
    return F


# ---------------------------------------------------------------- the propeller
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


def build_propeller(spec):
    """(spinner, blades) in the propeller node's frame: the disc centre at the origin, the axis local
    +X, the blades along local +-Y. A positive turn about +X moves the blade on +Y toward +Z (clockwise
    seen from behind); it carries its leading edge there, ahead of its trailing edge: thrust forward."""
    P = spec["geometry"]["propeller"]
    hub, blades = Part(), Part()
    rings = [[(u * math.sin(b) + w * math.cos(b), r, u * math.cos(b) - w * math.sin(b)) for (u, w) in sec]
             for (r, b, sec) in blade_sections(P, 0.5 * spec["propeller"]["D"])]
    v, f = loft(rings)
    for ang in (0.0, 180.0):
        blades.add(v, f, "prop_black", R("X", ang))
    (h0, h1), rs = P["spinner_x"], P["spinner_radius"]
    flat = 2.0 * h0                                              # the cylinder over the blades' roots, then the ogive
    ogive = [(rs * math.cos(q), flat + (h0 - h1 - flat) * math.sin(q)) for q in np.linspace(0.0, 0.5 * math.pi, 9)[1:-1]]
    spun(hub, [(rs - 0.003, 0.0), (rs, 0.003), (rs, flat)] + ogive + [(0.0, h0 - h1)], "spinner_alu", (h0, 0, 0), (-1, 0, 0), 32)
    return hub, blades


# ---------------------------------------------------------------- the nodes
def build_nodes(spec):
    """[(name, parent, part or None, local matrix, props, sharp angle)] in creation order, and what the
    checks need."""
    G = spec["geometry"]
    wing, tail, pod = Wing(spec), Tail(spec), Pod(spec)
    nodes = [("aerosonde", None, None, I4, {
        "frame": "X forward, Y up, Z starboard; origin = the centre of gravity (assumed, geometry.cg)",
        "spec": "aerosonde_spec.json", "body_axes": "body (x fwd, y right, z down) = (X, Z, -Y)",
        "mass": spec["physical"]["mass"], "model": "Beard and McLain, Small Unmanned Aircraft (2012), the Aerosonde"}, 32.0)]
    nodes.append(("airframe", "aerosonde", None, I4, {}, 32.0))
    for name, part, sharp in (("fuselage", build_fuselage(spec, pod, wing), 40.0), ("wing", build_wing(wing), 40.0), ("tail", build_tail(tail), 40.0),
                              ("booms", build_booms(spec, wing), 35.0), ("engine", build_engine(spec, pod), 32.0),
                              ("fittings", build_fittings(spec, pod), 32.0)):
        nodes.append((name, "airframe", part, I4, {}, sharp))
    for name, side in (("aileron_left", -1.0), ("aileron_right", 1.0)):
        part, M = build_aileron(wing, side)
        nodes.append((name, "aerosonde", part, M, {"axis": "+Z (hinge, toward starboard)", "positive": "trailing edge down"}, 40.0))
    for name, side in (("ruddervator_left", -1.0), ("ruddervator_right", 1.0)):
        part, M = build_ruddervator(tail, side)
        nodes.append((name, "aerosonde", part, M, {"axis": "+Z (hinge, toward starboard)",
                                                    "positive": "trailing edge toward the panel's lower, inner face"}, 40.0))
    hub, blades = build_propeller(spec)
    centre = (G["propeller"]["disc_x"], pod.y, 0.0)
    nodes.append(("propeller", "aerosonde", merged(hub, blades), T(*centre),
                  {"axis": "+X", "positive": "spin vector along body +x, clockwise seen from behind", "diameter": spec["propeller"]["D"]}, 32.0))
    SE = G["sensors"]
    for k in ("imu", "gnss", "pitot_tip"):
        nodes.append((k, "aerosonde", None, T(*SE[k]["position"]), {}, 32.0))
    for k in ("camera_nadir", "camera_fpv"):
        nodes.append((k, "aerosonde", None, look_frame(SE[k]), {"convention": "look -Z, up +Y", "assumed": SE[k]["note"]}, 32.0))
    return nodes, {"wing": wing, "tail": tail, "pod": pod, "hub": hub, "blades": blades, "centre": np.array(centre)}


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
    """Area of the union of the triangles seen along axis `drop`: the cell centres of a grid that are
    inside any of them."""
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
    return float(G.sum()) * cell * cell


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
    G, Ph = spec["geometry"], spec["physical"]
    Ga, D = G["gates"], spec["propeller"]["D"]
    W = world_matrices(nodes)
    parts = {name: part for name, _, part, _, _, _ in nodes if part is not None}
    tri = {name: triangles(part, W[name]) for name, part in parts.items()}
    print("[aerosonde] nodes:")
    for name, parent, part, _, _, _ in nodes:
        kind = "empty" if part is None else f"mesh, {sum(len(f) - 2 for f in part.faces)} triangles, " + ", ".join(sorted(set(part.mats)))
        print(f"[aerosonde]   {name:<18} < {parent or '-':<10} {kind}")
    tris_n = sum(len(t) for t in tri.values())
    errs = []

    def gate(label, value, ref, tol):
        ok = abs(value - ref) <= tol
        print(f"[aerosonde] gate {label:<40} {value:9.4f}  ({ref} +- {tol})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(label)

    def least(label, value, floor):
        ok = value >= floor
        print(f"[aerosonde] gate {label:<40} {value:9.4f}  (at least {floor})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(label)

    # the wing against the book's b, S and c
    wing, tail, pod = extra["wing"], extra["tail"], extra["pod"]
    lifting = np.concatenate([tri["wing"], tri["aileron_left"], tri["aileron_right"]])
    tailing = np.concatenate([tri["tail"], tri["ruddervator_left"], tri["ruddervator_right"]])
    span = float(lifting[:, :, 2].max() - lifting[:, :, 2].min())
    S_wing = projected_area(lifting, 1, 0.001)
    gate("wingspan b", span, Ph["b"], Ga["span"])
    gate("wing planform area S, from above", S_wing, Ph["S"], Ga["area"])
    gate("wing mean chord S / b", S_wing / span, Ph["c"], Ga["mean_chord"])
    sec = wing.sec
    S_sections = 2.0 * float(sum(0.5 * (a[2] + b[2]) * (b[1] - a[1]) for a, b in zip(sec[:-1], sec[1:])))
    # the origin where geometry.cg puts it: on the wing's quarter-chord line and on the propeller's axis
    cgf = G["cg"]["wing_chord_fraction"]
    assert all(abs(x - cgf * c) < 1e-4 for x, _, c in sec), "the origin is not on the wing's chord fraction of geometry.cg"
    assert abs(pod.y) < 1e-9 and abs(extra["centre"][1]) < 1e-9 and abs(extra["centre"][2]) < 1e-9, "the origin is not on the propeller's axis"
    # the propeller: its size, its pitch, its clearances
    V = np.asarray(extra["blades"].verts)
    gate("propeller tip circle D", 2.0 * float(np.hypot(V[:, 1], V[:, 2]).max()), D, Ga["tip_circle"])
    band = V[(V[:, 1] > 0.09) & (V[:, 1] < 0.13)]                # the blade on +Y leads toward +Z, its leading edge forward
    fw, bk = band[np.argmax(band[:, 0])], band[np.argmin(band[:, 0])]
    assert fw[2] > 0.0 > bk[2] and fw[0] - bk[0] > 0.004, "propeller blade pitched the wrong way"
    pp = extra["centre"]
    assert np.allclose(W["propeller"][:3, 3], pp) and np.allclose(W["propeller"][:3, :3], np.eye(3))
    Rp = 0.5 * D
    least("propeller disc to the booms, along X", rim_gap(pp[[1, 2]], Rp, tri["booms"], (1, 2)), Ga["clearance"])
    least("propeller disc to the tail, along X", rim_gap(pp[[1, 2]], Rp, tailing, (1, 2)), Ga["clearance"])
    blade_front, blade_back = float(V[:, 0].max() + pp[0]), float(V[:, 0].min() + pp[0])
    least("blades to the pod and the wing, ahead", float(min(tri["fuselage"][:, :, 0].min(), lifting[:, :, 0].min())) - blade_front, Ga["axial"])
    ev = tri["engine"].reshape(-1, 3)
    outside = ev[np.hypot(ev[:, 1] - pp[1], ev[:, 2] - pp[2]) > G["propeller"]["spinner_radius"]]
    least("blades to the engine, ahead", float(outside[:, 0].min()) - blade_front, Ga["axial"])
    tail_gap = blade_back - float(tailing[:, :, 0].max())
    assert tail_gap > 0.3, ("the tail is not behind the propeller", tail_gap)
    # the surfaces turn the way the nodes say
    turn = R("Z", math.degrees(0.3))[:3, :3]
    moved = {}
    for name in SURFACES:
        V = np.asarray(parts[name].verts)
        M = W[name]
        assert abs(M[0, 1]) < 1e-12 and M[2, 2] > 0.5, ("hinge node: local +Y has an X part, or +Z is not toward starboard", name)
        assert np.abs(V[:, 0]).min() < 1e-3 and np.abs(V[:, 1]).max() < 0.02, ("surface not on its hinge line", name)
        te = V[np.argsort(-np.hypot(V[:, 0], V[:, 1]))[:8]]       # the vertices farthest from the hinge
        d = ((te @ turn.T - te) @ M[:3, :3].T).mean(axis=0)
        if name.startswith("aileron"):
            moved[name] = float(d[1])
            assert d[1] < -0.005, ("aileron: +Z does not put the trailing edge down", name, d)
        else:
            s = -1.0 if name.endswith("left") else 1.0
            n = tail.normal(s)
            moved[name] = float(d[1] * n[0] + d[2] * n[1])
            assert moved[name] < -0.005 and d[1] < 0.0 and d[2] * s < 0.0, ("ruddervator: +Z does not move the trailing edge to the lower, inner face", name, d)
    # every part closed and wound outward (each primitive was checked as it was added, by Part.add)
    vols = {}
    for name, part in parts.items():
        Vp = np.asarray(part.verts)
        ns = [newell(Vp, f) for f in part.faces]
        tot, mag = float(np.linalg.norm(sum(ns))), float(sum(np.linalg.norm(n) for n in ns))
        vols[name] = signed_volume(Vp, part.faces)
        assert tot <= 1e-9 + 1e-6 * mag and vols[name] > 0.0, ("part not closed or wound inward", name, tot, mag, vols[name])
    # length and height
    body = np.concatenate([tri[k] for k in tri if k not in ("fittings", "propeller")])
    length = float(body[:, :, 0].max() - body[:, :, 0].min())
    belly, tail_top = float(tri["fuselage"][:, :, 1].min()), float(tailing[:, :, 1].max())
    everything = np.concatenate(list(tri.values()))
    gate("length, nose to the booms' ends", length, *Ga["length"])
    gate("height, belly to the tail's top", tail_top - belly, *Ga["height"])
    # what is printed, not gated
    S_tail = projected_area(tailing, 1, 0.001)
    tb = math.degrees(math.atan2(tail.top[0] - tail.foot[0], tail.foot[1]))
    print(f"[aerosonde] wing: span {span:.4f} m, planform {S_wing:.4f} m^2 on the mesh (the sections' straight-line area {S_sections:.4f}, "
          f"book S {Ph['S']}), mean chord {S_wing / span:.5f} m (book c {Ph['c']}), aspect ratio {span * span / S_wing:.2f}")
    print(f"[aerosonde] propeller: disc at X {pp[0]:.3f}, blades from X {blade_back:.3f} to {blade_front:.3f}; the tail's leading edge is "
          f"{tail_gap:.3f} m behind them; tip to the boom's axis {math.hypot(G['boom']['y'] - pp[1], G['boom']['z']) - Rp:.3f} m")
    print(f"[aerosonde] tail: panels {tail.length:.3f} m long at {tb:.1f} deg from the horizontal, {S_tail:.4f} m^2 seen from above, "
          f"top at Y {tail_top:.4f}")
    print(f"[aerosonde] surfaces, +0.3 rad about local +Z: ailerons' trailing edges {1000 * moved['aileron_left']:+.1f} and "
          f"{1000 * moved['aileron_right']:+.1f} mm in Y (down); ruddervators' {1000 * moved['ruddervator_left']:+.1f} and "
          f"{1000 * moved['ruddervator_right']:+.1f} mm along the panels' outer normals (toward the lower, inner face)")
    print(f"[aerosonde] length {length:.4f} m nose to the booms' ends, {float(everything[:, :, 0].max() - everything[:, :, 0].min()):.4f} m with the pitot tube "
          f"(pod {pod.xn - pod.xe:.3f} m long, {2.0 * max(pod.radius(x) for x in pod.stations()):.3f} m in diameter); height {tail_top - belly:.4f} m "
          f"from the belly (Y {belly:.4f}) to the tail's top, {tail_top - (pp[1] - Rp):.4f} m with a blade straight down")
    print(f"[aerosonde] triangles {tris_n}; every part closed and wound outward (Newell sum, signed volume): "
          + ", ".join(f"{k} {1e3 * v:.2f} l" for k, v in vols.items() if k in ("fuselage", "wing", "tail", "booms", "engine")))
    assert not errs, ("gates failed", errs)
    return tris_n


# ---------------------------------------------------------------- Blender side
def build_blender(spec, nodes):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    obj = {}
    for name, parent, part, M, props, sharp in nodes:
        obj[name] = new_object(name, part, mats, obj.get(parent), M, props=props, sharp_deg=sharp)
    return obj["aerosonde"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="aerosonde_spec.json")
    ap.add_argument("--out", default="aerosonde.glb")
    ap.add_argument("--check", action="store_true", help="build the geometry and print the checks, no Blender")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    nodes, extra = build_nodes(spec)
    used = set(m for _, _, p, _, _, _ in nodes if p is not None for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    check(spec, nodes, extra)
    if a.check:
        return
    build_blender(spec, nodes)
    tris_n = export(path(a.out))
    print(f"[aerosonde] wrote {os.path.basename(path(a.out))}  triangles={tris_n}  bytes={os.path.getsize(path(a.out))}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

"""Build the HobbyKing Bixler 3 from bixler3_spec.json: bixler3.glb (via Blender).

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_bixler3_blender.py -- --spec bixler3_spec.json --out bixler3.glb

Geometry only (plain Python + numpy, no Blender; prints the checks and fails on a gate):
    python build_bixler3_blender.py --check

The mesh accumulator, the primitives and the Blender export are build_common's (../build_common.py,
shared by every generator); this script adds the aircraft. All geometry is built with numpy in the
model frame (X forward, Y up, Z starboard, metres, origin at the centre of gravity) and converted to
Blender's Z-up only when a mesh or a transform is handed to bpy. Every number is the spec's:
`physical` for the wing's span, area and mean aerodynamic chord, `thesis_vlm.geometry` for where
the centre of gravity is under the wing and for the incidences, `geometry` for the shapes (the
vendor's length and span over the tips, the rest fitted by eye and marked ASSUMED), `materials`.

The wing is one closed loft from tip to tip through the pod: a Clark Y section at each station, set
at the thesis's incidence and raised along the dihedral line, which turns up toward the tips; where
an aileron is, the section stops at the hinge line and the aileron is the rest of it, lofted on its
own node. The tailplane and the fin are cut the same way for the one-piece elevator and the rudder.
No landing gear: a belly lander, as most autopilot conversions of the kit fly.

Node hierarchy (all names exact):
    bixler3                      root, identity, origin = the centre of gravity
      airframe                   an empty; the fixed meshes are its children:
        fuselage, wing, tailplane, fin, pylon, motor, fittings
      aileron_left, aileron_right
                                 on the hinge lines of the outer wing panels: local +Z along the hinge
                                 toward starboard; a positive turn about local +Z puts the trailing
                                 edge down on either side
      elevator                   one piece, on its hinge line: local +Z toward starboard; a positive
                                 turn about local +Z puts the trailing edge down
      rudder                     on its hinge line: local +Z along the hinge pointing down; a positive
                                 turn about local +Z moves the trailing edge to port
      propeller                  at the disc centre, spins about local +X (blades, hub, spinner)
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
from build_common import (Part, R, T, Z_TO_Y, export, frame_from_forward_up, frame_y_toward, loft,  # noqa: E402
                          make_materials, new_object, newell, pchip, revolve, signed_volume, strut, table)

I4 = np.eye(4)
WHITE = "foam_white"
SURFACES = ("aileron_left", "aileron_right", "elevator", "rudder")


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


def hinge_node(a, b):
    """4x4 node matrix on the hinge line from a to b: origin at its middle, local +Z along a -> b,
    local +Y across it with no X part (the rest rotation then has no turn about its own +Z)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = unit(b - a)
    y = unit((0.0, z[2], -z[1]))
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = np.cross(y, z), y, z, 0.5 * (a + b)
    return M


def trim(v, tr):
    """The colour of a band at the distance v out along a surface: white, then a blue band, then orange."""
    return "trim_orange" if v >= tr["orange_from"] else "trim_blue" if v >= tr["blue_from"] else WHITE


def banded(part, rings, keys, tr, M=None):
    """A loft through the rings, each band between two rings in the trim colour of its middle."""
    v, f = loft(rings)
    n = len(rings[0])
    mats = [m for a, b in zip(keys[:-1], keys[1:]) for m in [trim(0.5 * (abs(a) + abs(b)), tr)] * n]
    part.add_multi(v, f, mats + [trim(abs(keys[0]), tr), trim(abs(keys[-1]), tr)], M)


def round_off(u, floor, p=2.0):
    """The share of a chord left at the fraction u into a rounded tip: down to `floor` at its end."""
    return 1.0 if u <= 0.0 else floor + (1.0 - floor) * (1.0 - min(u, 1.0) ** p) ** (1.0 / p)


def quarter(a, r, n):
    """n + 1 stations from a to a + r, closer together toward a + r (where a rounded tip closes)."""
    return [a + r * math.sin(0.5 * math.pi * k / n) for k in range(n + 1)]


# ---------------------------------------------------------------- the airfoils
def thickness(x):
    """NACA four-digit half-thickness per unit thickness ratio, open trailing edge (0.0105 t at x = 1)."""
    return 5.0 * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def chord_points(x0, x1, n):
    """n + 1 chord fractions from x0 to x1, closer together toward x0."""
    return [x0 + (x1 - x0) * (1.0 - math.cos(0.5 * math.pi * i / n)) for i in range(n + 1)]


def around(xs, upper_all=False):
    """A section's loop as (chord fraction, upper?): the upper surface from the last fraction to the
    first, the lower back (from the first again if upper_all, for a surface that starts at a hinge)."""
    return [(x, True) for x in reversed(xs)] + [(x, False) for x in (xs if upper_all else xs[1:])]


# ---------------------------------------------------------------- the wing
class Wing:
    """geometry.wing on physical's span and thesis_vlm.geometry's datum: the planform, the dihedral
    line and the Clark Y section over them, as functions of the distance z from the centreline."""

    N_CHORD = 22                      # chordwise intervals per surface

    def __init__(self, spec):
        G, V = spec["geometry"]["wing"], spec["thesis_vlm"]["geometry"]
        self.G = G
        self.s = 0.5 * spec["physical"]["b"]
        self.z1, self.cr, self.ct = G["centre_half_span"], G["root_chord"], G["tip_chord"]
        self.r, self.p, self.floor = G["tip_round"], G["tip_round_exponent"], G["tip_floor"]
        self.share, self.tip_share = G["taper_leading_edge_share"], G["tip_leading_edge_share"]
        S = G["section"]                                          # ordinates in % of the chord, off the flat underside
        q = np.sqrt(np.asarray(S["x"], float) / 100.0)            # over sqrt(x): a round nose
        self._up, self._lo = pchip(q, np.asarray(S["upper"], float) / 100.0), pchip(q, np.asarray(S["lower"], float) / 100.0)
        self.h_le = S["upper"][0] / 100.0
        # the incidence is the chord line's (leading edge to trailing edge); the table's datum is the underside
        self.delta = math.radians(V["wing_incidence_deg"]) - math.atan(self.h_le - 0.5 * (S["upper"][-1] + S["lower"][-1]) / 100.0)
        zs = np.linspace(0.0, self.s, 1541)                       # the dihedral line: its slope, summed outward
        u = np.clip((zs - (self.s - self.r)) / self.r, 0.0, 1.0)
        t = np.tan(np.radians(G["dihedral_deg"] + (G["tip_angle_deg"] - G["dihedral_deg"]) * u * u))
        self._zs, self._yd = zs, np.concatenate([[0.0], np.cumsum(0.5 * (t[1:] + t[:-1]) * np.diff(zs))])
        self.x_le0, self.y_le0 = V["cg_x_aft_of_wing_root_leading_edge"], 0.0
        self.y_le0 = V["cg_z_below_wing_underside"] - self.at_x(0.0, False)      # the underside above the origin
        A = G["aileron"]
        (self.a_z0, self.a_z1), self.hinge, self.gap = A["z"], A["hinge_chord_fraction"], A["gap"]

    def chord_straight(self, z):
        z = abs(z)
        return self.cr if z <= self.z1 else self.cr + (self.ct - self.cr) * (z - self.z1) / (self.s - self.z1)

    def chord(self, z):
        return self.chord_straight(z) * round_off((abs(z) - (self.s - self.r)) / self.r, self.floor, self.p)

    def x_le(self, z):
        c = self.chord_straight(z)
        return self.x_le0 - self.share * (self.cr - c) - self.tip_share * (c - self.chord(z))

    def rise(self, z):
        return float(np.interp(abs(z), self._zs, self._yd))

    def surface(self, z, x, upper, k=1.0):
        """(X, Y) of the upper or lower surface at the chord fraction x; k scales the thickness about
        the section's middle line."""
        c, q = self.chord(z), math.sqrt(x)
        up, lo = self._up(q), self._lo(q)
        h = 0.5 * (up + lo) + (0.5 if upper else -0.5) * k * (up - lo)
        a, u = x * c, (h - self.h_le) * c
        cd, sd = math.cos(self.delta), math.sin(self.delta)
        return self.x_le(z) - (a * cd + u * sd), self.y_le0 + self.rise(z) - a * sd + u * cd

    def at_x(self, X, upper):
        """Y of the root section's upper or lower surface at the station X."""
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.surface(0.0, mid, upper)[0] > X:
                lo = mid
            else:
                hi = mid
        return self.surface(0.0, lo, upper)[1]

    def stations(self):
        """[(z, cut at the hinge?)] from the centreline to the tip."""
        s, r, tr = self.s, self.r, self.G["trim"]
        z0c, z1c = round(self.a_z0 - self.gap, 6), round(self.a_z1 + self.gap, 6)
        zs = set(np.round(np.arange(0.0, s - r, 0.045), 6).tolist()) | {self.z1, tr["blue_from"], tr["orange_from"]}
        cut = set(np.round(np.linspace(z0c, z1c, 9), 6).tolist()) | {z for z in zs if z0c < z < z1c}
        zs = {z for z in zs if not (z0c - 0.008 <= z <= z1c + 0.008)}
        zs |= {round(z0c - 0.003, 6), round(z1c + 0.003, 6)} | set(np.round(quarter(s - r, r, 10), 6).tolist())
        return [(z, z in cut) for z in sorted(zs | cut)]

    def ring(self, z, cut, side):
        xs = chord_points(0.0, self.hinge if cut else 1.0, self.N_CHORD)
        return [self.surface(z, x, up) + (side * z,) for x, up in around(xs)]


def build_wing(wing):
    st = wing.stations()
    half = [(z, c, 1.0) for (z, c) in st]
    both = [(z, c, -1.0) for (z, c) in reversed(st[1:])] + half
    part = Part()
    banded(part, [wing.ring(z, c, side) for (z, c, side) in both], [z for (z, _, _) in both], wing.G["trim"])
    return part, len(st) - 1                                     # and the index of the centreline's ring


def build_aileron(wing, side):
    """The aileron on one side: the mesh in its node's frame, and the node's matrix."""
    tr = wing.G["trim"]
    zs = sorted(set(np.round(np.linspace(wing.a_z0, wing.a_z1, 9), 6).tolist())
                | {z for z in (tr["blue_from"], tr["orange_from"]) if wing.a_z0 < z < wing.a_z1})
    if side < 0:
        zs = zs[::-1]                                             # from the port tip inward: toward starboard
    xs = chord_points(wing.hinge, 1.0, wing.N_CHORD // 2)
    rings = [[wing.surface(z, x, up) + (side * z,) for x, up in around(xs, True)] for z in zs]
    a, b = (wing.surface(z, wing.hinge, True, k=0.0) + (side * z,) for z in (zs[0], zs[-1]))     # on the middle line
    M = hinge_node(a, b)
    part = Part()
    banded(part, rings, zs, tr, np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the tailplane
class Tailplane:
    """geometry.tailplane: a thin symmetric section set at the thesis's incidence about the elevator's
    hinge line, which is straight from tip to tip."""

    N_CHORD = 12

    def __init__(self, spec):
        G = spec["geometry"]["tailplane"]
        self.G, self.hs = G, G["half_span"]
        self.delta = math.radians(spec["thesis_vlm"]["geometry"]["tail_incidence_deg"])

    def chords(self, z):
        """(fixed chord, elevator chord) at z: the hinge line to the leading edge, and to the trailing edge."""
        G, z = self.G, abs(z)
        f = round_off((z - (self.hs - G["tip_round"])) / G["tip_round"], G["tip_floor"])
        le = G["x_leading_edge_root"] + (G["x_leading_edge_tip"] - G["x_leading_edge_root"]) * z / self.hs
        return (le - G["x_hinge"]) * f, (G["x_hinge"] - G["x_trailing_edge"]) * f

    def point(self, z, x, upper, k=1.0):
        cf, ce = self.chords(z)
        c = cf + ce
        a, u = x * c - cf, (1.0 if upper else -1.0) * k * thickness(x) * self.G["thickness_ratio"] * c
        cd, sd = math.cos(self.delta), math.sin(self.delta)
        return (self.G["x_hinge"] - (a * cd + u * sd), self.G["hinge_y"] - a * sd + u * cd, z)

    def cut(self, z):
        cf, ce = self.chords(z)
        return cf / (cf + ce)

    def stations(self):
        G, tr = self.G, self.G["trim"]
        inner = self.hs - G["tip_round"]
        half = set(np.round(np.linspace(0.0, inner, 6), 6).tolist()) | {tr["blue_from"], tr["orange_from"]}
        half = sorted(half | set(np.round(quarter(inner, G["tip_round"], 6), 6).tolist()))
        return [-z for z in reversed(half[1:])] + half


def build_tailplane(tail):
    zs = tail.stations()
    fixed, elevator = Part(), Part()
    banded(fixed, [[tail.point(z, x, up) for x, up in around(chord_points(0.0, tail.cut(z), tail.N_CHORD))] for z in zs], zs, tail.G["trim"])
    a, b = (tail.point(z, tail.cut(z), True, k=0.0) for z in (zs[0], zs[-1]))
    M = hinge_node(a, b)
    banded(elevator, [[tail.point(z, x, up) for x, up in around(chord_points(tail.cut(z), 1.0, tail.N_CHORD // 2), True)] for z in zs],
           zs, tail.G["trim"], np.linalg.inv(M))
    return fixed, elevator, M, zs.index(0.0)


# ---------------------------------------------------------------- the fin
class Fin:
    """geometry.fin: sections stacked in Y from the boom to the rounded top, the rudder aft of the hinge
    line; the rudder's lower edge is cut upward toward its trailing edge, clear of the elevator."""

    N_CHORD = 12

    def __init__(self, spec):
        self.G = spec["geometry"]["fin"]
        self.y0, self.y1 = self.G["y"]

    def _lin(self, key, y):
        a, b = self.G[key]
        return a + (b - a) * (y - self.y0) / (self.y1 - self.y0)

    def chords(self, y):
        """(X of the hinge, fixed chord, rudder chord) at the height y."""
        G = self.G
        f = round_off((y - (self.y1 - G["top_round"])) / G["top_round"], G["top_floor"])
        xh = self._lin("x_hinge", y)
        return xh, (self._lin("x_leading_edge", y) - xh) * f, (xh - G["x_trailing_edge"]) * f

    def point(self, y, x, starboard, lift=0.0, k=1.0):
        xh, cf, ce = self.chords(y)
        c = cf + ce
        X = xh + cf - x * c
        t = k * thickness(x) * self.G["thickness_ratio"] * c
        return (X, y + lift * (xh - X), t if starboard else -t)

    def cut(self, y):
        _, cf, ce = self.chords(y)
        return cf / (cf + ce)

    def lift(self, y):
        """The slope of a rudder section at y: the lower edge's cut, gone a little way up."""
        G = self.G
        u = min(1.0, max(0.0, (y - G["rudder_bottom_y"]) / G["rudder_bottom_blend"]))
        return G["rudder_bottom_slope"] * (1.0 - u * u * (3.0 - 2.0 * u))

    def stations(self):
        G, tr = self.G, self.G["trim"]
        inner = self.y1 - G["top_round"]
        ys = set(np.round(np.linspace(self.y0, inner, 8), 6).tolist()) | {tr["blue_from"], tr["orange_from"]}
        return sorted(ys | set(np.round(quarter(inner, G["top_round"], 6), 6).tolist()))


def build_fin(fin):
    ys, tr = fin.stations(), fin.G["trim"]
    fixed, rudder = Part(), Part()
    banded(fixed, [[fin.point(y, x, up) for x, up in around(chord_points(0.0, fin.cut(y), fin.N_CHORD))] for y in ys], ys, tr)
    yb = fin.G["rudder_bottom_y"]
    yr = [yb] + [y for y in ys if y > yb + 0.004]
    a, b = (fin.point(y, fin.cut(y), True, k=0.0) for y in (yr[-1], yr[0]))      # from the top down
    M = hinge_node(a, b)
    banded(rudder, [[fin.point(y, x, up, fin.lift(y)) for x, up in around(chord_points(fin.cut(y), 1.0, fin.N_CHORD // 2), True)] for y in yr],
           yr, tr, np.linalg.inv(M))
    return fixed, rudder, M


# ---------------------------------------------------------------- the pod and boom
class Pod:
    """geometry.fuselage: the outlines over X and the section between them; the canopy is a patch of
    the same surface in another material."""

    N_RING = 40

    def __init__(self, spec):
        F = spec["geometry"]["fuselage"]
        self.F, self.C = F, spec["geometry"]["canopy"]
        self.xn, self.yn, self.xe = F["nose_x"], F["nose_y"], F["end_x"]
        self._w, self._t, self._b = table(F["half_width"]), table(F["top"]), table(F["bottom"])
        self.ye = 0.5 * (self._t(self.xe) + self._b(self.xe))

    def _nose(self, x, a):
        d = self.xn - x
        return 1.0 if d >= a else math.sqrt(max(0.0, 1.0 - ((a - d) / a) ** 2))

    def _end(self, x):
        a, d = self.F["end_length"], x - self.xe
        return 1.0 if d >= a else math.sqrt(max(0.0, 1.0 - ((a - d) / a) ** 2))

    def half_width(self, x):
        return self._w(x) * self._nose(x, self.F["nose_length"]["width"]) * self._end(x)

    def top(self, x):
        y = self.yn + (self._t(x) - self.yn) * self._nose(x, self.F["nose_length"]["top"])
        return self.ye + (y - self.ye) * self._end(x)

    def bottom(self, x):
        y = self.yn + (self._b(x) - self.yn) * self._nose(x, self.F["nose_length"]["bottom"])
        return self.ye + (y - self.ye) * self._end(x)

    def ring(self, x):
        F = self.F
        t, b, w = self.top(x), self.bottom(x), self.half_width(x)
        yw = t - F["widest_at"] * (t - b)
        return [(x, yw + y, z) for (y, z) in superellipse(w, t - yw, yw - b, self.N_RING, *F["exponent"])]

    def stations(self):
        F, C = self.F, self.C
        a, e = F["nose_length"]["width"], F["end_length"]
        nose = [self.xn - a * (1.0 - math.cos(q)) for q in [0.035, 0.09, 0.17] + np.linspace(0.26, 0.5 * math.pi, 12).tolist()]
        body = set(np.round(np.linspace(self.xn - a, self.xe + e, 44)[1:-1], 6).tolist())
        xs = set(np.round(nose, 6).tolist()) | {x for x in body if x < nose[-1] - 0.002}
        xs = {x for x in xs if not any(abs(x - m) < 0.005 for m in C["x"])} | set(C["x"])       # a ring at each end of the canopy
        return sorted(xs, reverse=True) + [self.xe + e * (1.0 - math.cos(q)) for q in (1.5, 1.1, 0.75, 0.45, 0.2, 0.06)]

    def canopy_arc(self, x):
        """Half the canopy's arc about the top of the section at x (radians of the section's own angle), 0 outside it."""
        c0, c1 = self.C["x"]
        return 0.5 * math.radians(self.C["arc_deg"]) if c0 < x < c1 else 0.0


def build_fuselage(pod):
    xs = pod.stations()
    v, f = loft([pod.ring(x) for x in xs])
    n, mats = pod.N_RING, []
    for a, b in zip(xs[:-1], xs[1:]):
        half = pod.canopy_arc(0.5 * (a + b))
        mats += ["canopy_black" if abs(2.0 * math.pi * (k + 0.5) / n - 0.5 * math.pi) < half else WHITE for k in range(n)]
    part = Part()
    part.add_multi(v, f, mats + [WHITE, WHITE])
    return part


# ---------------------------------------------------------------- the pylon, the motor, the propeller
def build_pylon(spec):
    """The motor's pod (a body of revolution about the thrust axis) on its streamlined strut."""
    P = spec["geometry"]["pylon"]
    part = Part()
    (x0, x1), r0, r1, a = P["pod_x"], P["pod_radius"], P["pod_aft_radius"], P["pod_nose"]
    L = x0 - x1
    prof = [(0.0, 0.0)] + [(r0 * math.sin(q), a * (1.0 - math.cos(q))) for q in np.linspace(0.25, 0.5 * math.pi, 8)]
    prof += [(r0 + (r1 - r0) * u * u, a + (L - a) * u) for u in np.linspace(0.25, 1.0, 5)]
    spun(part, prof, WHITE, (x0, P["axis_y"], 0.0), (-1, 0, 0), 28)
    S = P["strut"]
    (y0, y1), (f0, f1), (b0, b1) = S["y"], S["x_front"], S["x_back"]
    rings = []
    for y in np.linspace(y0, y1, 9):
        u = (y - y0) / (y1 - y0)
        xf, c = f0 + (f1 - f0) * u, (f0 + (f1 - f0) * u) - (b0 + (b1 - b0) * u)
        rings.append([(xf - x * c, y, (1.0 if up else -1.0) * thickness(x) * S["thickness_ratio"] * c) for x, up in around(chord_points(0.0, 1.0, 12))])
    v, f = loft(rings)
    part.add(v, f, WHITE)
    return part


def build_motor(spec):
    G = spec["geometry"]
    Mo, P = G["motor"], G["propeller"]
    y = G["pylon"]["axis_y"]
    part = Part()
    r, (m0, m1) = Mo["can_radius"], Mo["can_x"]
    L = m0 - m1
    spun(part, [(0.7 * r, -0.001), (r, 0.002), (r, L - 0.003), (r - 0.002, L - 0.0008), (0.45 * r, L)], "motor_grey", (m0, y, 0.0), (-1, 0, 0), 36)
    hub0 = P["disc_x"] + P["hub_x"][1]                           # the hub's forward face
    strut(part, (m1 + 0.001, y, 0.0), (hub0 - 0.001, y, 0.0), Mo["shaft_radius"], "motor_steel", 12)
    return part


def blade_sections(P, Rt, n=12, m=21):
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
    """(hub and spinner, blades) in the propeller node's frame: the disc centre at the origin, the axis
    local +X, the blades along local +-Y. A positive turn about +X moves the blade on +Y toward +Z
    (clockwise seen from behind); it carries its leading edge there, ahead of its trailing edge: thrust
    forward."""
    P = spec["geometry"]["propeller"]
    hub, blades = Part(), Part()
    rings = [[(u * math.sin(b) + w * math.cos(b), r, u * math.cos(b) - w * math.sin(b)) for (u, w) in sec]
             for (r, b, sec) in blade_sections(P, 0.5 * P["diameter"])]
    v, f = loft(rings)
    for ang in (0.0, 180.0):
        blades.add(v, f, "prop_grey", R("X", ang))
    h0, h1 = P["hub_x"]
    hr, sl = P["hub_radius"], P["spinner_length"]
    spun(hub, [(0.75 * hr, h0), (hr, h0 + 0.002), (hr, h1 - 0.002), (0.75 * hr, h1)], "prop_grey", (0, 0, 0), (1, 0, 0), 28)
    spun(hub, [(hr, -0.001)] + [(hr * math.cos(q), sl * math.sin(q)) for q in np.linspace(0.0, 1.35, 7)] + [(0.0, sl)], "trim_black",
         (h0, 0, 0), (-1, 0, 0), 28)                             # the spinner, aft
    return hub, blades


# ---------------------------------------------------------------- the fittings
def look_frame(c):
    return T(*c["position"]) @ frame_from_forward_up(unit(c["look"]), c["up"])


def build_fittings(spec, pod, wing):
    G = spec["geometry"]
    SE = G["sensors"]
    F = Part()
    pt = G["pitot"]                                              # out of the nose, on its axis
    x0, x1 = pt["x"]
    spun(F, [(pt["radius"], 0.0), (pt["radius"], x1 - x0 - 0.006), (0.55 * pt["radius"], x1 - x0)], "pitot_steel", (x0, pod.yn, 0.0), (1, 0, 0), 12)
    gn, gp = G["gnss"], SE["gnss"]["position"]                   # a puck on the wing's centre
    on = wing.at_x(gp[0], True)
    assert gp[1] - gn["height"] - 0.002 < on < gp[1], ("GNSS puck off the wing's top", on)
    r, h = gn["radius"], gn["height"]
    spun(F, [(r, -h - 0.008), (r, -0.003), (r - 0.003, 0.0), (0.0, 0.0)], "trim_black", gp, (0, 1, 0), 28)
    cn = SE["camera_nadir"]["position"]                          # the windows: in the belly, under the nose
    assert abs(pod.bottom(cn[0]) - cn[1]) < 0.002, ("nadir window off the belly", pod.bottom(cn[0]))
    lens(F, cn, (0, -1, 0), 0.016, 0.012, 0.0025)
    cf = SE["camera_fpv"]["position"]
    assert abs(pod.bottom(cf[0]) - cf[1]) < 0.002, ("forward window off the nose", pod.bottom(cf[0]))
    dx = 0.0005
    tangent = unit((2.0 * dx, pod.bottom(cf[0] + dx) - pod.bottom(cf[0] - dx), 0.0))
    lens(F, cf, (tangent[1], -tangent[0], 0.0), 0.011, 0.008, 0.002)
    return F


# ---------------------------------------------------------------- the nodes
def build_nodes(spec):
    """[(name, parent, part or None, local matrix, props, sharp angle)] in creation order, and what the
    checks need."""
    G = spec["geometry"]
    wing, tail, fin, pod = Wing(spec), Tailplane(spec), Fin(spec), Pod(spec)
    wing_part, centre = build_wing(wing)
    tail_part, elevator, M_elevator, tail_centre = build_tailplane(tail)
    fin_part, rudder, M_rudder = build_fin(fin)
    nodes = [("bixler3", None, None, I4, {
        "frame": "X forward, Y up, Z starboard; origin = the centre of gravity (the thesis's body origin)",
        "spec": "bixler3_spec.json", "body_axes": "body (x fwd, y right, z down) = (X, Z, -Y)",
        "mass": spec["physical"]["mass"], "model": spec["reference"]["thesis"]["url"]}, 32.0)]
    nodes.append(("airframe", "bixler3", None, I4, {}, 32.0))
    for name, part, sharp in (("fuselage", build_fuselage(pod), 40.0), ("wing", wing_part, 40.0), ("tailplane", tail_part, 40.0),
                              ("fin", fin_part, 40.0), ("pylon", build_pylon(spec), 40.0), ("motor", build_motor(spec), 32.0),
                              ("fittings", build_fittings(spec, pod, wing), 32.0)):
        nodes.append((name, "airframe", part, I4, {}, sharp))
    down = {"axis": "+Z (hinge, toward starboard)", "positive": "trailing edge down"}
    for name, side in (("aileron_left", -1.0), ("aileron_right", 1.0)):
        part, M = build_aileron(wing, side)
        nodes.append((name, "bixler3", part, M, down, 40.0))
    nodes.append(("elevator", "bixler3", elevator, M_elevator, down, 40.0))
    nodes.append(("rudder", "bixler3", rudder, M_rudder, {"axis": "+Z (hinge, pointing down)", "positive": "trailing edge to port"}, 40.0))
    hub, blades = build_propeller(spec)
    P = G["propeller"]
    nodes.append(("propeller", "bixler3", merged(hub, blades), T(P["disc_x"], G["pylon"]["axis_y"], 0.0),
                  {"axis": "+X", "positive": "spin vector along body +x, clockwise seen from behind", "diameter": P["diameter"]}, 32.0))
    SE = G["sensors"]
    for k in ("imu", "gnss", "pitot_tip"):
        nodes.append((k, "bixler3", None, T(*SE[k]["position"]), {}, 32.0))
    for k in ("camera_nadir", "camera_fpv"):
        nodes.append((k, "bixler3", None, look_frame(SE[k]), {"convention": "look -Z, up +Y", "assumed": SE[k]["note"]}, 32.0))
    return nodes, {"wing": wing, "pod": pod, "hub": hub, "blades": blades, "centre": centre, "tail_centre": tail_centre}


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


def planform(tris, cell):
    """The triangles seen from above on a grid of `cell`: the area of their union, and its mean
    aerodynamic chord (the chord of each spanwise strip squared, over the area)."""
    P = tris[:, :, [0, 2]]
    lo = P.min(axis=(0, 1)) - cell
    nx, nz = np.ceil((P.max(axis=(0, 1)) + cell - lo) / cell).astype(int)
    G = np.zeros((nx + 1, nz + 1), bool)
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
    chord = G.sum(axis=0) * cell
    return float(chord.sum() * cell), float((chord * chord).sum() / chord.sum())


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
    G, Ph, V = spec["geometry"], spec["physical"], spec["thesis_vlm"]["geometry"]
    Ga = G["gates"]
    W = world_matrices(nodes)
    parts = {name: part for name, _, part, _, _, _ in nodes if part is not None}
    tri = {name: triangles(part, W[name]) for name, part in parts.items()}
    print("[bixler3] nodes:")
    for name, parent, part, _, _, _ in nodes:
        kind = "empty" if part is None else f"mesh, {sum(len(f) - 2 for f in part.faces)} triangles, " + ", ".join(sorted(set(part.mats)))
        print(f"[bixler3]   {name:<14} < {parent or '-':<9} {kind}")
    tris_n = sum(len(t) for t in tri.values())
    errs = []

    def gate(label, value, ref, tol):
        ok = abs(value - ref) <= tol
        print(f"[bixler3] gate {label:<44} {value:9.4f}  ({ref} +- {tol})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(label)

    def least(label, value, floor):
        ok = value >= floor
        print(f"[bixler3] gate {label:<44} {value:9.4f}  (at least {floor})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(label)

    # the wing: span, area and mean aerodynamic chord on the mesh, seen from above
    lifting = np.concatenate([tri["wing"], tri["aileron_left"], tri["aileron_right"]])
    gate("wingspan, projected (thesis b)", float(lifting[:, :, 2].max() - lifting[:, :, 2].min()), Ph["b"], Ga["span"])
    N = 2 * Wing.N_CHORD + 1                                     # a wing ring: its leading edge is vertex N_CHORD
    Vw = np.asarray(parts["wing"].verts)
    rings = Vw[:len(Vw) - 2 * N].reshape(-1, N, 3)
    le = rings[:, Wing.N_CHORD, 1:]
    gate("span over the tips, along the wing (vendor)", float(np.linalg.norm(np.diff(le, axis=0), axis=1).sum()), G["span_over_tips"], Ga["span_over_tips"])
    S_wing, mac = planform(lifting, Ga["planform_cell"])
    gate("wing area, projected (thesis S)", S_wing, Ph["S"], Ga["area"])
    gate("mean aerodynamic chord (thesis cbar)", mac, Ph["cbar"], Ga["mac"])
    # the centre of gravity under the wing, the incidences
    root = rings[extra["centre"]]
    lead, trail = root[Wing.N_CHORD], 0.5 * (root[0] + root[-1])
    assert abs(lead[2]) < 1e-12, "the wing's middle ring is not on the centreline"
    gate("CG aft of the wing root's leading edge", float(lead[0]), V["cg_x_aft_of_wing_root_leading_edge"], Ga["datum"])
    lower = root[Wing.N_CHORD:]                                  # leading edge to trailing edge, X falling
    gate("CG below the wing's underside", float(np.interp(0.0, lower[::-1, 0], lower[::-1, 1])), V["cg_z_below_wing_underside"], Ga["datum"])
    gate("wing incidence, deg (chord line)", math.degrees(math.atan2(lead[1] - trail[1], lead[0] - trail[0])), V["wing_incidence_deg"], Ga["incidence_deg"])
    Nt = 2 * Tailplane.N_CHORD + 1
    Vt = np.asarray(parts["tailplane"].verts)
    t_lead = Vt[:len(Vt) - 2 * Nt].reshape(-1, Nt, 3)[extra["tail_centre"]][Tailplane.N_CHORD]
    Ne = 2 * (Tailplane.N_CHORD // 2 + 1)
    Ve = np.asarray(parts["elevator"].verts) @ W["elevator"][:3, :3].T + W["elevator"][:3, 3]
    e_ring = Ve[:len(Ve) - 2 * Ne].reshape(-1, Ne, 3)[extra["tail_centre"]]
    t_trail = 0.5 * (e_ring[0] + e_ring[-1])
    gate("tailplane incidence, deg", math.degrees(math.atan2(t_lead[1] - t_trail[1], t_lead[0] - t_trail[0])), V["tail_incidence_deg"], Ga["incidence_deg"])
    # the length: everything but the pitot tube
    body = np.concatenate([tri[k] for k in tri if k != "fittings"])
    everything = np.concatenate(list(tri.values()))
    gate("length (vendor)", float(body[:, :, 0].max() - body[:, :, 0].min()), G["length"], Ga["length"])
    # the propeller: its size, its pitch's sense, what its disc clears
    P = G["propeller"]
    Rp = 0.5 * P["diameter"]
    Vb = np.asarray(extra["blades"].verts)
    gate("propeller tip circle", 2.0 * float(np.hypot(Vb[:, 1], Vb[:, 2]).max()), P["diameter"], Ga["tip_circle"])
    sec = Vb[(Vb[:, 1] > 0.45 * Rp) & (Vb[:, 1] < 0.65 * Rp)]     # the blade on +Y leads toward +Z, its leading edge forward
    fw, bk = sec[np.argmax(sec[:, 0])], sec[np.argmin(sec[:, 0])]
    assert fw[2] > 0.0 > bk[2] and fw[0] - bk[0] > 0.002, "propeller blade pitched the wrong way"
    pp = W["propeller"][:3, 3]
    bx0, bx1 = pp[0] + float(Vb[:, 0].min()), pp[0] + float(Vb[:, 0].max())
    near = lifting.reshape(-1, 3)
    near = near[np.hypot(near[:, 1] - pp[1], near[:, 2] - pp[2]) < Rp + Ga["clearance"]]
    least("disc behind the wing's trailing edge", float(near[:, 0].min()) - bx1, Ga["clearance"])
    least("disc behind the pylon and its pod", float(tri["pylon"][:, :, 0].min()) - bx1, Ga["clearance"])
    aft = np.concatenate([tri[k] for k in ("fuselage", "tailplane", "fin", "elevator", "rudder")])
    slab = aft[(aft[:, :, 0].min(axis=1) <= bx1 + Ga["clearance"]) & (aft[:, :, 0].max(axis=1) >= bx0 - Ga["clearance"])]
    least("disc above the tail boom", rim_gap(pp[[1, 2]], Rp, slab, (1, 2)), Ga["clearance"])
    motor_gap = float(tri["motor"][:, :, 0].min()) - (pp[0] + P["hub_x"][1])
    assert motor_gap > -0.0021, ("the motor's shaft runs through the hub", motor_gap)
    # the surfaces turn the way the nodes say
    turn = R("Z", math.degrees(0.3))[:3, :3]
    moved = {}
    for name in SURFACES:
        Vs, M = np.asarray(parts[name].verts), W[name]
        te = Vs[np.argsort(-np.hypot(Vs[:, 0], Vs[:, 1]))[:8]]    # the vertices farthest from the hinge
        d = ((te @ turn.T - te) @ M[:3, :3].T).mean(axis=0)
        assert abs(M[0, 1]) < 1e-12, ("hinge node: local +Y has an X part", name)
        if name == "rudder":
            moved[name] = float(d[2])
            assert M[1, 2] < -0.9 and d[2] < -0.003, ("rudder: +Z is not down the hinge, or does not move the trailing edge to port", d)
        else:
            moved[name] = float(d[1])
            assert M[2, 2] > 0.9 and d[1] < -0.003, ("+Z is not toward starboard, or does not put the trailing edge down", name, d)
        assert float(np.hypot(Vs[:, 0], Vs[:, 1]).min()) < 0.008, ("the surface does not start at its node's hinge line", name)
    # the elevator turned fully up passes under the rudder's lower edge, wherever the rudder is
    up_deg, side_deg = Ga["throw_deg"]
    Vu = np.asarray(parts["elevator"].verts) @ R("Z", -up_deg)[:3, :3].T @ W["elevator"][:3, :3].T + W["elevator"][:3, 3]
    top = Vu[:len(Vu) - 2 * Ne].reshape(-1, Ne, 3)[extra["tail_centre"]][:Ne // 2]      # trailing edge to hinge, X rising
    under = []
    for deg in (-side_deg, 0.0, side_deg):
        Vr = np.asarray(parts["rudder"].verts) @ R("Z", deg)[:3, :3].T @ W["rudder"][:3, :3].T + W["rudder"][:3, 3]
        edge = Vr[:2 * (Fin.N_CHORD // 2 + 1)]                    # the rudder's lowest ring
        edge = edge[edge[:, 0] >= top[0, 0]]
        under.append(float((edge[:, 1] - np.interp(edge[:, 0], top[:, 0], top[:, 1])).min()))
    least("rudder's lower edge over the raised elevator", min(under), Ga["under_rudder"])
    # every part closed and wound outward (each primitive was checked as it was added, by Part.add)
    vols = {}
    for name, part in parts.items():
        Vp = np.asarray(part.verts)
        ns = [newell(Vp, f) for f in part.faces]
        tot, mag = float(np.linalg.norm(sum(ns))), float(sum(np.linalg.norm(n) for n in ns))
        vols[name] = signed_volume(Vp, part.faces)
        assert tot <= 1e-9 + 1e-6 * mag and vols[name] > 0.0, ("part not closed or wound inward", name, tot, mag, vols[name])
    least("triangles under the budget", float(Ga["triangles_max"] - tris_n), 0.0)
    # what is printed, not gated
    pod = extra["pod"]
    tz = tri["tailplane"][:, :, 2]
    S_tail, _ = planform(np.concatenate([tri["tailplane"], tri["elevator"]]), Ga["planform_cell"])
    fixed = np.concatenate([tri[k] for k in ("fuselage", "wing", "tailplane", "fin", "pylon", "motor", "aileron_left", "aileron_right", "elevator", "rudder")])
    print(f"[bixler3] surfaces, +0.3 rad about local +Z: ailerons' trailing edges {1000 * moved['aileron_left']:+.1f} and "
          f"{1000 * moved['aileron_right']:+.1f} mm in Y (down), the elevator's {1000 * moved['elevator']:+.1f} mm in Y (down), "
          f"the rudder's {1000 * moved['rudder']:+.1f} mm in Z (to port)")
    print(f"[bixler3] propeller: disc centre X {pp[0]:+.4f} Y {pp[1]:+.4f}, blades from X {bx0:+.4f} to {bx1:+.4f}; the motor's shaft ends "
          f"{1000 * motor_gap:+.1f} mm from the hub's forward face")
    print(f"[bixler3] length {float(body[:, :, 0].max() - body[:, :, 0].min()):.4f} m (nose X {float(body[:, :, 0].max()):+.4f} to the rudder's "
          f"trailing edge X {float(body[:, :, 0].min()):+.4f}), {float(everything[:, :, 0].max() - everything[:, :, 0].min()):.4f} with the pitot tube")
    print(f"[bixler3] height {float(fixed[:, :, 1].max() - fixed[:, :, 1].min()):.4f} m from the belly (Y {float(fixed[:, :, 1].min()):+.4f}) to the "
          f"fin's top (Y {float(fixed[:, :, 1].max()):+.4f}); the wing tips' top at Y {float(lifting[:, :, 1].max()):+.4f}")
    print(f"[bixler3] tailplane span {float(tz.max() - tz.min()):.4f} m, area with the elevator {S_tail:.4f} m^2; pod "
          f"{2.0 * max(pod.half_width(x) for x in pod.stations()):.3f} wide, {max(pod.top(x) - pod.bottom(x) for x in pod.stations()):.3f} high")
    print(f"[bixler3] triangles {tris_n}; every part closed and wound outward (Newell sum, signed volume): "
          + ", ".join(f"{k} {1e3 * v:.3f} l" for k, v in vols.items() if k in ("fuselage", "wing", "tailplane", "fin", "pylon")))
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
    return obj["bixler3"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="bixler3_spec.json")
    ap.add_argument("--out", default="bixler3.glb")
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
    print(f"[bixler3] wrote {path(a.out)}  triangles={tris_n}  bytes={os.path.getsize(path(a.out))}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

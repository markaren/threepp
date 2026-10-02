"""Build MS Trollfjord (Hurtigruten coastal express, 2002, 135.75 m) from trollfjord_spec.json:
trollfjord.glb (via Blender) and trollfjord_hydro.json.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup --python build_trollfjord_blender.py -- --spec trollfjord_spec.json --out trollfjord.glb

Hydrostatics only (plain Python + numpy, no Blender; writes trollfjord_hydro.json):
    python build_trollfjord_blender.py --hydro-only

The mesh primitives, the Newell closure check (Part), the hydrostatic integrator and
the Blender export are the Mariner generator's (../usv/build_mariner_blender.py); the
export and the gates are the sjark's (build_sjark_blender.py). All geometry is built
with numpy in the spec's vessel frame (X forward, Y up, Z starboard, metres, origin on
the centreline on the baseline at mid-LOA) and converted to Blender's Z-up only when a
mesh or a transform is handed to bpy; the glTF exporter maps it straight back.

The level of detail is for 60 m to 2 km: silhouette, paint bands, window rows as decals.

Node hierarchy (all names exact):
    trollfjord                   root, identity, origin = centreline x baseline x mid-LOA
      hull                       lofted hull (antifouling / black / red band split at exact paint
                                 edges, deck), bulbous bow, forecastle and aft bulwarks, hull
                                 windows, anchor pockets, thruster tunnel mouths, side and stern
                                 doors, rescue-boat recess (starboard), TROLLFJORD on both bows and the stern
      superstructure             white tiers deck4..deck8, lifeboat bays, bridge wings, roof rails,
                                 funnel casing, name board
      windows                    window rows, glass bands and the bridge front (own dark-glass material)
      funnel                     black funnel and exhaust pipes
      mast                       forward radar mast, aft mast on the funnel front, radomes, whips,
                                 masthead light housings
      lifeboats                  two enclosed lifeboats per side with davits, starboard rescue boat
      deck_fittings              bollards, sidelight boxes, stern light
      radar_fwd, radar_aft       open-array antennas, spin about local +Y
      rudder_port, rudder_stbd   the azimuth thrusters (strut + pod); steer about local +Y
        propeller_port / _stbd   child of its unit, spins about local +X (pulling, at the pod front)
        prop_thrust_port / _stbd empty, child of its unit, thrust along local +X
      bow_thruster_1, bow_thruster_2, stern_thruster   empties at the tunnel centres (thrust axis +Z)
      bridge                     empty at the conning position (looks down local -Z, +Y up)
      nav_light_port, nav_light_stbd, masthead_fwd, masthead_aft, stern_light
                                 light empties (look -Z, up +Y)
      bollard_fwd_port, bollard_fwd_stbd, bollard_aft_port, bollard_aft_stbd   bollard tops
      name_port, name_stbd       empties at the centre of each bow name (local +Z = outward hull
                                 normal, +X = reading direction)

The hydrostatics integrate the buoyant closed meshes (hull shell, bulb) column by column with
the Mariner's winding-number integrator, solve the floating position for each loading
condition and assert the design draft, trim and GM_T gates.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "usv"))
sys.path.insert(0, HERE)
from build_mariner_blender import (Hydro, Part, T, Z_TO_X, Z_TO_Y, box_between, clip_slab,  # noqa: E402
                                   cylinder, frustum, hydrostatics, loft, make_materials, mass_condition,
                                   new_object, pchip, revolve, sensor_frame, strut)
from build_sjark_blender import check_gates, export, mass_check  # noqa: E402


def table(t):
    a = np.asarray(t, float)
    return pchip(a[:, 0], a[:, 1])


# ---------------------------------------------------------------- hull lines
class Lines:
    """The hull as functions of x. Section (starboard half): keel -> superellipse bottom and
    bilge (exponent 'fullness') up to the antifouling top -> flared topside up to the deck
    edge, with the black/red paint level as a chain point -> cambered deck to the centreline."""

    DECK_Q = (0.66, 0.33)

    def __init__(self, spec):
        H, P = spec["hull"], spec["paint"]
        self.H = H
        self.keel, self.hw, self.hd = table(H["keel"]), table(H["half_breadth_low"]), table(H["half_breadth_deck"])
        self.n, self.yd = table(H["fullness"]), table(H["deck_edge_y"])
        self.black = table(P["black_top_y"])
        self.af = P["antifouling_top_y"]
        self.x0, self.x1 = H["transom_x"], H["stem_x"]
        self.nl, self.nh = H["chain"]["low"], H["chain"]["high"]

    def yref(self, x):
        return max(self.af, self.keel(x) + 1e-3)

    def low_pt(self, x, th):
        yk, yr, hw, n = self.keel(x), self.yref(x), self.hw(x), self.n(x)
        s, c = max(0.0, math.sin(th)), max(0.0, math.cos(th))
        return hw * s ** (2.0 / n), yr - (yr - yk) * c ** (2.0 / n)

    def high_z(self, x, y):
        yr, yd, hw, hd = self.yref(x), self.yd(x), self.hw(x), self.hd(x)
        t = min(1.0, max(0.0, (y - yr) / (yd - yr)))
        return hw + (hd - hw) * t ** self.H["flare_power"]

    def shell_z(self, x, y):
        if y >= self.yref(x):
            return self.high_z(x, y)
        lo, hi = 0.0, 0.5 * math.pi
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if self.low_pt(x, m)[1] < y:
                lo = m
            else:
                hi = m
        return self.low_pt(x, 0.5 * (lo + hi))[0]

    def black_at(self, x):
        yr, yd = self.yref(x), self.yd(x)
        return min(yd - 0.02, max(yr + 0.02, self.black(x)))

    def deck(self, x, z):
        hd = self.hd(x)
        q = min(1.0, abs(z) / hd) if hd > 1e-6 else 1.0
        return self.yd(x) + self.H["deck_camber"] * (1.0 - q * q)

    def section_half(self, x):
        pts = []
        for k in range(self.nl):
            z, y = self.low_pt(x, 0.5 * math.pi * k / self.nl)
            pts.append((z, y, "low"))
        yr, yb, yd = self.yref(x), self.black_at(x), self.yd(x)
        nb = int(round(0.72 * self.nh))
        nr = self.nh - nb
        for (a, b, n, kind) in ((yr, yb, nb, "black"), (yb, yd, nr, "red")):
            for k in range(n):
                y = a + (b - a) * k / n
                pts.append((self.high_z(x, y), y, kind))
        hd = self.hd(x)
        pts.append((hd, yd, "deck"))
        for q in self.DECK_Q:
            pts.append((hd * q, self.deck(x, hd * q), "deck"))
        pts.append((0.0, self.deck(x, 0.0), "deck"))
        return pts

    def ring(self, x):
        S = self.section_half(x)
        stb = [(x, y, z) for (z, y, _) in S]
        port = [(x, y, -z) for (z, y, _) in reversed(S[1:-1])]
        Sk = [k for (_, _, k) in S]
        port_kinds = [Sk[i - 1] for i in range(len(S) - 1, 0, -1)]
        return stb + port, Sk[:-1] + port_kinds


def stations(spec):
    H = spec["hull"]
    S = H["stations"]
    x0, x1 = H["transom_x"], H["stem_x"]
    xs = set(np.round(np.arange(x0, x1, S["dx"]), 6).tolist())
    for (a, b) in S["fine_ranges"]:
        xs |= set(np.round(np.arange(a, b, S["dx_fine"]), 6).tolist())
    xs |= {x0, x1}
    return sorted(set(round(x, 6) for x in xs if x0 <= x <= x1))


KIND_MAT = {"low": "antifouling", "black": "hull_black", "red": "hull_red", "deck": "deck_grey"}


def build_hull(spec, L, part):
    xs = stations(spec)
    rings, kinds = [], None
    for x in xs:
        r, kinds = L.ring(x)
        rings.append(r)
    v, f = loft(rings, caps=False)
    N = len(rings[0])
    mats = [KIND_MAT[kinds[q % N]] for q in range(len(f))]
    # transom: the aft ring as horizontal paint bands
    ring0 = [(p[1], p[2]) for p in rings[0]]
    yb = L.black_at(xs[0])
    for (a, b, mat) in ((-9.0, L.af, "antifouling"), (L.af, yb, "hull_black"), (yb, 99.0, "hull_red")):
        piece = clip_slab(ring0, a, b)
        if len(piece) < 3:
            continue
        base = len(v)
        v += [(xs[0], y, z) for (y, z) in piece]
        f.append(tuple(base + k for k in range(len(piece) - 1, -1, -1)))
        mats.append(mat)
    base = len(v)
    v += [tuple(p) for p in rings[-1]]
    f.append(tuple(base + k for k in range(N)))
    mats.append("hull_red")
    part.add_multi(v, f, mats, buoyant=True)
    return xs


def build_bulb(spec, part):
    B = spec["hull"]["bulb"]
    rings = []
    n = 20
    for s in np.concatenate([np.linspace(0.0, 0.9, 12), np.linspace(0.92, 0.995, 6)]):
        x = B["x_aft"] + (B["x_nose"] - B["x_aft"]) * s
        sc = (1.0 - s ** B["nose_power"]) ** (1.0 / B["nose_power"])
        ring = []
        for k in range(n):
            a = 2 * math.pi * k / n
            ring.append((x, B["axis_y"] + B["half_height"] * sc * math.cos(a), B["half_width"] * sc * math.sin(a)))
        rings.append(ring)
    v, f = loft(rings)
    part.add(v, f, "antifouling", buoyant=True)


def wall_along_deck(L, part, xa, xb, h, th, mat, n=40):
    """A bulwark on the deck edge from xa to xb, both sides (closed lofts)."""
    xs = np.linspace(xa, xb, n)
    for s in (1, -1):
        rings = []
        for x in xs:
            zs, ys = L.hd(x), L.yd(x)
            zi = max(zs - th, 0.02)
            rings.append([(x, ys - 0.05, s * zs), (x, ys + h, s * zs), (x, ys + h, s * zi), (x, ys - 0.05, s * zi)])
        part.add(*loft(rings), mat)


def hull_decal(L, part, x0, x1, y0, y1, mat, s, off=0.03, nx=6, ny=4, ellipse=False):
    """Open decal on the hull side s (+1 stbd, -1 port) following the shell."""
    if ellipse:
        pts = []
        for k in range(12):
            a = 2 * math.pi * k / 12
            x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * math.cos(a)
            y = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * math.sin(a)
            pts.append((x, y, s * (L.shell_z(x, y) + off)))
        part.add(pts, [tuple(range(12))], mat, closed=False, outward=(0.0, 0.0, s))
        return
    V, F = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny
            V.append((x, y, s * (L.shell_z(x, y) + off)))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            F.append((a, a + 1, a + nx + 2, a + nx + 1))
    part.add(V, F, mat, closed=False, outward=(0.0, 0.0, s))


def build_hull_details(spec, L, part):
    HD = spec["hull_details"]
    hw_ = HD["hull_windows"]
    for s in (1, -1):
        for xc in np.arange(hw_["x0"], hw_["x1"] + 1e-6, hw_["pitch"]):
            if any(a <= xc <= b for (a, b) in hw_["gap_x"]):
                continue
            hull_decal(L, part, xc - 0.5 * hw_["width"], xc + 0.5 * hw_["width"], hw_["y0"], hw_["y1"], "window_glass", s, nx=1, ny=1)
        ap = HD["anchor_pocket"]
        hull_decal(L, part, ap["x"] - ap["rx"], ap["x"] + ap["rx"], ap["y"] - ap["ry"], ap["y"] + ap["ry"], "door_dark", s, ellipse=True)
        sd = HD["side_door"]
        hull_decal(L, part, sd["x0"], sd["x1"], sd["y0"], sd["y1"], "door_dark", s, off=0.02)
        hull_decal(L, part, sd["x0"] + 0.15, sd["x1"] - 0.15, sd["y0"] + 0.15, sd["y1"] - 0.15, "hull_black", s, off=0.03)
        for t in spec["propulsion"]["bow_thrusters"] + [spec["propulsion"]["stern_thruster"]]:
            r = t["tunnel_radius"]
            hull_decal(L, part, t["x"] - r, t["x"] + r, t["y"] - r, t["y"] + r, "door_dark", s, ellipse=True)
    rr = HD["rescue_recess"]
    s = -1 if rr["side"] == "port" else 1
    hull_decal(L, part, rr["x0"], rr["x1"], rr["y0"], rr["y1"], "door_dark", s, off=0.02)
    # stern door outline on the transom
    st = HD["stern_door"]
    x0 = L.x0 - 0.02
    for (a, b, c, d, m) in ((st["z0"], st["z1"], st["y0"], st["y1"], "door_dark"),
                            (st["z0"] + 0.18, st["z1"] - 0.18, st["y0"] + 0.18, st["y1"] - 0.18, "hull_black")):
        xo = x0 - (0.0 if m == "door_dark" else 0.01)
        q = [(xo, c, a), (xo, c, b), (xo, d, b), (xo, d, a)]
        part.add(q, [(0, 1, 2, 3)], m, closed=False, outward=(-1.0, 0.0, 0.0))
    H = spec["hull"]
    fb = H["forecastle_bulwark"]
    wall_along_deck(L, part, fb["x0"], L.x1 - 0.05, fb["height"], fb["thickness"], "super_white")
    xe = L.x1 - 0.05
    v, f, M = box_between((xe - 0.6, L.yd(xe) - 0.05, -L.hd(xe) - 0.01), (L.x1 + 0.02, L.yd(xe) + fb["height"], L.hd(xe) + 0.01))
    part.add(v, f, "super_white", M)
    ab = H["aft_bulwark"]
    wall_along_deck(L, part, L.x0, ab["x1"], ab["height"], ab["thickness"], "hull_red", n=12)
    zs = L.hd(L.x0)
    v, f, M = box_between((L.x0, L.yd(L.x0) - 0.05, -zs), (L.x0 + ab["thickness"], L.yd(L.x0) + ab["height"], zs))
    part.add(v, f, "hull_red", M)


# ---------------------------------------------------------------- superstructure
def tier_outline(t, L):
    """Plan outline [(x, z)] of a tier: starboard run from the aft end forward round the front,
    then the port run back (Part orients closed prisms). With aft_len the aft end is a
    superellipse (aft_power) of that length, else a square end with small chamfers."""
    w, xa, xf, fl, p = t["half_width"], t["x_aft"], t["x_fwd"], t["front_len"], t["front_power"]
    al, pa = t.get("aft_len", 0.0), t.get("aft_power", 2.0)
    ch = 0.4

    def zf(x):
        z = w
        if x > xf - fl:
            u = min(1.0, (x - (xf - fl)) / fl)
            z = w * max(0.0, 1.0 - u ** p) ** (1.0 / p)
        if al > 0.0 and x < xa + al:
            u = min(1.0, (xa + al - x) / al)
            z = min(z, w * max(0.0, 1.0 - u ** pa) ** (1.0 / pa))
        if t.get("clip_to_deck"):
            z = min(z, L.hd(x) + t.get("clip_margin", 0.0))
        return z
    rc = t.get("recess")
    if al > 0.0:
        aft = [xa + al * (1 - math.cos(0.5 * math.pi * k / 24)) for k in range(0, 24)]
        side = list(np.linspace(xa + al, xf - fl, 14))
    else:
        aft = []
        side = list(np.linspace(xa + ch, xf - fl, 14))
    arc = [xf - fl + fl * (1 - math.cos(0.5 * math.pi * k / 24)) for k in range(1, 25)]
    xs = aft + side + arc
    stb = []
    for x in xs:
        if rc and rc["x0"] < x < rc["x1"]:
            continue
        stb.append((x, zf(x)))
    if rc:
        # insert the bay (a notch in plan)
        out = []
        done = False
        for (x, z) in stb:
            if not done and x > rc["x1"]:
                out += [(rc["x0"], w), (rc["x0"], w - rc["depth"]), (rc["x1"], w - rc["depth"]), (rc["x1"], w)]
                done = True
            out.append((x, z))
        stb = out
    if al > 0.0:
        stb[0] = (stb[0][0], max(stb[0][1], 0.05))
    else:
        stb = [(xa, w - ch)] + stb
    stb[-1] = (stb[-1][0], max(stb[-1][1], 0.05))
    outline = stb + [(x, -z) for (x, z) in reversed(stb)]
    return outline, zf


def panes_along(glass, poly, centre, y0, y1, pitch, width, off):
    """Window panes (open decals) along a plan polyline [(x, z)], one every `pitch` metres of
    arc length, each `width` long, offset `off` outward (away from `centre`)."""
    P = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    def at(sv):
        i = int(min(len(seg) - 1, max(0, np.searchsorted(cum, sv) - 1)))
        f = (sv - cum[i]) / max(seg[i], 1e-9)
        q = P[i] + (P[i + 1] - P[i]) * f
        d = P[i + 1] - P[i]
        n = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
        if np.dot(n, q - np.asarray(centre)) < 0:
            n = -n
        return q + off * n, n
    n_p = int(total / pitch)
    s0 = 0.5 * (total - n_p * pitch)
    for k in range(n_p + 1):
        sc = s0 + k * pitch
        a, b = sc - 0.5 * width, sc + 0.5 * width
        if a < 0.0 or b > total:
            continue
        pts = [at(a + (b - a) * j / 3) for j in range(4)]
        V = [(q[0], y0, q[1]) for (q, _) in pts] + [(q[0], y1, q[1]) for (q, _) in reversed(pts)]
        nm = sum(n for (_, n) in pts)
        glass.add(V, [tuple(range(8))], "window_glass", closed=False, outward=(nm[0], 0.0, nm[1]))


def point_in(poly, x, z):
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % n]
        if (z1 > z) != (z2 > z):
            xi = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if xi > x:
                inside = not inside
    return inside


def prism(outline, y0, y1):
    return loft([[(x, y0, z) for (x, z) in outline], [(x, y1, z) for (x, z) in outline]])


def build_superstructure(spec, L, part, glass):
    S = spec["superstructure"]
    off = S["window_offset"]
    outlines = {}
    for t in S["tiers"]:
        ol, zf = tier_outline(t, L)
        outlines[t["name"]] = (ol, t)
        v, f = prism(ol, t["y0"], t["y1"])
        n = len(ol)
        part.add_multi(v, f, ["super_white"] * n + ["roof_grey", "roof_grey"])
        rc = t.get("recess")
        w = zf(0.5 * (t["x_aft"] + t["x_fwd"] - t["front_len"]))     # the straight side (clipped to the deck edge)

        def straight(x):
            if rc and rc["x0"] - 0.2 < x < rc["x1"] + 0.2:
                return False
            return abs(zf(x) - w) < 1e-3 and t["x_aft"] + 0.5 < x
        for wr in t["windows"]:
            if wr["kind"] in ("rect", "port"):
                for xc in np.arange(wr["x0"], wr["x1"] + 1e-6, wr["pitch"]):
                    a, b = xc - 0.5 * wr["width"], xc + 0.5 * wr["width"]
                    if not (straight(a) and straight(b)):
                        continue
                    for s in (1, -1):
                        z = s * (w + off)
                        if wr["kind"] == "rect":
                            q = [(a, wr["y0"], z), (b, wr["y0"], z), (b, wr["y1"], z), (a, wr["y1"], z)]
                        else:
                            yc, r = 0.5 * (wr["y0"] + wr["y1"]), 0.5 * (wr["y1"] - wr["y0"])
                            q = [(xc + r * math.cos(2 * math.pi * k / 8), yc + r * math.sin(2 * math.pi * k / 8), z) for k in range(8)]
                        glass.add(q, [tuple(range(len(q)))], "window_glass", closed=False, outward=(0.0, 0.0, s))
            elif wr["kind"] == "band":
                a, b = wr["x0"], wr["x1"]
                while a < b and not straight(a):
                    a += 0.25
                while b > a and not straight(b):
                    b -= 0.25
                if b - a < 0.5:
                    continue
                for s in (1, -1):
                    z = s * (w + off)
                    q = [(a, wr["y0"], z), (b, wr["y0"], z), (b, wr["y1"], z), (a, wr["y1"], z)]
                    glass.add(q, [(0, 1, 2, 3)], "window_glass", closed=False, outward=(0.0, 0.0, s))
            elif wr["kind"] == "front":
                pts = [(x, z) for (x, z) in ol if x >= wr["arc_from"]]
                # order: the stb run goes forward, the port run comes back: already continuous
                V = []
                for i, (x, z) in enumerate(pts):
                    if i == 0:
                        d = np.array(pts[1]) - np.array(pts[0])
                    elif i == len(pts) - 1:
                        d = np.array(pts[-1]) - np.array(pts[-2])
                    else:
                        d = np.array(pts[i + 1]) - np.array(pts[i - 1])
                    nrm = np.array([d[1], -d[0]])
                    nrm /= (np.linalg.norm(nrm) + 1e-12)
                    if np.dot(nrm, (x - (t["x_fwd"] - t["front_len"]), z)) < 0:
                        nrm = -nrm
                    xo, zo = x + off * nrm[0], z + off * nrm[1]
                    V += [(xo, wr["y0"], zo), (xo, wr["y1"], zo)]
                for i in range(len(pts) - 1):
                    a = 2 * i
                    quad = (a, a + 2, a + 3, a + 1)
                    xm = 0.5 * (V[a][0] + V[a + 2][0])
                    zm = 0.5 * (V[a][2] + V[a + 2][2])
                    glass.add([V[k] for k in quad], [(0, 1, 2, 3)], "window_glass", closed=False,
                              outward=(xm - (t["x_fwd"] - t["front_len"]), 0.0, zm))
        if t.get("aft_len", 0.0) > 0.0:
            # the aft end as one contiguous run: port aft arc then starboard aft arc
            k = len(ol) // 2
            rot = ol[k:] + ol[:k]
            xa_end = t["x_aft"] + t["aft_len"] + 0.01
            arcpts = [q for q in rot if q[0] <= xa_end]
            ctr = (t["x_aft"] + t["aft_len"], 0.0)
            for aw in t.get("aft_windows", []):
                if aw.get("kind") == "band":
                    V = []
                    for q in arcpts:
                        d = np.asarray(q) - np.asarray(ctr)
                        nq = d / max(np.linalg.norm(d), 1e-9)
                        V.append((q[0] + off * nq[0], q[1] + off * nq[1]))
                    for i in range(len(V) - 1):
                        (xa_, za_), (xb_, zb_) = V[i], V[i + 1]
                        quad = [(xa_, aw["y0"], za_), (xb_, aw["y0"], zb_), (xb_, aw["y1"], zb_), (xa_, aw["y1"], za_)]
                        xm, zm = 0.5 * (xa_ + xb_) - ctr[0], 0.5 * (za_ + zb_)
                        glass.add(quad, [(0, 1, 2, 3)], "window_glass", closed=False, outward=(xm, 0.0, zm))
                else:
                    panes_along(glass, arcpts, ctr, aw["y0"], aw["y1"], aw["pitch"], aw["width"], off)
        rg = t.get("roof_glass")
        if rg:
            # a glass roof (open decal) over the tier's outline short of its front arc, shrunk by
            # `inset` (affinely, so the rounded aft end stays convex for the fan)
            ins, w_ = rg["inset"], t["half_width"]
            al = t.get("aft_len", 0.0)
            xc = t["x_aft"] + al
            q = []
            for (x, z) in ol:
                if x > t["x_fwd"] - t["front_len"] + 1e-6:
                    continue
                xx = xc + (x - xc) * (al - ins) / al if (al > 0.0 and x < xc) else max(x, t["x_aft"] + ins)
                p = (xx, t["y1"] + off, z * (w_ - ins) / w_)
                if not q or math.hypot(p[0] - q[-1][0], p[2] - q[-1][2]) > 1e-3:
                    q.append(p)
            glass.add(q, [tuple(range(len(q)))], "window_glass", closed=False, outward=(0.0, 1.0, 0.0))
    # roof rails where no tier stands on the roof
    RR = S["roof_rail"]
    for name in RR["tiers"]:
        ol, t = outlines[name]
        above = [o for (o, tt) in outlines.values() if abs(tt["y0"] - t["y1"]) < 1e-6]
        y = t["y1"]
        ins = RR["inset"]
        pts = []
        for (x, z) in ol:
            zz = z - math.copysign(ins, z) if abs(z) > ins else z
            pts.append((x, zz))
        for i in range(len(pts)):
            p, q = np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
            if np.linalg.norm(q - p) < 0.05:
                continue
            m = 0.5 * (p + q)
            if any(point_in(o, m[0], m[1]) or point_in(o, m[0] + 0.3, m[1]) or point_in(o, m[0] - 0.3, m[1]) for o in above):
                continue
            strut(part, (p[0], y + RR["height"], p[1]), (q[0], y + RR["height"], q[1]), RR["thickness"], "rail_white", n=4)
            nseg = max(1, int(np.linalg.norm(q - p) / 2.5))
            for k in range(nseg):
                c = p + (q - p) * k / nseg
                strut(part, (c[0], y, c[1]), (c[0], y + RR["height"], c[1]), 0.5 * RR["thickness"], "rail_white", n=4)
    BW = S["bridge_wings"]
    for s in (1, -1):
        v, f, M = box_between((BW["x0"], BW["y0"], s * BW.get("inner_z", 9.5)), (BW["x1"], BW["y1"], s * BW["half_span"]))
        part.add(v, f, "super_white", M)
        zi = BW.get("window_inner_z", 10.6)
        for (q, n_) in (([(BW["x1"] + off, BW["window_y0"], s * zi), (BW["x1"] + off, BW["window_y0"], s * (BW["half_span"] - 0.2)),
                          (BW["x1"] + off, BW["window_y1"], s * (BW["half_span"] - 0.2)), (BW["x1"] + off, BW["window_y1"], s * zi)], (1.0, 0.0, 0.0)),
                        ([(BW["x0"] + 0.3, BW["window_y0"], s * (BW["half_span"] + off)), (BW["x1"] - 0.3, BW["window_y0"], s * (BW["half_span"] + off)),
                          (BW["x1"] - 0.3, BW["window_y1"], s * (BW["half_span"] + off)), (BW["x0"] + 0.3, BW["window_y1"], s * (BW["half_span"] + off))], (0.0, 0.0, s))):
            glass.add(q, [(0, 1, 2, 3)], "window_glass", closed=False, outward=n_)
    SB = S["sign_board"]
    for s in (1, -1):
        z = s * (5.5 + 0.05)
        v, f, M = box_between((SB["x0"], SB["y0"], z - 0.05), (SB["x1"], SB["y1"], z + 0.05))
        part.add(v, f, "sign_yellow", M)
    return outlines


# ---------------------------------------------------------------- funnel
def build_funnel(spec, part):
    F = spec["funnel"]
    rings = []
    n = 28
    for k, y in enumerate(np.linspace(F["y0"], F["y1"], 5)):
        dx = -F["rake"] * (y - F["y0"]) / (F["y1"] - F["y0"])
        ring = []
        for i in range(n):
            a = 2 * math.pi * i / n
            c, s = math.cos(a), math.sin(a)
            p = F["corner_power"]
            ring.append((F["x"] + dx + 0.5 * F["length"] * math.copysign(abs(c) ** (2 / p), c), y,
                         0.5 * F["width"] * math.copysign(abs(s) ** (2 / p), s)))
        rings.append(ring)
    part.add(*loft(rings), "funnel_black")
    for pp in F["pipes"]:
        v, f = cylinder(pp["r"], F["y1"] - 1.0, pp["top"], 16)
        part.add(v, f, "funnel_black", T(pp["x"] - F["rake"], 0.0, pp["z"]) @ Z_TO_Y())


# ---------------------------------------------------------------- masts, radar
def build_masts(spec, part, fit):
    M = spec["mast"]
    out = {}
    for key in ("fwd", "aft"):
        m = M[key]
        x, y0 = m["x"], m["y0"]
        v, f = frustum(m["radius"], 0.6 * m["radius"], 0.0, m["height"], 14)
        part.add(v, f, "radar_white", T(x, y0, 0.0) @ Z_TO_Y())
        sx, sy, sz = m["platform"]
        py = m["platform_y"]
        v, f, Mx = box_between((x - 0.5 * sx, py - sy, -0.5 * sz), (x + 0.5 * sx, py, 0.5 * sz))
        part.add(v, f, "radar_white", Mx)
        strut(part, (x - 0.5 * sx, py - sy, 0.0), (x - 1.8, y0, 0.0), 0.12, "radar_white", n=8)
        strut(part, (x + 0.5 * sx, py - sy, 0.0), (x + 1.8, y0, 0.0), 0.12, "radar_white", n=8)
        R_ = M["radar"]
        px, pyy, pz = R_["pedestal"]
        v, f, Mx = box_between((x - 0.5 * px, py, -0.5 * pz), (x + 0.5 * px, py + pyy, 0.5 * pz))
        part.add(v, f, "radar_white", Mx)
        yy = m["yard_y"]
        strut(part, (x, yy, -m["yard_half_span"]), (x, yy, m["yard_half_span"]), 0.1, "radar_white", n=8)
        top = y0 + m["height"]
        v, f, Mx = box_between((x - 0.25, top, -0.25), (x + 0.25, top + 0.5, 0.25))
        fit.add(v, f, "light_white", Mx)
        out[key] = {"radar": (x, py + pyy, 0.0), "masthead": (x + 0.26, top + 0.25, 0.0)}
    for d in M["domes"]:
        y0 = M["fwd"]["y0"]
        strut(part, (d["x"], y0, d["z"]), (d["x"], y0 + 1.6, d["z"]), 0.08, "radar_white", n=8)
        v, f = revolve([(0.0, -d["r"]), (0.7 * d["r"], -0.7 * d["r"]), (d["r"], 0.0), (0.7 * d["r"], 0.7 * d["r"]), (0.0, d["r"])], 16)
        part.add(v, f, "radar_white", T(d["x"], y0 + 1.6 + d["r"], d["z"]) @ Z_TO_Y())
    for w in M["whips"]:
        y0 = M["fwd"]["y0"]
        v, f = cylinder(0.04, 0.0, w["length"], 6)
        part.add(v, f, "radar_white", T(w["x"], y0, w["z"]) @ Z_TO_Y())
    return out


def build_radar(spec):
    R_ = spec["mast"]["radar"]
    part = Part()
    v, f = cylinder(0.3, 0.0, 0.35, 16)
    part.add(v, f, "radar_white", Z_TO_Y())
    L_ = R_["array_length"]
    w, h = R_["array_section"]
    v, f, M = box_between((-0.5 * w + 0.1, 0.35, -0.5 * L_), (0.5 * w + 0.1, 0.35 + h, 0.5 * L_))
    part.add(v, f, "radar_white", M)
    v, f, M = box_between((0.5 * w + 0.1, 0.4, -0.5 * L_ + 0.05), (0.5 * w + 0.11, 0.3 + h, 0.5 * L_ - 0.05))
    part.add(v, f, "black_plastic", M)
    return part


# ---------------------------------------------------------------- lifeboats
def boat_part(part, xc, zc, y_keel, length, beam, depth, canopy, mat_top, mat_hull, n_ring=16, n_st=20):
    rings = []
    for s in np.linspace(-1.0, 1.0, n_st):
        k = (1.0 - abs(s) ** 2.4) ** (1.0 / 2.4)
        k = max(k, 0.06)
        bw, dd, cc = 0.5 * beam * k, depth * (0.55 + 0.45 * k), canopy * (0.4 + 0.6 * k)
        ring = []
        for i in range(n_ring):
            a = 2 * math.pi * i / n_ring
            c, sn = math.cos(a), math.sin(a)
            y = y_keel + depth + (cc * abs(sn) ** 0.7 if sn > 0 else -dd * abs(sn) ** 0.8)
            ring.append((xc + 0.5 * length * s, y, zc + bw * math.copysign(abs(c) ** 0.6, c)))
        rings.append(ring)
    v, f = loft(rings)
    V = np.asarray(v)
    mats = [mat_top if V[list(fc), 1].mean() > y_keel + depth + 0.05 else mat_hull for fc in f]
    part.add_multi(v, f, mats)


def build_lifeboats(spec, part):
    LB = spec["lifeboats"]
    d6 = next(t for t in spec["superstructure"]["tiers"] if t.get("recess"))
    ya, yb = d6["y0"], d6["y1"] - 0.1
    for s in (1, -1):
        for b in LB["boats"]:
            boat_part(part, b["x"], s * LB["z_centre"], LB["y_keel"], b["length"], LB["beam"], LB["hull_depth"],
                      LB["canopy_height"], "lifeboat_orange", "lifeboat_white")
            for dx in (-0.36, 0.36):
                xb = b["x"] + dx * b["length"]
                strut(part, (xb, ya, s * 6.75), (xb, ya + LB["davit"]["arm_height"] * 0.7, s * 8.2),
                      LB["davit"]["arm_radius"], "steel_galv", n=8)
                strut(part, (xb, ya + LB["davit"]["arm_height"] * 0.7, s * 8.2), (xb, yb, s * 8.55),
                      LB["davit"]["arm_radius"], "steel_galv", n=8)
    RB = LB["rescue_boat"]
    s = -1 if spec["hull_details"]["rescue_recess"]["side"] == "port" else 1
    boat_part(part, RB["x"], s * (10.75 - 0.55), RB["y_keel"], RB["length"], RB["beam"], 0.7, RB["height"] - 0.7,
              "lifeboat_orange", "lifeboat_orange")


# ---------------------------------------------------------------- fittings
def build_fittings(spec, L, part):
    Mo = spec["mooring"]
    posts = {}
    for key in ("fwd", "aft"):
        x, zc = Mo[key]["x"], Mo[key]["z"]
        for s, side in ((-1, "port"), (1, "stbd")):
            z = s * zc
            y = L.deck(x, z)
            v, f, M = box_between((x - 0.9, y - 0.1, z - 0.4), (x + 0.9, y + 0.08, z + 0.4))
            part.add(v, f, "steel_galv", M)
            for dx in (-0.5 * Mo["spacing"], 0.5 * Mo["spacing"]):
                v, f = revolve([(0.0, -0.1), (Mo["radius"], -0.1), (Mo["radius"], Mo["height"] - 0.1),
                                (Mo["radius"] + 0.08, Mo["height"] - 0.05), (Mo["radius"] + 0.08, Mo["height"]), (0.0, Mo["height"])], 14)
                part.add(v, f, "steel_galv", T(x + dx, y, z) @ Z_TO_Y())
            posts[f"bollard_{key}_{side}"] = (x, y + Mo["height"], z)
    LI = spec["lights"]
    BW = spec["superstructure"]["bridge_wings"]
    for s, name, mat in ((-1, "nav_light_port", "light_red"), (1, "nav_light_stbd", "light_green")):
        nl = LI[name]
        z = s * (BW["half_span"] - 0.3)
        v, f, M = box_between((nl["x"] - 0.5, nl["y"] - 0.3, z - 0.25), (nl["x"] + 0.5, nl["y"] + 0.3, z + 0.25))
        part.add(v, f, mat, M)
    sl = LI["stern_light"]
    v, f, M = box_between((sl["x"] + 0.1, sl["y"] - 0.2, -0.2), (sl["x"] + 0.5, sl["y"] + 0.2, 0.2))
    part.add(v, f, "light_white", M)
    return posts


# ---------------------------------------------------------------- azimuth units, propellers
def build_azimuth(spec):
    A = spec["propulsion"]["azimuth"]
    part = Part()
    ya = A["axis_y"] - A["strut_top_y"]            # pod axis in the unit frame (origin at the strut top)
    c, t = A["strut_chord"], A["strut_half_thickness"]
    sec = []
    for k in range(14):
        a = 2 * math.pi * k / 14
        u = 0.5 * (1 + math.cos(a))
        xx = -0.6 * c + c * u
        th = t * 2.6 * math.sqrt(max(u, 0.0)) * (1 - u) + 0.02
        sec.append((xx, th * math.sin(a)))
    rings = [[(x, y, z) for (x, z) in sec] for y in (ya, 0.3)]
    part.add(*loft(rings), "antifouling")
    r, Lp = A["pod_radius"], A["pod_length"]
    prof = [(0.0, -0.5 * Lp - 0.6), (0.5 * r, -0.5 * Lp - 0.35), (0.9 * r, -0.5 * Lp), (r, -0.2 * Lp), (r, 0.3 * Lp),
            (0.8 * r, 0.5 * Lp), (0.0, 0.5 * Lp + 0.05)]
    v, f = revolve(prof, 24)
    part.add(v, f, "antifouling", T(0.0, ya, 0.0) @ Z_TO_X())
    return part


def build_propeller(spec):
    P = spec["propulsion"]["propeller"]
    part = Part()
    rh, hl, R_ = P["hub_radius"], P["hub_length"], 0.5 * P["diameter"]
    v, f = revolve([(0.0, -0.5 * hl), (rh, -0.5 * hl), (rh, 0.3 * hl), (0.5 * rh, 0.5 * hl + 0.3), (0.0, 0.5 * hl + 0.45)], 20)
    part.add(v, f, "bronze", Z_TO_X())
    pitch = math.radians(P["pitch_deg"])
    cmax = 0.42 * R_
    for b in range(P["blades"]):
        th0 = 2 * math.pi * b / P["blades"]
        rings = []
        for r in np.linspace(0.8 * rh, R_, 9):
            s = (r - 0.8 * rh) / (R_ - 0.8 * rh)
            c = cmax * (0.55 + 0.45 * math.sin(math.pi * min(1.0, s * 0.95 + 0.05))) * (1.0 - 0.75 * s ** 4)
            tk = 0.08 * (1.0 - 0.8 * s) + 0.01
            skew = 0.15 * cmax * s * s
            phi = math.atan2(math.tan(pitch) * R_ * 0.7, r)
            er = np.array([0.0, math.cos(th0), math.sin(th0)])
            et = np.array([0.0, -math.sin(th0), math.cos(th0)])
            ex = np.array([1.0, 0.0, 0.0])
            cd = math.cos(phi) * et + math.sin(phi) * ex
            td = -math.sin(phi) * et + math.cos(phi) * ex
            rings.append([r * er + (0.5 * c * math.cos(2 * math.pi * k / 8) + skew) * cd + tk * math.sin(2 * math.pi * k / 8) * td
                          for k in range(8)])
        part.add(*loft(rings), "bronze")
    return part


# ---------------------------------------------------------------- the name (Blender font -> mesh)
def text_mesh(text, cap_height, extrude):
    import bpy
    cu = bpy.data.curves.new("name_tmp", "FONT")
    cu.body = text
    cu.size = cap_height / 0.72
    cu.extrude = extrude
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.resolution_u = 3
    ob = bpy.data.objects.new("name_tmp", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    uvw = np.array([v.co[:] for v in me.vertices])
    faces = [tuple(p.vertices) for p in me.polygons]
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return uvw, faces


def name_part(spec, L):
    N = spec["ship_name"]
    part = Part()
    uvw, faces = text_mesh(N["text"], N["cap_height"], N["extrude"])
    xc, yc = N["bow"]["x_centre"], N["bow"]["y_centre"]
    for s in (-1, 1):
        V = []
        for (u, v, w) in uvw:
            x, y = xc + s * u, yc + v
            V.append((x, y, s * (L.shell_z(x, y) + N["offset"] + w)))
        part.add(V, faces, "name_white", closed=False)
    uvw, faces = text_mesh(N["text"], N["stern"]["cap_height"], N["extrude"])
    V = [(L.x0 - N["offset"] - w, N["stern"]["y_centre"] + v, u) for (u, v, w) in uvw]
    part.add(V, faces, "name_white", closed=False)
    return part


def name_frame(spec, L, s):
    N = spec["ship_name"]
    x, y = N["bow"]["x_centre"], N["bow"]["y_centre"]
    e = 0.05
    p = np.array([x, y, s * L.shell_z(x, y)])
    dx = np.array([2 * e, 0.0, s * (L.shell_z(x + e, y) - L.shell_z(x - e, y))])
    dy = np.array([0.0, 2 * e, s * (L.shell_z(x, y + e) - L.shell_z(x, y - e))])
    n = np.cross(dx, dy) * s
    n /= np.linalg.norm(n)
    rd = s * dx / np.linalg.norm(dx)
    up = np.cross(n, rd)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = rd, up, n, p
    return M


# ---------------------------------------------------------------- assembly
def build_geometry(spec):
    L = Lines(spec)
    hull, sup, glass, fun, mast, boats, fit = Part(), Part(), Part(), Part(), Part(), Part(), Part()
    build_hull(spec, L, hull)
    build_bulb(spec, hull)
    tris = np.concatenate(hull.tris)
    build_hull_details(spec, L, hull)
    build_superstructure(spec, L, sup, glass)
    build_funnel(spec, fun)
    mp = build_masts(spec, mast, fit)
    build_lifeboats(spec, boats)
    posts = build_fittings(spec, L, fit)
    parts = {"hull": hull, "superstructure": sup, "windows": glass, "funnel": fun, "mast": mast,
             "lifeboats": boats, "deck_fittings": fit}
    return parts, tris, {"L": L, "mast": mp, "posts": posts}


def build_blender(spec, parts, hydro, info):
    import bpy
    import bmesh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    L = info["L"]
    npart = name_part(spec, L)
    base = len(parts["hull"].verts)
    parts["hull"].verts.extend(npart.verts)
    parts["hull"].faces.extend(tuple(base + k for k in f) for f in npart.faces)
    parts["hull"].mats.extend(npart.mats)
    d = hydro["conditions"][hydro["design_condition"]]
    m_dry, com = mass_condition(spec, "lightship")
    root = new_object("trollfjord", None, mats, None, I, props={
        "frame": "X forward, Y up, Z starboard; origin = centreline x baseline x mid-LOA",
        "spec": "trollfjord_spec.json", "hydro": "trollfjord_hydro.json",
        "mass_lightship": m_dry, "com_lightship": [round(c, 4) for c in com],
        "design_waterline_y": d["waterline_y_at_x0"], "design_trim_deg": d["trim_deg"]})
    objs = {}
    for name, sharp in (("hull", 30.0), ("superstructure", 30.0), ("windows", 30.0), ("funnel", 35.0),
                        ("mast", 30.0), ("lifeboats", 40.0), ("deck_fittings", 30.0)):
        objs[name] = new_object(name, parts[name], mats, root, I, sharp_deg=sharp, tile=2.0)
    me = objs["hull"].data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=bm.edges)
    bm.to_mesh(me)
    bm.free()
    me.set_sharp_from_angle(angle=math.radians(30.0))
    me.update()
    R_ = spec["mast"]["radar"]
    for key in ("fwd", "aft"):
        new_object(f"radar_{key}", build_radar(spec), mats, root, T(*info["mast"][key]["radar"]),
                   props={"axis": "+Y", "rpm": R_["rpm"], "range_max": R_["range_max"]})
    A = spec["propulsion"]["azimuth"]
    P = spec["propulsion"]["propeller"]
    for s, side in ((-1, "port"), (1, "stbd")):
        unit = new_object(f"rudder_{side}", build_azimuth(spec), mats, root, T(A["x"], A["strut_top_y"], s * A["z"]),
                          props={"axis": "+Y", "kind": "azimuth thruster (steers the thrust)", "limit_deg": A["limit_deg"],
                                 "positive": "right-handed about +Y: the thrust swings to port, the stern to starboard"})
        prel = (P["x_rel"], A["axis_y"] - A["strut_top_y"], 0.0)
        new_object(f"propeller_{side}", build_propeller(spec), mats, unit, T(*prel),
                   props={"axis": "+X", "blades": P["blades"], "diameter": P["diameter"],
                          "positive": "right-handed about +X", "parent": "its azimuth unit"})
        new_object(f"prop_thrust_{side}", None, mats, unit, T(*prel),
                   props={"thrust_axis": "+X", "bollard_thrust_n": P["bollard_thrust_n"], "steered_by": f"rudder_{side}",
                          "engine_power_kw": 0.5 * spec["propulsion"]["engine_power_kw"]})
    for bt in spec["propulsion"]["bow_thrusters"]:
        new_object(bt["name"], None, mats, root, T(bt["x"], bt["y"], 0.0),
                   props={"thrust_axis": "+Z", "thrust_n": bt["thrust_n"], "tunnel_radius": bt["tunnel_radius"]})
    st = spec["propulsion"]["stern_thruster"]
    new_object("stern_thruster", None, mats, root, T(st["x"], st["y"], 0.0),
               props={"thrust_axis": "+Z", "thrust_n": st["thrust_n"], "tunnel_radius": st["tunnel_radius"]})
    B = spec["bridge"]
    new_object("bridge", None, mats, root, sensor_frame(B["position"], B["look"]), props={"convention": "look -Z, up +Y"})
    LI = spec["lights"]
    BW = spec["superstructure"]["bridge_wings"]
    for s, name in ((-1, "nav_light_port"), (1, "nav_light_stbd")):
        nl = LI[name]
        new_object(name, None, mats, root, sensor_frame((nl["x"] + 0.5, nl["y"], s * (BW["half_span"] - 0.3)), (1.0, 0.0, 0.0)),
                   props={"convention": "look -Z, up +Y", "colour": nl["colour"], "sector_deg": nl["sector_deg"]})
    for key in ("fwd", "aft"):
        ml = LI[f"masthead_{key}"]
        new_object(f"masthead_{key}", None, mats, root, sensor_frame(info["mast"][key]["masthead"], (1.0, 0.0, 0.0)),
                   props={"convention": "look -Z, up +Y", "colour": ml["colour"], "sector_deg": ml["sector_deg"]})
    sl = LI["stern_light"]
    new_object("stern_light", None, mats, root, sensor_frame((sl["x"] + 0.1, sl["y"], 0.0), (-1.0, 0.0, 0.0)),
               props={"convention": "look -Z, up +Y", "colour": sl["colour"], "sector_deg": sl["sector_deg"]})
    for name, p in info["posts"].items():
        new_object(name, None, mats, root, T(*p), props={"role": "mooring bollard top"})
    for s, name in ((-1, "name_port"), (1, "name_stbd")):
        new_object(name, None, mats, root, name_frame(spec, L, s),
                   props={"text": spec["ship_name"]["text"], "convention": "local +Z = outward hull normal, +X = reading direction"})
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="trollfjord_spec.json")
    ap.add_argument("--out", default="trollfjord.glb")
    ap.add_argument("--hydro-out", default="trollfjord_hydro.json")
    ap.add_argument("--hydro-only", action="store_true")
    ap.add_argument("--probe", action="store_true", help="print displacement, KB, KM at level waterlines and stop")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    parts, tris, info = build_geometry(spec)
    if a.probe:
        hy = Hydro(tris, spec["hydrostatics"]["grid"])
        print("closure", hy.closure)
        for h in (4.6, 4.8, 5.0, 5.1, 5.2):
            V, xB, yB = hy.state(h, 0.0)
            r = hy.report(h, 0.0, V * 1025.0, (xB, 0.0), 1025.0, 9.81)
            print(f"T={h:.2f} V={V:.0f} m3 disp={V * 1.025:.0f} t LCB={xB:+.2f} KB={yB:.3f} KM_T={r['gm_t']:.3f} "
                  f"LWL={r['lwl']:.1f} BWL={r['bwl']:.2f} Cb={V / (r['lwl'] * r['bwl'] * h):.3f}")
        return
    com = mass_check(spec)
    print(f"[trollfjord] spec ok; lightship CoM from budget = ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
    used = set(m for p in parts.values() for m in p.mats) | {"name_white", "bronze"}
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    hydro = hydrostatics(spec, tris, os.path.basename(a.spec))
    for cond, r in hydro["conditions"].items():
        print(f"[trollfjord] {cond:10s} {r['mass'] / 1000:8.1f} t  disp {r['displacement_m3']:.0f} m3  waterline y "
              f"{r['waterline_y_at_x0']:.3f}  max draft {r['draft_max']:.3f}  trim {r['trim_deg']:+.3f} deg  "
              f"LCB {r['lcb']:+.2f}  KB {r['kb']:.3f}  GM_T {r['gm_t']:.3f}  GM_L {r['gm_l']:.0f}  "
              f"LWL {r['lwl']:.1f}  BWL {r['bwl']:.2f}  Awp {r['waterplane_area']:.0f}")
    with open(path(a.hydro_out), "w", encoding="utf-8") as fh:
        json.dump(hydro, fh, indent=1)
    print(f"[trollfjord] wrote {path(a.hydro_out)}")
    check_gates(spec, hydro)
    print("[trollfjord] gates PASS (draft, trim, GM_T, closed primitives, closed buoyant mesh)")
    if a.hydro_only:
        return
    build_blender(spec, parts, hydro, info)
    n = export(path(a.out))
    print(f"[trollfjord] wrote {path(a.out)}  triangles={n}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

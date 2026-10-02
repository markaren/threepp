"""Build the sjark (a Norwegian 10.99 m GRP coastal fishing boat) from sjark_spec.json:
sjark.glb (via Blender) and sjark_hydro.json.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup --python build_sjark_blender.py -- --spec sjark_spec.json --out sjark.glb

Seeded liveries (colours, registration, gear) for synthetic-data diversity, one Blender run:
    blender --background --factory-startup --python build_sjark_blender.py -- --seeds 1,2,3,4,5
    -> out/sjark_s<N>.glb (the hull and the hydrostatics are the default build's)

Hydrostatics only (plain Python + numpy, no Blender; writes sjark_hydro.json):
    python build_sjark_blender.py --hydro-only

The mesh primitives, the hydrostatic integrator and the Blender export are the
Mariner generator's (../usv/build_mariner_blender.py); this script adds the sjark.
All geometry is built with numpy in the spec's vessel frame (X forward, Y up,
Z starboard, metres) and converted to Blender's Z-up only when a mesh or a
transform is handed to bpy. The glTF exporter's Y-up conversion maps it straight
back, so every number in the spec is the number in the .glb.

Node hierarchy (all names exact):
    sjark                        root, identity, origin = centreline x baseline x mid-LOA
      hull                       lofted hard-chine hull (antifouling / boot-top / topside split at
                                 the paint levels, deck), bar keel and skeg, skeg shoe, rubbing
                                 strakes, stem band, registration text on both bows
      bulwark                    side and transom bulwarks, cap rails, freeing ports, bow rail
      wheelhouse                 house, roof, roof rail, aft door, sidelight boxes
      windows                    dark glass panes (own material)
      mast                       pole mast, radar platform and pedestal, yard, masthead and
                                 all-round light, VHF whips, GNSS pucks, searchlight
      radar                      open-array antenna, spins about local +Y
      deck_fittings              bollards, fish hatch, anchor winch, net stacker, hauling roller,
                                 derrick post, stern light post
      net_hauler                 net hauler (or line hauler in a seeded build), starboard side
      derrick                    landing derrick boom + hook block, turns about +Y at its pivot
      rudder                     turns about local +Y (the stock)
      propeller                  4 blades + hub, spins about local +X (the shaft axis)
      prop_thrust, bow_thruster  empties, identity rotation (thrust axes in the extras)
      bollard_fwd_port, bollard_fwd_stbd, bollard_aft_port, bollard_aft_stbd, tow_point
                                 mooring empties at the bollard tops / towing eye
      camera_mast                empty where a demo can mount a camera (looks down local -Z, +Y up)
      nav_light_port, nav_light_stbd, masthead_light, stern_light   light empties (look -Z, up +Y)
      reg_mark_port, reg_mark_stbd   empties at the centre of each registration mark
                                 (local +Z = outward hull normal, +X = reading direction)

The hydrostatics integrate the buoyant closed meshes (hull shell, keel/skeg) column
by column with the Mariner's winding-number integrator, solve the floating position
for each loading condition and assert the design draft, trim and GM_T gates.
"""
import argparse
import copy
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "usv"))
from build_mariner_blender import (Part, T, Z_TO_X, Z_TO_Y, box_between, clip_slab,  # noqa: E402
                                   cylinder, frame_from_forward_up, frustum, hydrostatics, loft, make_materials,
                                   mass_condition, new_object, pchip, revolve, sensor_frame, strut)


def table(t):
    a = np.asarray(t, float)
    return pchip(a[:, 0], a[:, 1])


# ---------------------------------------------------------------- hull lines
class Lines:
    """The hard-chine hull as functions of x.

    Section (starboard half): keel -> convex bottom -> chine -> flared topside -> sheer ->
    cambered deck. Both paint levels (antifouling top, boot-top) are dedicated points of the
    bottom chain and of the topside chain (clamped to the chine when the level lies in the
    other chain), so the loft carries them at a fixed ring index and the paint lines are
    exact hull edges."""

    NB = (4, 1, 4)        # bottom chain segments: keel..L1, L1..L2, L2..chine
    NT = (2, 1, 6)        # topside chain segments: chine..L1, L1..L2, L2..sheer
    DECK_Q = (0.66, 0.33)

    def __init__(self, spec):
        H = spec["hull"]
        self.H = H
        self.keel = table(H["keel_profile"])
        self.zc, self.yc = table(H["chine_halfbreadth"]), table(H["chine_height"])
        self.zs, self.ys = table(H["sheer_halfbreadth"]), table(H["sheer_height"])
        self.x0, self.x1 = H["transom_x"], H["stem_x"]
        P = spec["paint"]
        self.levels = (P["antifouling_top_y"], P["boot_top_y"])

    def bottom_pt(self, x, t):
        yk, zc, yc = self.keel(x), self.zc(x), self.yc(x)
        return zc * t, yk + (yc - yk) * t - self.H["bottom_convexity"] * math.sin(math.pi * t) * min(1.0, zc)

    def topside_pt(self, x, t):
        zc, yc, zs, ys = self.zc(x), self.yc(x), self.zs(x), self.ys(x)
        return zc + (zs - zc) * t ** self.H["topside_flare_power"], yc + (ys - yc) * t

    def shell_z(self, x, y):
        """|z| of the hull surface at (x, y) (topside or bottom)."""
        yc = self.yc(x)
        if y >= yc:
            t = min(1.0, (y - yc) / (self.ys(x) - yc))
            return self.topside_pt(x, t)[0]
        lo, hi = 0.0, 1.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if self.bottom_pt(x, m)[1] < y:
                lo = m
            else:
                hi = m
        return self.bottom_pt(x, 0.5 * (lo + hi))[0]

    def _t_at(self, fn, x, y):
        a, b = fn(x, 0.0)[1], fn(x, 1.0)[1]
        eps = 1e-4
        if y <= a:
            return eps
        if y >= b:
            return 1.0 - eps
        lo, hi = 0.0, 1.0
        for _ in range(50):
            m = 0.5 * (lo + hi)
            if fn(x, m)[1] < y:
                lo = m
            else:
                hi = m
        return min(1.0 - eps, max(eps, 0.5 * (lo + hi)))

    def _chain(self, fn, x, counts):
        t1, t2 = (self._t_at(fn, x, L) for L in self.levels)
        t2 = max(t2, t1 + 1e-5)
        knots = [0.0, t1, t2, 1.0]
        pts = []
        for (a, b), n in zip(zip(knots[:-1], knots[1:]), counts):
            for k in range(n):
                pts.append(fn(x, a + (b - a) * k / n))
        return pts                                  # without the chain's last point

    def deck(self, x, z):
        zs = self.zs(x)
        q = min(1.0, abs(z) / zs) if zs > 1e-6 else 1.0
        return self.ys(x) + self.H["deck_camber"] * (1.0 - q * q)

    def section_half(self, x):
        """[(z, y, kind)] from the keel (z = 0) to the deck centre; kind of the segment
        starting at the point: 'shell' or 'deck'."""
        pts = [(z, y, "shell") for (z, y) in self._chain(self.bottom_pt, x, self.NB)]
        pts += [(z, y, "shell") for (z, y) in self._chain(self.topside_pt, x, self.NT)]
        zs, ys = self.zs(x), self.ys(x)
        pts.append((zs, ys, "deck"))
        for q in self.DECK_Q:
            pts.append((zs * q, self.deck(x, zs * q), "deck"))
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


def paint_mat(y, levels):
    return "antifouling" if y < levels[0] else ("boot_top" if y < levels[1] else "hull_topside")


def build_hull(spec, L, part):
    xs = stations(spec)
    rings, kinds = [], None
    for x in xs:
        r, kinds = L.ring(x)
        rings.append(r)
    v, f = loft(rings, caps=False)
    N = len(rings[0])
    V = np.asarray(v)
    mats = []
    for q, face in enumerate(f):
        if kinds[q % N] == "deck":
            mats.append("deck_grey")
        else:
            mats.append(paint_mat(float(V[list(face), 1].mean()), L.levels))
    # transom: the aft ring as horizontal paint bands (vertical, no hydro contribution)
    ring0 = [(p[1], p[2]) for p in rings[0]]
    bands = [(-9.0, L.levels[0], "antifouling"), (L.levels[0], L.levels[1], "boot_top"), (L.levels[1], 9.0, "hull_topside")]
    for (a, b, mat) in bands:
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
    mats.append("hull_topside")
    part.add_multi(v, f, mats, buoyant=True)
    return xs


def build_keel(spec, L, part):
    K = spec["appendages"]["keel"]
    bot, ht = table(K["bottom"]), table(K["half_thickness"])
    rings = []
    for x in np.linspace(K["x_aft"], K["x_fwd"], 50):
        t = ht(x)
        yb = bot(x)
        yt = L.keel(x) + K["embed"] + 0.3 * t
        rings.append([(x, yb, -0.6 * t), (x, yb, 0.6 * t), (x, yb + 0.04, t), (x, yt, t), (x, yt, -t), (x, yb + 0.04, -t)])
    part.add_multi(*loft(rings), ["antifouling"] * (6 * 49 + 2), buoyant=True)
    S = spec["appendages"]["skeg_shoe"]
    v, f, M = box_between((S["x0"], S["y0"], -S["half_width"]), (S["x1"], S["y1"], S["half_width"]))
    part.add(v, f, "antifouling", M, buoyant=True)
    P = spec["propulsion"]["propeller"]
    v, f = cylinder(P["shaft_radius"], P["shaft_x0"], P["x"], 12)
    part.add(v, f, "stainless", T(0.0, P["axis_y"], 0.0) @ Z_TO_X())


def build_strakes(spec, L, part):
    RS = spec["hull"]["rubbing_strake"]
    h, th, yb = RS["height"], RS["thickness"], RS["y_below_sheer"]
    xs = [x for x in stations(spec) if L.zs(x) > 0.12] + [L.x1 - 0.02]
    for s in (1, -1):
        rings = []
        for x in xs:
            y1 = L.ys(x) - yb
            y0 = y1 - h
            zi = L.shell_z(x, y0) - 0.02
            zo = L.zs(x) + th
            rings.append([(x, y0, s * zi), (x, y0, s * (zo - 0.01)), (x, y1, s * zo), (x, y1, s * (zi - 0.02))])
        part.add(*loft(rings), "rub_black")
    x0 = L.x0
    y1 = L.ys(x0) - yb
    v, f, M = box_between((x0 - th, y1 - h, -L.zs(x0) - th), (x0 + 0.02, y1, L.zs(x0) + th))
    part.add(v, f, "rub_black", M)
    # stem band up the stem face
    x1 = L.x1
    v, f, M = box_between((x1 - 0.02, L.keel(x1) - 0.3, -0.03), (x1 + th, L.ys(x1) + 0.02, 0.03))
    part.add(v, f, "stainless", M)


# ---------------------------------------------------------------- bulwark
def build_bulwark(spec, L, part):
    B = spec["bulwark"]
    hb = table(B["height"])
    th, fl = B["thickness"], B["flare_per_m"]
    cap = B["cap"]
    xs = [x for x in stations(spec) if L.zs(x) > th + 0.03]
    for s in (1, -1):
        rings, caps = [], []
        for x in xs:
            zs, ys, h = L.zs(x), L.ys(x), hb(x)
            zt = zs + fl * h
            rings.append([(x, ys - 0.03, s * zs), (x, ys + h, s * zt), (x, ys + h, s * (zt - th)), (x, ys - 0.03, s * (zs - th))])
            cw = cap["width"]
            caps.append([(x, ys + h - 0.01, s * (zt - th - 0.5 * (cw - th))), (x, ys + h - 0.01, s * (zt + 0.5 * (cw - th))),
                         (x, ys + h + cap["height"], s * (zt + 0.5 * (cw - th))),
                         (x, ys + h + cap["height"], s * (zt - th - 0.5 * (cw - th)))])
        n = len(rings) - 1
        part.add_multi(*loft(rings), ["bulwark_paint", "bulwark_paint", "bulwark_inner", "bulwark_inner"] * n
                       + ["bulwark_paint", "bulwark_paint"])
        part.add(*loft(caps), "cap_rail")
        # freeing ports: dark decals on the outer face just above the deck
        FP = B["freeing_ports"]
        for xc in FP["x"]:
            quad = []
            for (dx, dy) in ((-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)):
                x = xc + dx * FP["length"]
                y = L.ys(x) + 0.02 + dy * FP["height"]
                quad.append((x, y, s * (L.zs(x) + fl * (y - L.ys(x) + 0.03) + 0.004)))
            part.add(quad, [(0, 1, 2, 3)], "freeing_port", closed=False, outward=(0.0, 0.0, s))
    # stem piece closing the two side walls at the bow
    xe = max(xs)
    ye, he = L.ys(xe), hb(xe)
    v, f, M = box_between((xe - 0.1, ye - 0.03, -L.zs(xe) - 0.004), (L.x1 + 0.02, ye + he, L.zs(xe) + 0.004))
    part.add(v, f, "bulwark_paint", M)
    v, f, M = box_between((xe - 0.12, ye + he - 0.01, -L.zs(xe) - 0.03), (L.x1 + 0.04, ye + he + cap["height"], L.zs(xe) + 0.03))
    part.add(v, f, "cap_rail", M)
    # transom bulwark and its cap
    x0 = L.x0
    zs, ys, h = L.zs(x0), L.ys(x0), B["transom_height"]
    v, f, M = box_between((x0, ys - 0.03, -zs), (x0 + th, ys + h, zs))
    part.add(v, f, "bulwark_paint", M)
    v, f, M = box_between((x0 - 0.025, ys + h - 0.01, -zs - 0.05), (x0 + th + 0.025, ys + h + cap["height"], zs + 0.05))
    part.add(v, f, "cap_rail", M)
    # bow rail (pulpit) above the foredeck bulwark
    BR = B["bow_rail"]
    for s in (1, -1):
        xr = list(np.arange(BR["x0"], L.x1 - 0.25, 0.3)) + [L.x1 - 0.12]
        path = []
        for x in xr:
            zt = L.zs(x) + fl * hb(x) - 0.5 * th
            path.append((x, L.ys(x) + hb(x) + BR["height"], s * max(zt - 0.02, 0.03)))
        for p, q in zip(path[:-1], path[1:]):
            strut(part, p, q, BR["radius"], "stainless", n=10)
        for x in np.arange(BR["x0"], L.x1 - 0.3, BR["stanchion_dx"]):
            zt = L.zs(x) + fl * hb(x) - 0.5 * th
            y0 = L.ys(x) + hb(x) + cap["height"] - 0.01
            strut(part, (x, y0, s * zt), (x, y0 + BR["height"], s * zt), BR["radius"], "stainless", n=10)
    tip = (L.x1 - 0.12, L.ys(L.x1 - 0.12) + hb(L.x1 - 0.12) + BR["height"], 0.0)
    strut(part, (tip[0], tip[1] - BR["height"] - 0.1, 0.0), tip, BR["radius"], "stainless", n=10)


# ---------------------------------------------------------------- wheelhouse + windows
class House:
    def __init__(self, spec, L):
        W = spec["wheelhouse"]
        self.W = W
        self.xf, self.xa = W["x_front"], W["x_aft"]
        self.wf, self.wa = W["half_width_front"], W["half_width_aft"]
        self.y_f, self.y_a = L.deck(self.xf, 0.0) - 0.08, L.deck(self.xa, 0.0) - 0.08
        self.y_top = W["floor_y"] + W["height"]
        th, rk = W["side_tumblehome"], W["front_rake"]
        # corners: bottom ring (port aft, port front, stbd front, stbd aft), top ring same order
        self.bot = [(self.xa, self.y_a, -self.wa), (self.xf, self.y_f, -self.wf),
                    (self.xf, self.y_f, self.wf), (self.xa, self.y_a, self.wa)]
        self.top = [(self.xa, self.y_top, -(self.wa - th)), (self.xf + rk, self.y_top, -(self.wf - th)),
                    (self.xf + rk, self.y_top, self.wf - th), (self.xa, self.y_top, self.wa - th)]

    def face(self, k):
        """Face k (0 port side, 1 front, 2 stbd side, 3 aft) as (b0, b1, t0, t1) seen from outside,
        left to right: b = bottom corners, t = top corners."""
        B, Tp = np.asarray(self.bot), np.asarray(self.top)
        order = {0: (1, 0), 1: (2, 1), 2: (3, 2), 3: (0, 3)}[k]
        a, b = order
        return B[a], B[b], Tp[a], Tp[b]

    def point(self, k, u, y):
        b0, b1, t0, t1 = self.face(k)
        pb, pt = b0 + (b1 - b0) * u, t0 + (t1 - t0) * u
        s = (y - pb[1]) / (pt[1] - pb[1])
        return pb + (pt - pb) * s

    def normal(self, k):
        b0, b1, t0, t1 = self.face(k)
        n = np.cross(b1 - b0, t0 - b0)
        return n / np.linalg.norm(n)


def build_wheelhouse(spec, L, house, part, glass):
    W = spec["wheelhouse"]
    v, f = loft([house.bot, house.top])
    part.add_multi(v, f, ["house_paint"] * 4 + ["house_paint", "roof_white"])
    # roof slab with overhang
    R_ = W["roof"]
    o = R_["overhang"]
    tp_ = np.asarray(house.top)
    outline = []
    for (x, y, z) in tp_:
        outline.append((x + (o if x > 0 else -0.5 * o), z + math.copysign(o, z)))
    y0, y1 = house.y_top - 0.02, house.y_top + R_["thickness"]
    rings = [[(x, y0, z) for (x, z) in outline], [(x, y1, z) for (x, z) in outline]]
    part.add(*loft(rings), "roof_white")
    house.roof_y = y1
    house.roof_outline = outline
    # roof rail
    RR = W["roof_rail"]
    pts = [(x - (0.06 if x > 0 else -0.06), z - math.copysign(0.06, z)) for (x, z) in outline]
    for i in range(4):
        (xa, za), (xb, zb) = pts[i], pts[(i + 1) % 4]
        n = max(1, int(math.hypot(xb - xa, zb - za) / RR["stanchion_dx"]))
        strut(part, (xa, y1 + RR["height"], za), (xb, y1 + RR["height"], zb), RR["radius"], "stainless", n=10)
        for k in range(n):
            t = k / n
            x, z = xa + (xb - xa) * t, za + (zb - za) * t
            strut(part, (x, y1, z), (x, y1 + RR["height"], z), RR["radius"], "stainless", n=8)
    # windows
    Wn = W["windows"]
    head = house.y_top - Wn["head_below_roof"]
    sill = head - Wn["depth"]
    off = 0.008

    def pane(k, u0, u1, ya, yb, mat="window_glass", into=glass):
        n = house.normal(k)
        q = [house.point(k, u0, ya), house.point(k, u1, ya), house.point(k, u1, yb), house.point(k, u0, yb)]
        q = [p + off * n for p in q]
        into.add(q, [(0, 1, 2, 3)], mat, closed=False, outward=tuple(n))

    def row(k, n, inset, gap, ya, yb, skip=()):
        span = 1.0 - 2 * inset
        w = (span - (n - 1) * gap) / n
        for i in range(n):
            if i in skip:
                continue
            u0 = inset + i * (w + gap)
            pane(k, u0, u0 + w, ya, yb)
    row(1, Wn["front_panes"], 0.03, 0.015, sill, head)
    for k in (0, 2):
        row(k, Wn["side_panes"], 0.1, 0.14, head - Wn["side_depth"], head)
    row(3, Wn["aft_panes"], 0.08, 0.5, head - Wn["aft_depth"], head)
    # aft door (decal on the aft face)
    D = W["door"]
    wa = house.wa
    u0, u1 = 0.5 + D["x0"] / (2 * wa), 0.5 + D["x1"] / (2 * wa)
    pane(3, u0, u1, house.y_a + 0.12, house.y_a + D["height"], mat="door_dark", into=part)


# ---------------------------------------------------------------- mast, radar, lights
def build_mast(spec, house, part):
    M = spec["mast"]
    ry = house.roof_y
    x = M["x"]
    v, f = frustum(M["radius_base"], M["radius_top"], 0.0, M["height_above_roof"], 16)
    part.add(v, f, "radar_white", T(x, ry, 0.0) @ Z_TO_Y())
    py = ry + M["platform_y_above_roof"]
    sx, sy, sz = M["platform"]
    v, f, Mx = box_between((x - 0.1, py - sy, -0.5 * sz), (x - 0.1 + sx, py, 0.5 * sz))
    part.add(v, f, "radar_white", Mx)
    strut(part, (x + 0.35, py - sy, 0.0), (x + 0.6, ry, 0.0), 0.03, "radar_white", n=10)
    R_ = M["radar"]
    px, pyy, pz = R_["pedestal"]
    v, f, Mx = box_between((x + 0.2 - 0.5 * px, py, -0.5 * pz), (x + 0.2 + 0.5 * px, py + pyy, 0.5 * pz))
    part.add(v, f, "radar_white", Mx)
    Y = M["yard"]
    yy = ry + Y["y_above_roof"]
    strut(part, (x, yy, -Y["half_span"]), (x, yy, Y["half_span"]), Y["radius"], "radar_white", n=10)
    # masthead light (forward face) and all-round light at the top
    v, f, Mx = box_between((x + 0.05, yy + 0.05, -0.05), (x + 0.15, yy + 0.2, 0.05))
    part.add(v, f, "light_white", Mx)
    top = ry + M["height_above_roof"]
    v, f = cylinder(0.045, 0.0, 0.12, 12)
    part.add(v, f, "light_white", T(x, top, 0.0) @ Z_TO_Y())
    for a in M["vhf"]:
        b = np.asarray(a["base"]) + (0.0, ry, 0.0)
        v, f = cylinder(0.012, 0.0, a["length"], 8)
        part.add(v, f, "radar_white", T(*b) @ Z_TO_Y())
        v, f = cylinder(0.03, 0.0, 0.12, 10)
        part.add(v, f, "black_plastic", T(*b) @ Z_TO_Y())
    for g in M["gnss"]:
        b = np.asarray(g["base"]) + (0.0, ry, 0.0)
        v, f = revolve([(0.0, 0.0), (0.08, 0.0), (0.08, 0.04), (0.05, 0.08), (0.0, 0.085)], 16)
        part.add(v, f, "radar_white", T(*b) @ Z_TO_Y())
    sl = np.asarray(M["searchlight"]["base"]) + (0.0, ry, 0.0)
    v, f = cylinder(0.04, 0.0, 0.18, 10)
    part.add(v, f, "black_plastic", T(*sl) @ Z_TO_Y())
    v, f = revolve([(0.0, -0.14), (0.1, -0.12), (0.13, 0.1), (0.0, 0.1)], 16)
    part.add(v, f, "black_plastic", T(sl[0], sl[1] + 0.3, sl[2]) @ Z_TO_X())
    return {"radar": (x + 0.2, py + pyy, 0.0), "masthead": (x + 0.15, yy + 0.125, 0.0), "top": top}


def build_radar():
    R_ = spec_cache["mast"]["radar"]
    part = Part()
    v, f = cylinder(0.1, 0.0, 0.14, 16)
    part.add(v, f, "radar_white", Z_TO_Y())
    L_ = R_["array_length"]
    w, h = R_["array_section"]
    v, f, M = box_between((-0.5 * w + 0.03, 0.14, -0.5 * L_), (0.5 * w + 0.03, 0.14 + h, 0.5 * L_))
    part.add(v, f, "radar_white", M)
    v, f, M = box_between((0.5 * w + 0.03, 0.16, -0.5 * L_ + 0.02), (0.5 * w + 0.034, 0.12 + h, 0.5 * L_ - 0.02))
    part.add(v, f, "black_plastic", M)
    return part


# ---------------------------------------------------------------- deck fittings and gear
def build_fittings(spec, L, house, part):
    G = spec["gear"]
    B = spec["bulwark"]
    bo = G["bollards"]
    posts = {}
    for key, x in (("fwd", bo["fwd_x"]), ("aft", bo["aft_x"])):
        for s, side in ((-1, "port"), (1, "stbd")):
            z = s * (L.zs(x) - B["thickness"] - bo["inset"])
            y = L.deck(x, z)
            v, f, M = box_between((x - 0.22, y - 0.05, z - 0.09), (x + 0.22, y + 0.03, z + 0.09))
            part.add(v, f, "steel_galv", M)
            for dx in (-0.5 * bo["spacing"], 0.5 * bo["spacing"]):
                v, f = revolve([(0.0, -0.05), (bo["radius"], -0.05), (bo["radius"], bo["height"] - 0.04),
                                (bo["radius"] + 0.025, bo["height"] - 0.02), (bo["radius"] + 0.025, bo["height"]),
                                (0.0, bo["height"])], 16)
                part.add(v, f, "steel_galv", T(x + dx, y, z) @ Z_TO_Y())
            posts[f"bollard_{key}_{side}"] = (x, y + bo["height"], z)
    FH = G["fish_hatch"]
    sx, sy, sz = FH["size"]
    x = FH["x"]
    y = L.deck(x, 0.0)
    v, f, M = box_between((x - 0.5 * sx, y - 0.1, -0.5 * sz), (x + 0.5 * sx, y + sy, 0.5 * sz))
    part.add(v, f, "bulwark_inner", M)
    v, f, M = box_between((x - 0.5 * sx - 0.05, y + sy, -0.5 * sz - 0.05), (x + 0.5 * sx + 0.05, y + sy + FH["lid"], 0.5 * sz + 0.05))
    part.add(v, f, "deck_grey", M)
    AW = G["anchor_winch"]
    x = AW["x"]
    y = L.deck(x, 0.0)
    sx, sy, sz = AW["size"]
    v, f, M = box_between((x - 0.5 * sx, y - 0.05, -0.5 * sz), (x + 0.5 * sx, y + sy, 0.5 * sz))
    part.add(v, f, "hydraulic_blue", M)
    v, f = cylinder(AW["drum_radius"], -0.5 * sz - 0.18, 0.5 * sz + 0.18, 16)
    part.add(v, f, "steel_galv", T(x, y + sy * 0.6, 0.0))
    # net stacker aft: two posts and a roller across the deck
    NS = G["net_stacker"]
    x = NS["x"]
    for s in (-1, 1):
        z = s * NS["half_span"]
        y = L.deck(x, z)
        strut(part, (x, y - 0.05, z), (x, y + NS["post_height"], z), 0.06, "hydraulic_blue", n=12)
    yt = L.deck(x, 0.0) + NS["post_height"]
    v, f = cylinder(NS["roller_radius"], -NS["half_span"], NS["half_span"], 16)
    part.add(v, f, "hauler_rubber", T(x, yt, 0.0))
    v, f, M = box_between((x - 0.15, yt - 0.15, NS["half_span"] - 0.05), (x + 0.15, yt + 0.15, NS["half_span"] + 0.2))
    part.add(v, f, "hydraulic_blue", M)
    # derrick post (the boom is its own node)
    D = G["derrick"]
    px, py, pz = D["pivot"]
    strut(part, (px, L.deck(px, pz) - 0.05, pz), (px, py + 0.35, pz), D["post_radius"], "steel_galv", n=14)
    # stern light post at the transom centre
    x0 = L.x0 + 0.1
    ybt = L.ys(L.x0) + B["transom_height"] + B["cap"]["height"]
    LI = spec["lights"]["stern_light"]
    strut(part, (x0, ybt - 0.02, 0.0), (x0, LI["y"] - 0.06, 0.0), 0.02, "stainless", n=8)
    v, f, M = box_between((x0 - 0.1, LI["y"] - 0.06, -0.05), (x0 - 0.02, LI["y"] + 0.06, 0.05))
    part.add(v, f, "light_white", M)
    # sidelight boxes on the house sides
    for s, mat in ((-1, "light_red"), (1, "light_green")):
        nl = spec["lights"]["nav_light_port" if s < 0 else "nav_light_stbd"]
        x = nl["x"]
        y = house.y_top + nl["y_above_roof"]
        u = (house.xf - x) / (house.xf - house.xa)
        z = house.point(0 if s < 0 else 2, u if s < 0 else 1.0 - u, y)[2]
        v, f, M = box_between((x - 0.12, y - 0.07, z - s * 0.01), (x + 0.12, y + 0.07, z + s * 0.1))
        part.add(v, f, "black_plastic", M)
        v, f, M = box_between((x + 0.02, y - 0.05, z + s * 0.1), (x + 0.1, y + 0.05, z + s * 0.104))
        part.add(v, f, mat, M)
    return posts


def build_hauler(spec, L, gear, part, roller):
    G = spec["gear"]
    B = spec["bulwark"]
    hb = table(B["height"])
    key = "net_hauler" if gear == "net" else "line_hauler"
    H = G[key]
    x = H["x"]
    rail_y = L.ys(x) + hb(x) + B["cap"]["height"]
    c = np.array([x, rail_y + H["y_above_rail"], H["z"]])
    R_, w = 0.5 * H["disc_diameter"], H["disc_width"]
    prof = [(0.0, -0.5 * w), (R_, -0.5 * w), (R_, -0.5 * w + 0.03), (0.55 * R_, 0.0),
            (R_, 0.5 * w - 0.03), (R_, 0.5 * w), (0.0, 0.5 * w)]
    v, f = revolve(prof, 32)
    Mdisc = T(*c) @ (Z_TO_X() if gear == "net" else frame_from_forward_up((0.0, -0.5, 0.87), (1.0, 0.0, 0.0)))
    part.add(v, f, "hydraulic_blue", Mdisc)
    v, f = cylinder(0.56 * R_, -0.5 * w + 0.02, 0.5 * w - 0.02, 32)
    part.add(v, f, "hauler_rubber", Mdisc)
    v, f = cylinder(0.12, 0.5 * w, 0.5 * w + 0.22, 16)
    part.add(v, f, "hydraulic_blue", Mdisc)
    # post from the deck and an arm out to the hub
    zp = H["post_z"]
    y0 = L.deck(x, zp)
    top = np.array([x - 0.25, c[1] - 0.1, zp])
    strut(part, (x - 0.25, y0 - 0.05, zp), top, 0.09, "hydraulic_blue", n=14)
    strut(part, top, c + (-0.12, 0.0, 0.0) if gear == "net" else c, 0.07, "hydraulic_blue", n=12)
    if gear == "net":
        RL = G["hauling_roller"]
        zt = L.zs(x) + B["flare_per_m"] * hb(x) - 0.5 * B["thickness"]
        v, f = cylinder(RL["radius"], RL["x0"], RL["x1"], 16)
        roller.add(v, f, "hauler_rubber", T(0.0, rail_y + RL["radius"], zt) @ Z_TO_X() @ T(0.0, 0.0, 0.0))
    return c


def build_derrick(spec):
    D = spec["gear"]["derrick"]
    part = Part()
    el, hd = math.radians(D["boom_elev_deg"]), math.radians(D["boom_heading_deg"])
    d = np.array([math.cos(el) * math.cos(hd), math.sin(el), -math.cos(el) * math.sin(hd)])
    tip = d * D["boom_length"]
    v, f = revolve([(0.0, -0.1), (0.1, -0.1), (0.1, 0.1), (0.0, 0.1)], 14)
    part.add(v, f, "steel_galv", Z_TO_Y())
    strut(part, (0.0, 0.0, 0.0), tip, D["boom_radius"], "hydraulic_blue", n=14)
    strut(part, tip, tip + (0.0, -0.9, 0.0), 0.008, "steel_galv", n=6)
    v, f, M = box_between(tip + (-0.06, -1.1, -0.04), tip + (0.06, -0.9, 0.04))
    part.add(v, f, "hydraulic_blue", M)
    # luffing cylinder from the post up to mid boom
    strut(part, (0.0, -0.7, 0.0), 0.45 * tip, 0.04, "stainless", n=10)
    return part


def build_rudder(spec):
    Rd = spec["propulsion"]["rudder"]
    part = Part()
    c, lf, t = Rd["chord"], Rd["leading_frac"], Rd["half_thickness"]
    le, te = lf * c, -(1.0 - lf) * c
    sec = []
    for k in range(12):
        a = 2 * math.pi * k / 12
        u = 0.5 * (1 + math.cos(a))                   # 1 at LE, 0 at TE
        xx = te + (le - te) * u
        th = t * 2.6 * math.sqrt(max(u, 0.0)) * (1 - u) + 0.004
        sec.append((xx, th * math.sin(a)))
    y0, y1 = Rd["y0"], Rd["y1"]
    rings = [[(x, y, z) for (x, z) in sec] for y in (y0, y1)]
    part.add(*loft(rings), "antifouling")
    v, f = cylinder(Rd["stock_radius"], y1 - 0.02, y1 + 0.35, 14)
    part.add(v, f, "steel_galv", Z_TO_Y())
    return part


def build_propeller(spec):
    P = spec["propulsion"]["propeller"]
    part = Part()
    rh, hl, R_ = P["hub_radius"], P["hub_length"], 0.5 * P["diameter"]
    v, f = revolve([(0.0, -0.5 * hl - 0.1), (0.6 * rh, -0.5 * hl - 0.05), (rh, -0.5 * hl), (rh, 0.5 * hl), (0.0, 0.5 * hl + 0.02)], 20)
    part.add(v, f, "bronze", Z_TO_X())
    pitch = math.radians(P["pitch_deg"])
    cmax = 0.42 * R_
    for b in range(P["blades"]):
        th0 = 2 * math.pi * b / P["blades"]
        rings = []
        for r in np.linspace(0.8 * rh, R_, 9):
            s = (r - 0.8 * rh) / (R_ - 0.8 * rh)
            c = cmax * (0.55 + 0.45 * math.sin(math.pi * min(1.0, s * 0.95 + 0.05))) * (1.0 - 0.75 * s ** 4)
            tk = 0.02 * (1.0 - 0.8 * s) + 0.003
            skew = 0.15 * cmax * s * s
            phi = math.atan2(math.tan(pitch) * R_ * 0.7, r)
            er = np.array([0.0, math.cos(th0), math.sin(th0)])
            et = np.array([0.0, -math.sin(th0), math.cos(th0)])
            ex = np.array([1.0, 0.0, 0.0])
            cd = math.cos(phi) * et + math.sin(phi) * ex
            td = -math.sin(phi) * et + math.cos(phi) * ex
            ring = []
            for k in range(8):
                a = 2 * math.pi * k / 8
                ring.append(r * er + (0.5 * c * math.cos(a) + skew) * cd + tk * math.sin(a) * td)
            rings.append(ring)
        part.add(*loft(rings), "bronze")
    return part


# ---------------------------------------------------------------- registration text (Blender font -> mesh)
def reg_text_part(spec, L, text):
    import bpy
    RG = spec["registration"]
    cu = bpy.data.curves.new("reg_tmp", "FONT")
    cu.body = text
    cu.size = RG["cap_height"] / 0.72
    cu.extrude = RG["extrude"]
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.resolution_u = 4
    cu.offset = RG["stroke_offset"]                     # heavier, block-letter strokes
    ob = bpy.data.objects.new("reg_tmp", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    uvw = np.array([v.co[:] for v in me.vertices])
    faces = [tuple(p.vertices) for p in me.polygons]
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    part = Part()
    xc, yc = RG["x_centre"], RG["y_centre"]
    for s in (-1, 1):
        V = []
        for (u, v, w) in uvw:
            x = xc + s * u
            y = yc + v
            V.append((x, y, s * (L.shell_z(x, y) + RG["offset"] + w)))
        part.add(V, faces, "reg_mark", closed=False)       # the map keeps Blender's outward winding
    return part


def reg_frame(spec, L, s):
    RG = spec["registration"]
    x, y = RG["x_centre"], RG["y_centre"]
    e = 0.01
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


# ---------------------------------------------------------------- livery
def livery(spec, seed):
    """(livery dict, registration text, gear) for a seed; seed None = the reference look."""
    LV = spec["livery"]
    if seed is None:
        return {"hull": "white", "house": "white", "bulwark": "house", "antifouling": "red", "boot_top": "black"}, \
            spec["registration"]["text"], "net"
    rng = random.Random(seed)
    hull = rng.choice(["white", "white", "red", "blue", "green", "cream", "yellow", "orange", "white", "red"])
    house = "white" if rng.random() < 0.7 else rng.choice([k for k in LV["house"] if k != "white"])
    if hull == "white" and house == "white" and rng.random() < 0.3:
        house = "blue"
    af = rng.choice(["red", "red", "black", "blue"])
    bt = rng.choice([k for k in LV["boot_top"] if k != hull])
    reg = f"{rng.choice(LV['county'])}-{rng.randint(1, 450)}-{rng.choice(LV['district'])}"
    gear = "net" if rng.random() < 0.7 else "line"
    bulwark = "hull" if (hull != "white" and rng.random() < 0.4) else "house"
    return {"hull": hull, "house": house, "bulwark": bulwark, "antifouling": af, "boot_top": bt}, reg, gear


def apply_livery(spec, lv):
    sp = copy.deepcopy(spec)
    LV, M = sp["livery"], sp["materials"]
    M["hull_topside"]["color"] = LV["hull"][lv["hull"]]
    M["bulwark_paint"]["color"] = LV["hull"][lv["hull"]] if lv["bulwark"] == "hull" else LV["house"][lv["house"]]
    M["house_paint"]["color"] = LV["house"][lv["house"]]
    M["antifouling"]["color"] = LV["antifouling"][lv["antifouling"]]
    M["boot_top"]["color"] = LV["boot_top"][lv["boot_top"]]
    hc = LV["hull"][lv["hull"]]
    if 0.2126 * hc[0] + 0.7152 * hc[1] + 0.0722 * hc[2] < LV["reg_light_below_luminance"]:
        M["reg_mark"]["color"] = LV["reg_light"]            # painted light on a dark hull
    return sp


# ---------------------------------------------------------------- assembly
spec_cache = {}


def mass_check(spec):
    M = spec["mass"]
    tot = sum(b["mass"] for b in M["budget"])
    assert abs(tot - M["dry"]) < 1e-6, ("budget does not sum to dry", tot)
    m, com = mass_condition(spec, "lightship")
    assert all(abs(com[i] - M["com_dry"][i]) < 0.005 for i in range(3)), ("com_dry stale", [round(c, 4) for c in com])
    return com


def build_geometry(spec, gear):
    L = Lines(spec)
    house = House(spec, L)
    hull, bulwark, wh, glass, mast, fit, hauler = Part(), Part(), Part(), Part(), Part(), Part(), Part()
    build_hull(spec, L, hull)
    build_keel(spec, L, hull)
    build_strakes(spec, L, hull)
    build_bulwark(spec, L, bulwark)
    build_wheelhouse(spec, L, house, wh, glass)
    mast_pts = build_mast(spec, house, mast)
    posts = build_fittings(spec, L, house, fit)
    hub = build_hauler(spec, L, gear, hauler, fit)
    tris = np.concatenate(hull.tris)
    parts = {"hull": hull, "bulwark": bulwark, "wheelhouse": wh, "windows": glass, "mast": mast,
             "deck_fittings": fit, "net_hauler": hauler}
    return parts, tris, {"L": L, "house": house, "mast": mast_pts, "posts": posts, "hub": hub}


def check_gates(spec, hydro):
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


def build_blender(spec, parts, hydro, info, lv, reg, gear, seed):
    import bpy
    import bmesh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    L, house = info["L"], info["house"]
    parts["hull"] = copy.deepcopy(parts["hull"])
    rp = reg_text_part(spec, L, reg)
    base = len(parts["hull"].verts)
    parts["hull"].verts.extend(rp.verts)
    parts["hull"].faces.extend(tuple(base + k for k in f) for f in rp.faces)
    parts["hull"].mats.extend(rp.mats)
    d = hydro["conditions"][hydro["design_condition"]]
    m_dry, com = mass_condition(spec, "lightship")
    root = new_object("sjark", None, mats, None, I, props={
        "frame": "X forward, Y up, Z starboard; origin = centreline x baseline x mid-LOA",
        "spec": "sjark_spec.json", "hydro": "sjark_hydro.json",
        "mass_dry": m_dry, "com_dry": [round(c, 4) for c in com],
        "design_waterline_y": d["waterline_y_at_x0"], "design_trim_deg": d["trim_deg"],
        "seed": -1 if seed is None else seed, "livery": json.dumps(lv), "registration": reg, "gear": gear})
    objs = {}
    for name, sharp in (("hull", 25.0), ("bulwark", 35.0), ("wheelhouse", 30.0), ("windows", 30.0),
                        ("mast", 30.0), ("deck_fittings", 30.0), ("net_hauler", 30.0)):
        objs[name] = new_object(name, parts[name], mats, root, I, sharp_deg=sharp)
    # weld the hull loft (collapsed paint-level points) and drop the slivers
    me = objs["hull"].data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bm.to_mesh(me)
    bm.free()
    me.set_sharp_from_angle(angle=math.radians(25.0))
    me.update()

    MP = info["mast"]
    new_object("radar", build_radar(), mats, root, T(*MP["radar"]),
               props={"axis": "+Y", "rpm": spec["mast"]["radar"]["rpm"], "range_max": spec["mast"]["radar"]["range_max"]})
    D = spec["gear"]["derrick"]
    new_object("derrick", build_derrick(spec), mats, root, T(*D["pivot"]),
               props={"axis": "+Y", "limit_deg": D["limit_deg"], "rest": "boom pointing aft over the working deck"})
    Rd = spec["propulsion"]["rudder"]
    new_object("rudder", build_rudder(spec), mats, root, T(Rd["stock_x"], 0.0, 0.0),
               props={"axis": "+Y", "limit_deg": Rd["limit_deg"],
                      "positive": "trailing edge swings to port, the boat turns to port"})
    P = spec["propulsion"]["propeller"]
    new_object("propeller", build_propeller(spec), mats, root, T(P["x"], P["axis_y"], 0.0),
               props={"axis": "+X", "blades": P["blades"], "diameter": P["diameter"],
                      "positive": "right-handed about +X (seen from aft: clockwise = ahead)"})
    new_object("prop_thrust", None, mats, root, T(P["x"], P["axis_y"], 0.0),
               props={"thrust_axis": "+X", "bollard_thrust_n": P["bollard_thrust_n"], "steered_by": "rudder",
                      "engine_power_kw": spec["propulsion"]["engine_power_kw"]})
    bt = spec["propulsion"]["bow_thruster"]
    new_object("bow_thruster", None, mats, root, T(*bt["position"]),
               props={"thrust_axis": bt["thrust_axis"], "thrust_n": bt["thrust_n"], "tunnel_radius": bt["tunnel_radius"]})
    for name, p in info["posts"].items():
        new_object(name, None, mats, root, T(*p), props={"role": "mooring bollard top"})
    new_object("tow_point", None, mats, root, T(*spec["gear"]["tow_point"]), props={"role": "towing eye"})
    cm = spec["mast"]["camera_mast"]
    cp = np.asarray(cm["position_above_roof"]) + (spec["mast"]["x"], house.roof_y, 0.0)
    cp[0] = spec["mast"]["x"] + cm["position_above_roof"][0]
    new_object("camera_mast", None, mats, root, sensor_frame(cp.tolist(), cm["look"]),
               props={"convention": "look -Z, up +Y"})
    LI = spec["lights"]
    for s, name in ((-1, "nav_light_port"), (1, "nav_light_stbd")):
        nl = LI[name]
        y = house.y_top + nl["y_above_roof"]
        u = (house.xf - nl["x"]) / (house.xf - house.xa)
        z = house.point(0 if s < 0 else 2, u if s < 0 else 1.0 - u, y)[2] + s * 0.105
        new_object(name, None, mats, root, sensor_frame((nl["x"] + 0.06, y, z), (1.0, 0.0, 0.0)),
                   props={"convention": "look -Z, up +Y", "colour": nl["colour"], "sector_deg": nl["sector_deg"]})
    ml = LI["masthead_light"]
    new_object("masthead_light", None, mats, root, sensor_frame(MP["masthead"], (1.0, 0.0, 0.0)),
               props={"convention": "look -Z, up +Y", "colour": ml["colour"], "sector_deg": ml["sector_deg"]})
    sl = LI["stern_light"]
    new_object("stern_light", None, mats, root, sensor_frame((sl["x"] - 0.1, sl["y"], 0.0), (-1.0, 0.0, 0.0)),
               props={"convention": "look -Z, up +Y", "colour": sl["colour"], "sector_deg": sl["sector_deg"]})
    for s, name in ((-1, "reg_mark_port"), (1, "reg_mark_stbd")):
        new_object(name, None, mats, root, reg_frame(spec, L, s),
                   props={"text": reg, "convention": "local +Z = outward hull normal, +X = reading direction"})
    return root


def export(path_out):
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
                              export_texcoords=True, export_normals=True)
    os.replace(tmp, path_out)
    return tris_n


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="sjark_spec.json")
    ap.add_argument("--out", default="sjark.glb")
    ap.add_argument("--hydro-out", default="sjark_hydro.json")
    ap.add_argument("--hydro-only", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="one seeded livery -> out/sjark_s<N>.glb")
    ap.add_argument("--seeds", default="", help="comma list of seeds, one Blender run")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    spec_cache.update(spec)
    com = mass_check(spec)
    print(f"[sjark] spec ok; dry CoM from budget = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
    parts, tris, info = build_geometry(spec, "net")
    extra = [build_radar(), build_derrick(spec), build_rudder(spec), build_propeller(spec)]
    used = set(m for p in list(parts.values()) + extra for m in p.mats) | {"reg_mark"}
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    hydro = hydrostatics(spec, tris, os.path.basename(a.spec))
    for cond, r in hydro["conditions"].items():
        print(f"[sjark] {cond:10s} {r['mass']:7.0f} kg  disp {r['displacement_m3']:.2f} m3  waterline y "
              f"{r['waterline_y_at_x0']:.3f}  max draft {r['draft_max']:.3f}  trim {r['trim_deg']:+.2f} deg  "
              f"LCB {r['lcb']:+.3f}  KB {r['kb']:.3f}  GM_T {r['gm_t']:.3f}  GM_L {r['gm_l']:.1f}  "
              f"LWL {r['lwl']:.2f}  BWL {r['bwl']:.2f}  Awp {r['waterplane_area']:.2f}")
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] + ([a.seed] if a.seed is not None else [])
    if not seeds:
        with open(path(a.hydro_out), "w", encoding="utf-8") as fh:
            json.dump(hydro, fh, indent=1)
        print(f"[sjark] wrote {path(a.hydro_out)}")
    check_gates(spec, hydro)
    print("[sjark] gates PASS (draft, trim, GM_T, closed primitives, closed buoyant mesh)")
    if a.hydro_only:
        return
    jobs = [(None, path(a.out))] if not seeds else [(s, os.path.join(HERE, "out", f"sjark_s{s}.glb")) for s in seeds]
    for seed, out in jobs:
        lv, reg, gear = livery(spec, seed)
        sp = apply_livery(spec, lv)
        p2, _, info2 = build_geometry(sp, gear)
        build_blender(sp, p2, hydro, info2, lv, reg, gear, seed)
        n = export(out)
        print(f"[sjark] wrote {out}  seed={seed} livery={lv} reg={reg} gear={gear} triangles={n}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

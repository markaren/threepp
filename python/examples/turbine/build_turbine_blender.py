"""Build the IEA 15 MW offshore turbine on its monopile from turbine_spec.json: turbine.glb.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_turbine_blender.py -- --spec turbine_spec.json --out turbine.glb

Geometry only (plain Python + numpy, no Blender; runs the checks):
    python build_turbine_blender.py --check

The mesh primitives and the Blender export are the Mariner generator's
(../usv/build_mariner_blender.py). Frame of the spec: X upwind (the rotor is on
+X of the tower), Y up, Z = X x Y, metres, origin on the pile axis at MSL.

Node hierarchy (all names exact):
    turbine                      root, identity
      monopile                   pile below the TP, cable entry collar
        marine_growth            fouling sleeve (species colours: turbine_site.py)
        anode_cage               ring frames, clamps, bar anodes
      transition_piece           TP, platform, railing, boat landing, ladder, davit, door, ID
      tower                      tapered tube with section flanges
      nacelle                    origin on the yaw axis at the tower top
        rotor                    origin at the hub centre, turns about local +X (the shaft)
          pitch_1..3             pitch axis = local +Y (the span); baked feathered
            blade_1..3
        helihoist                empty, centre of the roof platform
      cable_entry, boat_landing  empties (look -Z, up +Y)

Inspection pose: rotor locked with blade_1 at 6 o'clock in front of the tower, feathered.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "usv"))
from build_mariner_blender import (Part, R, T, Z_TO_X, Z_TO_Y, box, box_between, cylinder,  # noqa: E402
                                   frame_from_forward_up, frustum, loft, make_materials, new_object,
                                   revolve, strut)
from build_otterx_blender import annulus  # noqa: E402


def val(d):
    return d["value"] if isinstance(d, dict) and "value" in d else d


def pol(r, az, y):
    """Point at radius r, azimuth az (rad, atan2(z, x)), height y."""
    return np.array([r * math.cos(az), y, r * math.sin(az)])


def radial(az):
    return np.array([math.cos(az), 0.0, math.sin(az)])


def tangent(az):
    return np.array([-math.sin(az), 0.0, math.cos(az)])


# ---------------------------------------------------------------- local primitives
def arc_box(r0, r1, a0, a1, y0, y1, n=8):
    """Curved panel about +Y: radii r0..r1, azimuths a0..a1 (rad), heights y0..y1."""
    rings = []
    for k in range(n + 1):
        a = a0 + (a1 - a0) * k / n
        rings.append([tuple(pol(r, a, y)) for (r, y) in ((r0, y0), (r1, y0), (r1, y1), (r0, y1))])
    return loft(rings)


def torus(Rr, rs, y, n=96, m=8):
    """Horizontal ring tube about +Y at height y."""
    v, f = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        for j in range(m):
            b = 2 * math.pi * j / m
            v.append(tuple(pol(Rr + rs * math.cos(b), a, y + rs * math.sin(b))))
    for i in range(n):
        i1 = (i + 1) % n
        for j in range(m):
            j1 = (j + 1) % m
            f.append((i * m + j, i1 * m + j, i1 * m + j1, i * m + j1))
    return v, f


def ring_y(part, r0, r1, y0, y1, mat, n=96):
    """Annulus about +Y (a flange, a deck)."""
    v, f = annulus(r0, r1, y0, y1, n)
    part.add(v, f, mat, Z_TO_Y())


def vcyl(part, x, z, r, y0, y1, mat, n=16):
    v, f = cylinder(r, y0, y1, n)
    part.add(v, f, mat, T(x, 0.0, z) @ Z_TO_Y())


def surface_stroke(part, Rc, az_c, yc, p0, p1, w, d0, d1, mat):
    """A painted stroke from p0 to p1 (u right, v up, metres) on a cylinder of radius Rc,
    as a thin prism from Rc + d0 to Rc + d1 along the surface normal."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    d /= L
    nrm = np.array([-d[1], d[0]])
    a = p0 - d * 0.5 * w
    b = p1 + d * 0.5 * w
    rings = []
    for s in np.linspace(0.0, 1.0, max(2, int(math.ceil((L + w) / 0.12)) + 1)):
        c = a + (b - a) * s
        ring = []
        for (k, rr) in ((-1, Rc + d0), (1, Rc + d0), (1, Rc + d1), (-1, Rc + d1)):
            u, vv = c + nrm * 0.5 * w * k
            ring.append(tuple(pol(rr, az_c - u / Rc, yc + vv)))    # viewer's right = decreasing azimuth
        rings.append(ring)
    part.add(*loft(rings), mat)


GLYPHS = {
    "K": [((0, 0), (0, 1)), ((0.05, 0.42), (0.9, 1)), ((0.32, 0.6), (0.95, 0))],
    "0": [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 0), (0, 0))],
    "1": [((0.5, 0), (0.5, 1)), ((0.5, 1), (0.2, 0.8))],
    "7": [((0, 1), (1, 1)), ((1, 1), (0.35, 0))],
    "2": [((0, 1), (1, 1)), ((1, 1), (1, 0.5)), ((1, 0.5), (0, 0.5)), ((0, 0.5), (0, 0)), ((0, 0), (1, 0))],
    "5": [((1, 1), (0, 1)), ((0, 1), (0, 0.5)), ((0, 0.5), (1, 0.5)), ((1, 0.5), (1, 0)), ((1, 0), (0, 0))],
}


def paint_text(part, Rc, az, y, text, h, mat, d0=-0.02, d1=0.035):
    w, pitch, sw = 0.6 * h, 0.85 * h, 0.14 * h
    u0 = -0.5 * (pitch * (len(text) - 1) + w)
    for k, ch in enumerate(text):
        for (a, b) in GLYPHS[ch]:
            pa = (u0 + k * pitch + a[0] * w, (a[1] - 0.5) * h)
            pb = (u0 + k * pitch + b[0] * w, (b[1] - 0.5) * h)
            surface_stroke(part, Rc, az, y, pa, pb, sw, d0, d1, mat)


# ---------------------------------------------------------------- below water
def build_monopile(spec, pile, growth, cage):
    MP, UW = spec["monopile"], spec["underwater"]
    rp = 0.5 * val(MP["outer_diameter"])
    tp_r = 0.5 * val(spec["transition_piece"]["outer_diameter"])
    tp_y0 = val(spec["transition_piece"]["bottom_y"])
    v, f = cylinder(rp, val(MP["modelled_bottom_y"]), val(MP["top_y"]), 96)
    pile.add(v, f, "pile_steel", Z_TO_Y())

    # marine growth: a lumpy sleeve, thickest just under the splash zone
    G = UW["marine_growth"]
    rng = np.random.default_rng(7)
    ph = rng.uniform(0, 2 * np.pi, 8)
    N = 128
    ys = np.arange(-29.6, G["top_y"] + 1e-6, 0.25)

    def thick(y):
        if y > tp_y0:
            return 0.03 + 0.05 * (G["top_y"] - y) / (G["top_y"] - tp_y0)
        if y > G["peak_y"]:
            return 0.08 + (G["max_thickness"] - 0.08) * (tp_y0 - y) / (tp_y0 - G["peak_y"])
        s = (G["peak_y"] - y) / (G["peak_y"] + 29.6)
        return G["max_thickness"] + (G["bottom_thickness"] - G["max_thickness"]) * s ** 0.7

    def lump(a, y):
        return (0.55 + 0.25 * math.sin(7 * a + ph[0] + 0.9 * y) + 0.15 * math.sin(19 * a + ph[1] - 2.1 * y)
                + 0.12 * math.sin(41 * a + ph[2] + 3.7 * y) + 0.08 * math.sin(83 * a + ph[3] - 5.3 * y))

    DM = UW["depth_marks"]
    maz = math.radians(DM["azimuth_deg"])
    ang = 2 * math.pi * np.arange(N) / N
    arc = np.abs((ang - maz + math.pi) % (2 * math.pi) - math.pi) * rp
    keep = 1.0 - np.clip((DM["strip_half_width"] + 0.5 - arc) / 0.5, 0.0, 1.0)   # the cleaned strip
    CLEAN = 0.012                              # the cleaned strip's surface: a film over the coating
    rings, ring_y_ = [], []
    for y in ys:
        base = tp_r if y > tp_y0 + 1e-6 else rp
        g = thick(y)
        ring = [tuple(pol(base + CLEAN + g * keep[i] * (max(lump(ang[i], y), 0.15) + 0.35 * rng.uniform()), ang[i], y))
                for i in range(N)]
        rings.append(ring)
        ring_y_.append(y)
        if abs(y - tp_y0) < 0.13:              # step out onto the TP skirt
            ring = [tuple(pol(tp_r + 0.03, 2 * math.pi * i / N, y + 0.001)) for i in range(N)]
            rings.append(ring)
            ring_y_.append(y + 0.001)
    v, f = loft(rings)
    growth.add(v, f, "growth_base")          # species colours are vertex colours, set by the site
    # painted depth marks in the cleaned strip: paint, 3 mm proud of it and curved with the pile;
    # a tick every metre, a longer tick and a numeral every 5 m, duller with depth
    rm = rp + CLEAN
    for d in range(DM["from_m"], DM["to_m"] + 1):
        y = -float(d)
        big = d % 5 == 0
        mat = "mark_paint" if d < 15 else "mark_paint_deep"
        surface_stroke(pile, rm, maz, y, (-0.55 if big else -0.3, 0.0), (0.0, 0.0), 0.06, -0.003, 0.003, mat)
        if big:
            paint_text(pile, rm, maz - 0.5 / rm, y, str(d), 0.35, mat, -0.003, 0.003)

    # anode cage: two ring frames, clamps to the pile, bar anodes on stand-offs
    C = UW["anode_cage"]
    rr = C["ring_radius"]
    for y in C["ring_y"]:
        cage.add(*torus(rr, 0.11, y, 128, 10), "galv_steel")
        for k in range(8):
            a = 2 * math.pi * (k + 0.5) / 8
            strut(cage, pol(rp - 0.05, a, y), pol(rr, a, y), 0.09, "galv_steel", 10)
    sx, sy, sz = C["anode_size"]
    yc = 0.5 * sum(C["ring_y"])
    for k in range(C["anodes"]):
        a = 2 * math.pi * k / C["anodes"]
        ra = rr + 0.32
        v, f = box(sx, sy, sz)
        cage.add(v, f, "anode_alu", T(*pol(ra, a, yc)) @ R("Y", -math.degrees(a)))
        for y in C["ring_y"]:
            strut(cage, pol(rr, a, y), pol(ra, a, y if abs(y - yc) < 0.5 * sy else yc + math.copysign(0.5 * sy - 0.2, y - yc)),
                  0.05, "galv_steel", 8)
    # cable entry: a collar round the hole through the pile wall
    ce = spec["empties"]["cable_entry"]
    a = math.radians(ce["azimuth_deg"])
    rh = 0.5 * ce["hole_diameter"]
    M = T(*pol(0.0, a, ce["y"])) @ frame_from_forward_up(-radial(a), (0.0, 1.0, 0.0))   # local +Z = outward
    v, f = annulus(rh, rh + 0.16, rp - 0.1, rp + 0.55, 48)
    pile.add(v, f, "galv_steel", M)
    v, f = cylinder(rh + 0.01, rp - 0.2, rp + 0.25, 48)
    pile.add(v, f, "hole_dark", M)


# ---------------------------------------------------------------- transition piece
def build_tp(spec, part):
    TPs = spec["transition_piece"]
    R0 = 0.5 * val(TPs["outer_diameter"])
    y0, y1 = val(TPs["bottom_y"]), val(TPs["top_y"])
    py = val(TPs["platform_y"])
    Ro = val(TPs["platform_outer_radius"])
    v, f = revolve([(R0 - 0.35, y0), (R0, y0), (R0, y1), (R0 - 0.35, y1)], 128)
    part.add(v, f, "tp_yellow", Z_TO_Y())
    ring_y(part, R0 - 0.05, R0 + 0.12, y1 - 0.35, y1, "flange_grey", 128)
    # platform deck, its knee braces, railing
    ring_y(part, R0 - 0.1, Ro, py - 0.25, py, "grating", 128)
    ring_y(part, Ro - 0.02, Ro + 0.12, py - 0.35, py + 0.12, "rail_yellow", 128)       # toe board
    for k in range(12):
        a = 2 * math.pi * (k + 0.5) / 12
        strut(part, pol(R0 - 0.05, a, py - 3.2), pol(Ro - 0.4, a, py - 0.2), 0.14, "tp_yellow", 10)
    rh = val(TPs["railing_height"])
    n_posts = 40
    for k in range(n_posts):
        a = 2 * math.pi * k / n_posts
        strut(part, pol(Ro - 0.06, a, py - 0.02), pol(Ro - 0.06, a, py + rh), 0.03, "rail_yellow", 8)
    part.add(*torus(Ro - 0.06, 0.035, py + rh, 160, 8), "rail_yellow")
    part.add(*torus(Ro - 0.06, 0.025, py + 0.55 * rh, 160, 8), "rail_yellow")

    # boat landing: two fender tubes on stubs, the ladder between them
    az = math.radians(val(TPs["boat_landing_azimuth_deg"]))
    BL, LD = TPs["boat_landing"], TPs["ladder"]
    n, t = radial(az), tangent(az)
    rf = R0 + BL["standoff"]
    for s in (-1.0, 1.0):
        c = n * rf + t * s * 0.5 * BL["fender_spacing"]
        ya, yb = BL["y_range"]
        v, f = revolve([(0.0, ya - 0.3), (BL["fender_radius"], ya), (BL["fender_radius"], yb),
                        (0.0, yb + 0.3)], 24)
        part.add(v, f, "rail_yellow", T(c[0], 0.0, c[2]) @ Z_TO_Y())
        for yb_ in np.linspace(ya, yb, 9)[1:-1:2]:
            v, f = cylinder(BL["fender_radius"] + 0.04, yb_ - 0.25, yb_ + 0.25, 24)   # rubber fender bands
            part.add(v, f, "fender_black", T(c[0], 0.0, c[2]) @ Z_TO_Y())
        for ys in (ya + 0.8, 0.5 * (ya + yb) + 1.0, yb - 0.6):
            p0 = n * (R0 - 0.1) + t * s * 0.5 * BL["fender_spacing"] + np.array([0.0, ys, 0.0])
            p1 = c + np.array([0.0, ys, 0.0])
            strut(part, p0, p1, 0.16, "tp_yellow", 12)
            strut(part, p0 + np.array([0.0, -1.1, 0.0]), p1 + np.array([0.0, 0.0, 0.0]), 0.09, "tp_yellow", 10)
    rl = R0 + LD["standoff"]
    ya, yb = LD["y_range"]
    for s in (-1.0, 1.0):
        c = n * rl + t * s * 0.5 * LD["width"]
        strut(part, c + [0.0, ya, 0.0], c + [0.0, yb + 1.0, 0.0], 0.04, "galv_steel", 8)
        for yk in np.arange(ya + 1.0, yb, 3.0):
            strut(part, n * (R0 - 0.05) + t * s * 0.5 * LD["width"] + [0.0, yk, 0.0], c + [0.0, yk, 0.0], 0.03,
                  "galv_steel", 8)
    for yk in np.arange(ya + 0.2, yb, LD["rung_pitch"]):
        strut(part, n * rl - t * 0.5 * LD["width"] + [0.0, yk, 0.0], n * rl + t * 0.5 * LD["width"] + [0.0, yk, 0.0],
              0.016, "galv_steel", 6)
    # safety cage hoops above the landing
    for yk in np.arange(BL["y_range"][1] + 0.5, yb, 0.9):
        pts = [n * (rl + 0.35 + 0.35 * math.sin(q)) + t * 0.45 * math.cos(q) + [0.0, yk, 0.0]
               for q in np.linspace(0.0, math.pi, 7)]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            strut(part, p0, p1, 0.02, "galv_steel", 6)

    # door into the TP, above the platform
    DR = TPs["door"]
    ad = math.radians(DR["azimuth_deg"])
    hw = 0.5 * DR["width"] / R0
    part.add(*arc_box(R0 - 0.03, R0 + 0.05, ad - hw - 0.02, ad + hw + 0.02, py + 0.05, py + DR["height"] + 0.12, 6),
             "paint_black")
    part.add(*arc_box(R0 - 0.03, R0 + 0.09, ad - hw, ad + hw, py + 0.1, py + DR["height"], 6), "door_grey")
    # ID number, twice, black on yellow
    ID = TPs["id_text"]
    for a_deg in ID["azimuths_deg"]:
        paint_text(part, R0, math.radians(a_deg), ID["y"], ID["value"], ID["height"], "paint_black")
    # davit crane on the platform
    DV = TPs["davit"]
    ad = math.radians(DV["azimuth_deg"])
    base = pol(Ro - 1.0, ad, py)
    v, f = cylinder(0.45, py, py + 0.3, 20)
    part.add(v, f, "rail_yellow", T(base[0], 0.0, base[2]) @ Z_TO_Y())
    v, f = cylinder(0.2, py + 0.3, py + DV["height"], 20)
    part.add(v, f, "rail_yellow", T(base[0], 0.0, base[2]) @ Z_TO_Y())
    top = base + [0.0, DV["height"] - 0.2, 0.0]
    tip = top + radial(ad) * DV["jib"]
    strut(part, top, tip, 0.14, "rail_yellow", 12)
    strut(part, base + [0.0, DV["height"] - 1.6, 0.0], top + radial(ad) * 1.6, 0.08, "rail_yellow", 10)
    strut(part, tip, tip - [0.0, 1.8, 0.0], 0.012, "galv_steel", 6)
    v, f, M = box_between(tip - [0.15, 2.1, 0.15], tip - [-0.15, 1.8, -0.15])
    part.add(v, f, "paint_black", M)


# ---------------------------------------------------------------- tower
def build_tower(spec, part):
    TW = spec["tower"]
    prof = val(TW["outer_diameter_profile"])
    ys = [p[0] for p in prof]
    rs = [0.5 * p[1] for p in prof]
    v, f = revolve([(r, y) for (y, r) in zip(ys, rs)], 96)
    part.add(v, f, "tower_grey", Z_TO_Y())
    for y in val(TW["flange_y"]):
        r = float(np.interp(y, ys, rs))
        ring_y(part, r - 0.1, r + 0.05, y - 0.12, y + 0.12, "flange_grey", 96)
    ring_y(part, rs[0] - 0.1, rs[0] + 0.08, ys[0], ys[0] + 0.35, "flange_grey", 96)


# ---------------------------------------------------------------- nacelle (built in its node frame: yaw axis at the tower top)
def rounded_rect(x, y0, y1, hw, rc, n=4):
    """Section in the YZ plane at x, rounded corners; a closed loop."""
    pts = []
    cs = [(hw - rc, y1 - rc, 0.0), (-hw + rc, y1 - rc, 90.0), (-hw + rc, y0 + rc, 180.0), (hw - rc, y0 + rc, 270.0)]
    for (cz, cy, a0) in cs:
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((x, cy + rc * math.sin(a), cz + rc * math.cos(a)))
    return pts


def build_nacelle(spec, part):
    NA, RO = spec["nacelle"], spec["rotor"]
    ytop = val(spec["tower"]["top_y"])
    hub_y = val(RO["hub_height"]) - ytop
    ov = val(RO["hub_overhang"])
    tilt = math.radians(val(RO["shaft_tilt_deg"]))
    x0, x1 = val(NA["x_range"])
    hw = 0.5 * val(NA["width"])
    yb, yt = 0.4, 0.4 + val(NA["height"])
    # yaw bearing housing
    v, f = cylinder(3.35, -0.05, yb + 0.3, 48)
    part.add(v, f, "flange_grey", Z_TO_Y())
    rings = [rounded_rect(x0, yb + 0.6, yt - 0.3, hw - 0.4, 0.8), rounded_rect(x0 + 0.8, yb, yt, hw, 1.0),
             rounded_rect(x1 - 0.6, yb, yt, hw, 1.0), rounded_rect(x1, yb + 0.3, yt - 0.2, hw - 0.3, 0.9)]
    part.add(*loft(rings), "nacelle_white")
    # direct-drive generator and the front bearing housing on the shaft axis
    ax = np.array([math.cos(tilt), math.sin(tilt), 0.0])
    hub = np.array([ov, hub_y, 0.0])
    Mshaft = T(*hub) @ R("Z", math.degrees(tilt)) @ Z_TO_X()            # local +Z = shaft, toward the hub
    rg = 0.5 * val(NA["generator_diameter"])
    s1 = -(ov - x1) / math.cos(tilt) + 0.2
    s0 = s1 - val(NA["generator_length"])
    v, f = revolve([(rg - 0.6, s0 - 0.4), (rg - 0.15, s0), (rg, s0 + 0.3), (rg, s1 - 0.3), (rg - 0.2, s1),
                    (rg - 1.4, s1 + 0.15)], 96)
    part.add(v, f, "nacelle_white", Mshaft)
    for k in range(3):
        sk = s0 + 0.5 + k * 0.8
        v, f = annulus(rg - 0.1, rg + 0.04, sk, sk + 0.12, 96)
        part.add(v, f, "flange_grey", Mshaft)
    v, f = frustum(3.3, 2.9, s1 + 0.1, -3.2, 64)
    part.add(v, f, "flange_grey", Mshaft)
    _ = ax
    # side dressing: panel seams, a hatch, louvred vents, the registration painted on both sides
    for sz in (1.0, -1.0):
        def plate(xa, xb, ya, yb, proud, mat):
            za, zb = sz * (hw - 0.01), sz * (hw + proud)
            v, f, M = box_between((xa, ya, min(za, zb)), (xb, yb, max(za, zb)))
            part.add(v, f, mat, M)
        for xs in (-7.0, -4.5, -2.0, 0.5, 3.0):
            plate(xs - 0.015, xs + 0.015, yb + 1.1, yt - 1.1, 0.008, "flange_grey")
        plate(x0 + 1.2, x1 - 0.9, 0.5 * (yb + yt) - 0.015, 0.5 * (yb + yt) + 0.015, 0.008, "flange_grey")
        plate(-6.3, -4.7, yb + 1.3, yb + 3.5, 0.012, "flange_grey")
        plate(-6.2, -4.8, yb + 1.4, yb + 3.4, 0.022, "door_grey")
        plate(-8.6, -7.1, yb + 4.6, yb + 6.6, 0.01, "paint_black")
        for k in range(7):
            yk = yb + 4.75 + k * 0.27
            plate(-8.5, -7.2, yk, yk + 0.1, 0.03, "flange_grey")
        h_ = 1.1
        w_, pitch = 0.6 * h_, 0.85 * h_
        u0 = -0.5 * (pitch * 2 + w_)
        for k, ch in enumerate("K07"):
            for (a, b) in GLYPHS[ch]:
                pa = np.array([u0 + k * pitch + a[0] * w_, (a[1] - 0.5) * h_])
                pb = np.array([u0 + k * pitch + b[0] * w_, (b[1] - 0.5) * h_])
                d = pb - pa
                L = float(np.linalg.norm(d))
                d = d / L
                nn = np.array([-d[1], d[0]]) * 0.07
                a2, b2 = pa - d * 0.07, pb + d * 0.07
                q = [a2 - nn, b2 - nn, b2 + nn, a2 + nn]
                xc, yc = -1.9, yb + 2.6
                v = []
                for zz in (sz * (hw - 0.01), sz * (hw + 0.006)):
                    for (u, vv) in q:
                        v.append((xc + sz * u, yc + vv, zz))       # upright and left-to-right from outside
                f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
                part.add(v, f, "paint_black")
    # rooftop cooler with fins
    v, f, M = box_between((-2.4, yt - 0.1, -3.0), (1.6, yt + 1.3, 3.0))
    part.add(v, f, "nacelle_white", M)
    for k in range(9):
        z = -2.6 + k * 0.65
        v, f, M = box_between((-2.2, yt + 1.25, z - 0.05), (1.4, yt + 1.9, z + 0.05))
        part.add(v, f, "flange_grey", M)
    # met mast pair with cup anemometers, vane and aviation lights
    for s in (-1.0, 1.0):
        base = np.array([-3.4, yt, s * 3.2])
        strut(part, base, base + [0.0, 4.2, 0.0], 0.07, "galv_steel", 10)
        strut(part, base + [0.0, 4.2, 0.0], base + [0.0, 4.2, -s * 0.9], 0.04, "galv_steel", 8)
        v, f = cylinder(0.18, 0.0, 0.34, 16)
        part.add(v, f, "aviation_red", T(*(base + [0.0, 4.2, 0.0])) @ Z_TO_Y())
        for q in range(3):
            a = 2 * math.pi * q / 3
            c = base + [0.0, 4.2, -s * 0.9] + np.array([0.25 * math.cos(a), 0.2, 0.25 * math.sin(a)])
            strut(part, base + [0.0, 4.2, -s * 0.9], c, 0.02, "galv_steel", 6)
            v, f = revolve([(0.0, -0.07), (0.07, 0.0), (0.0, 0.07)], 10)
            part.add(v, f, "paint_black", T(*c))
    # helihoist platform on the rear roof
    HH = NA["helihoist"]
    ha, hb = HH["x_range"]
    hz = hw - 0.6
    v, f, M = box_between((ha, yt - 0.1, -hz), (hb, yt + 0.25, hz))
    part.add(v, f, "helideck_green", M)
    cx = 0.5 * (ha + hb)
    v, f = annulus(1.9, 2.25, yt + 0.24, yt + 0.28, 64)
    part.add(v, f, "marking_yellow", T(cx, 0.0, 0.0) @ Z_TO_Y())
    for (p, q) in (((ha + 0.2, -hz + 0.2), (hb - 0.2, -hz + 0.2)), ((ha + 0.2, hz - 0.2), (hb - 0.2, hz - 0.2)),
                   ((ha + 0.2, -hz + 0.2), (ha + 0.2, hz - 0.2)), ((hb - 0.2, -hz + 0.2), (hb - 0.2, hz - 0.2))):
        a = np.array([p[0], yt + 0.25 + HH["railing_height"], p[1]])
        b = np.array([q[0], yt + 0.25 + HH["railing_height"], q[1]])
        strut(part, a, b, 0.04, "rail_yellow", 8)
        strut(part, a - [0.0, 0.5, 0.0], b - [0.0, 0.5, 0.0], 0.03, "rail_yellow", 8)
        L = float(np.linalg.norm(b - a))
        for k in range(int(L / 1.4) + 1):
            pp = a + (b - a) * (k / max(int(L / 1.4), 1))
            strut(part, pp - [0.0, HH["railing_height"], 0.0], pp, 0.035, "rail_yellow", 8)
    v, f = cylinder(0.14, 0.0, 0.3, 16)
    part.add(v, f, "aviation_red", T(ha + 0.2, yt + 0.25 + HH["railing_height"], 0.0) @ Z_TO_Y())
    return np.array([cx, yt + 0.25, 0.0])


# ---------------------------------------------------------------- rotor (built in its node frame: hub centre, +X = shaft)
def build_hub(spec, part):
    rh = 0.5 * val(spec["rotor"]["hub_diameter"])
    v, f = revolve([(0.0, -3.6), (3.3, -3.6), (rh, -2.2), (rh, 0.9), (rh - 0.4, 2.3), (2.9, 3.6), (1.6, 4.6),
                    (0.5, 5.1), (0.0, 5.2)], 64)
    part.add(v, f, "nacelle_white", Z_TO_X())


def airfoil_section(chord, twist_deg, t, pa, npts):
    """Closed loop in the blade frame's XZ plane: chord along Z (leading edge at +Z),
    thickness along X; a circle at t = 1 blending into a cambered NACA-type foil."""
    b = min(max((t - 0.4) / 0.6, 0.0), 1.0)
    th = 2 * np.pi * np.arange(npts) / npts
    xc = 0.5 + 0.5 * np.cos(th)
    up = np.sin(th) >= 0.0
    tf = min(t, 0.4)
    yt = 5 * tf * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
    yt = np.maximum(yt, 0.002)
    camber = 0.035 * (1.0 - b) * 4.0 * xc * (1.0 - xc) * (1.0 - xc * 0.3)
    yfoil = np.where(up, yt, -yt) + camber
    ycirc = 0.5 * np.sin(th)
    y = b * ycirc + (1.0 - b) * yfoil
    Zc = (pa - xc) * chord
    Xc = y * chord
    a = math.radians(twist_deg)
    return np.stack([Xc * math.cos(a) + Zc * math.sin(a), -Xc * math.sin(a) + Zc * math.cos(a)], 1)


def build_blade(spec, part):
    B = spec["blade"]
    Lb = val(B["length"])
    tab = np.asarray(val(B["planform"]), float)
    ns, npts = val(B["stations"]), val(B["section_points"])
    pre = val(B["tip_prebend"])
    u = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, ns))           # clustered at root and tip
    u = np.unique(np.concatenate([u, [0.02, 0.05]]))
    rings = []
    rc = val(B["root_cylinder_length"]) / Lb
    for s in u:
        c, tw, t, pa = (float(np.interp(s, tab[:, 0], tab[:, k])) for k in (1, 2, 3, 4))
        if s <= rc:
            t, c = 1.0, val(B["root_diameter"])
        sec = airfoil_section(c, tw, t, pa, npts)
        y = -1.0 + s * (Lb + 1.0)
        xoff = pre * max((s - 0.2) / 0.8, 0.0) ** 2
        rings.append([(float(p[0]) + xoff, y, float(p[1])) for p in sec])
    part.add(*loft(rings), "blade_white")
    v, f = annulus(2.45, 2.72, 0.0, 0.5, 64)                        # root flange at the pitch bearing
    part.add(v, f, "blade_root_grey", Z_TO_Y())


# ---------------------------------------------------------------- blade surface (for the paint detail)
def blade_rings(spec):
    """build_blade's loft grid: station y (ns,) and rings (ns, npts, 3) in the blade frame, the LE at index npts/2."""
    B = spec["blade"]
    Lb = val(B["length"])
    tab = np.asarray(val(B["planform"]), float)
    ns, npts = val(B["stations"]), val(B["section_points"])
    pre = val(B["tip_prebend"])
    u = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, ns))
    u = np.unique(np.concatenate([u, [0.02, 0.05]]))
    rc = val(B["root_cylinder_length"]) / Lb
    ys, rings = [], []
    for s in u:
        c, tw, t, pa = (float(np.interp(s, tab[:, 0], tab[:, k])) for k in (1, 2, 3, 4))
        if s <= rc:
            t, c = 1.0, val(B["root_diameter"])
        sec = airfoil_section(c, tw, t, pa, npts)
        y = -1.0 + s * (Lb + 1.0)
        xoff = pre * max((s - 0.2) / 0.8, 0.0) ** 2
        ys.append(y)
        rings.append(np.stack([sec[:, 0] + xoff, np.full(npts, y), sec[:, 1]], 1))
    return np.array(ys), np.array(rings)


def blade_surface(spec, spans, arcs, grid=None):
    """Points (len(spans), len(arcs), 3) on the lofted blade (the rendered facets, linear between
    stations and section points), arc measured around the section from the LE (m, + toward the
    lower index side flipped: + = increasing section index), and outward unit normals."""
    ys, G = grid if grid is not None else blade_rings(spec)
    Lb = val(spec["blade"]["length"])
    npts = G.shape[1]
    order = (np.arange(npts + 1) + npts // 2) % npts          # LE, round the loop, back to the LE

    def pts(span, arc):
        y = -1.0 + span / Lb * (Lb + 1.0)
        j = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))
        f = (y - ys[j]) / (ys[j + 1] - ys[j])
        ring = (1.0 - f) * G[j] + f * G[j + 1]
        loop = ring[order]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
        a = np.where(np.asarray(arc) < 0.0, s[-1] + np.asarray(arc), np.asarray(arc))
        P = np.stack([np.interp(a, s, loop[:, k]) for k in range(3)], -1)
        return P, ring.mean(0)
    P = np.empty((len(spans), len(arcs), 3))
    N = np.empty_like(P)
    for i, sp in enumerate(spans):
        p, c = pts(sp, arcs)
        ds = pts(sp + 0.02, arcs)[0] - pts(sp - 0.02, arcs)[0]
        da = pts(sp, np.asarray(arcs) + 0.004)[0] - pts(sp, np.asarray(arcs) - 0.004)[0]
        n = np.cross(ds, da)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n *= np.sign(np.sum(n * (p - c), axis=1))[:, None]
        P[i], N[i] = p, n
    return P, N


def _hash2(i, j, k):
    return np.modf(np.abs(np.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453))[0]


def blade_paint(spec, blade):
    """Paint on blade `blade` (1..3), in the blade frame: a list of (name, P (m, n, 3), keep cells
    (m-1, n-1) or None, vertex colours (m, n, 3) or None, material key). Each sheet is already
    offset along the surface normal by its few mm. Spec block blade_detail."""
    bd = spec["blade_detail"]
    Lb = val(spec["blade"]["length"])
    grid = blade_rings(spec)
    out = []
    ero = bd["erosion"]
    e_sp, e_len = ero["span_m"], ero["length_m"]

    def ragged(sp, arc, half_w, seed):          # the erosion outline: an irregular ellipse
        u = (sp - e_sp) / (0.5 * e_len)
        w = half_w * (0.8 + 0.2 * np.sin(9.0 * u + seed) + 0.12 * np.sin(23.0 * u + 2 * seed))
        rr = u * u + (arc / w) ** 2
        return rr + 0.18 * np.sin(31.0 * arc + 7.0 * u + seed) * np.sin(13.0 * u)
    lep = bd["lep"]
    sp = np.arange(lep["span_from_m"], Lb - 0.4, 0.05)
    ar = np.linspace(-lep["half_width_m"], lep["half_width_m"], 25)
    P, N = blade_surface(spec, sp, ar, grid)
    keep = None
    if blade == ero["blade"]:                   # the strip is chipped off round the patch
        S_, A_ = np.meshgrid(0.5 * (sp[1:] + sp[:-1]), 0.5 * (ar[1:] + ar[:-1]), indexing="ij")
        keep = ragged(S_, A_, ero["half_width_m"] * 1.35, 1.7) > 1.25
    out.append(("lep", P + N * lep["proud_m"], keep, None, "lep"))
    bl = bd["bond_line"]                        # the LE bond line, full span, a hair wide
    sp = np.arange(bl["span_from_m"], Lb - 0.4, 0.25)
    for a0 in (0.0, ):
        P, N = blade_surface(spec, sp, np.array([a0 - 0.5 * bl["width_m"], a0 + 0.5 * bl["width_m"]]), grid)
        out.append(("bond", P + N * bl["proud_m"], None, None, "bond"))
    for s0, a0 in bd["receptors"]["at_span_arc_m"]:          # lightning receptors, round discs
        r = bd["receptors"]["radius_m"]
        sp = np.linspace(s0 - r, s0 + r, 9)
        ar = np.linspace(a0 - r, a0 + r, 9)
        P, N = blade_surface(spec, sp, ar, grid)
        S_, A_ = np.meshgrid(0.5 * (sp[1:] + sp[:-1]) - s0, 0.5 * (ar[1:] + ar[:-1]) - a0, indexing="ij")
        out.append(("receptor", P + N * 0.003, S_ ** 2 + A_ ** 2 < r * r, None, "receptor"))
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("arialbd.ttf", 48)

    def text(txt, span0, arc0, hgt, name, rev=False):
        """Painted text reading along the span (toward the root when rev), unmirrored from outside."""
        x0, y0, x1, y1 = font.getbbox(txt)
        im = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8), 0)
        ImageDraw.Draw(im).text((4 - x0, 4 - y0), txt, fill=255, font=font)
        ink = np.asarray(im) > 110                   # (rows = text down, cols = text right)
        px = hgt / ink.shape[0]
        d = -1.0 if rev else 1.0
        c, n = blade_surface(spec, [span0, span0 + 0.1 * d], [arc0, arc0 + 0.1], grid)
        # text right = d * span; text down = sgn * arc, so right x down points INTO the blade
        sgn = -float(np.sign(np.dot(np.cross(c[1, 0] - c[0, 0], c[0, 1] - c[0, 0]), n[0, 0])))
        sp = span0 + d * np.arange(ink.shape[1] + 1) * px
        ar = arc0 + (np.arange(ink.shape[0] + 1) - 0.5 * ink.shape[0]) * px * sgn
        P, N = blade_surface(spec, sp, ar, grid)
        out.append((name, P + N * 0.003, ink.T, None, "id"))
    idp = bd["blade_id"]                        # the blade ID near the root
    text(f"{idp['prefix']}{blade}", idp["span_m"], idp["arc_m"], idp["height_m"], "blade_id")
    sm = bd["span_marks"]                       # inspection span marks beside the LE, every 10 m
    for s0 in range(sm["from_m"], int(Lb) - 5, sm["every_m"]):
        text(f"{s0}", s0 + 0.35, sm["arc_m"], sm["height_m"], f"span_{s0}", rev=True)
        sp = np.array([s0 - 0.02, s0 + 0.02])
        P, N = blade_surface(spec, sp, np.linspace(sm["arc_m"] - 0.25, sm["arc_m"] + 0.25, 6), grid)
        out.append((f"tick_{s0}", P + N * 0.003, None, None, "id"))
    for s0 in np.arange(lep["span_from_m"] + lep["joint_every_m"], Lb - 1.0, lep["joint_every_m"]):   # LEP segment joints
        sp = np.array([s0 - 0.02, s0 + 0.02])
        P, N = blade_surface(spec, sp, np.linspace(-lep["half_width_m"], lep["half_width_m"], 13), grid)
        if blade == ero["blade"] and abs(s0 - e_sp) < 0.6 * e_len:
            continue
        out.append(("lep_joint", P + N * (lep["proud_m"] + 0.0008), None, None, "id"))
    if blade == ero["blade"]:                   # the erosion patch: pitted, darker, rough
        sp = np.arange(e_sp - 0.6 * e_len, e_sp + 0.6 * e_len, 0.01)
        hw = ero["half_width_m"] * 1.5
        ar = np.arange(-hw, hw, 0.008)
        P, N = blade_surface(spec, sp, ar, grid)
        S_, A_ = np.meshgrid(sp, ar, indexing="ij")
        rr = ragged(S_, A_, ero["half_width_m"], 1.7)
        core = np.clip(1.0 - rr, 0.0, 1.0)
        pit = (_hash2(np.floor(S_ / 0.018), np.floor(A_ / 0.018), 3.0) > 0.62 - 0.35 * core) & (rr < 1.0)
        big = _hash2(np.floor(S_ / 0.05), np.floor(A_ / 0.05), 5.0) > 0.82
        base = np.array([0.36, 0.33, 0.28]) * (1.0 - 0.35 * core)[..., None]      # abraded filler, darker inward
        lam = np.array([0.52, 0.48, 0.38])                                          # exposed laminate
        col = np.where((big & (core > 0.25))[..., None], lam, base)
        col = np.where(pit[..., None], np.array([0.07, 0.065, 0.06]), col)
        col = col * (0.85 + 0.3 * _hash2(np.floor(S_ / 0.01), np.floor(A_ / 0.01), 9.0))[..., None]
        edge = (rr > 0.82) & (rr < 1.0)
        col = np.where(edge[..., None], np.array([0.20, 0.19, 0.17]), col)
        off = np.where(pit, 0.0012, 0.0028)
        keep = (0.25 * (rr[1:, 1:] + rr[:-1, :-1] + rr[1:, :-1] + rr[:-1, 1:])) < 1.0
        out.append(("erosion", P + N * off[..., None], keep, col, "erosion"))
    return out


# ---------------------------------------------------------------- assembly, checks
def pitch_node(spec, i):
    RO = spec["rotor"]
    az = 180.0 + 120.0 * (i - 1)                                    # blade_1 at 6 o'clock
    rh = 0.5 * val(RO["hub_diameter"])
    return R("X", az) @ R("Z", -val(RO["precone_deg"])) @ T(0.0, rh, 0.0) @ R("Y", val(RO["pitch_deg"]))


def rotor_node(spec):
    RO = spec["rotor"]
    ytop = val(spec["tower"]["top_y"])
    return T(val(RO["hub_overhang"]), val(RO["hub_height"]) - ytop, 0.0) @ R("Z", val(RO["shaft_tilt_deg"]))


def build_geometry(spec):
    P = {k: Part() for k in ("monopile", "marine_growth", "anode_cage", "transition_piece", "tower", "nacelle",
                             "rotor", "blade")}
    build_monopile(spec, P["monopile"], P["marine_growth"], P["anode_cage"])
    build_tp(spec, P["transition_piece"])
    build_tower(spec, P["tower"])
    heli = build_nacelle(spec, P["nacelle"])
    build_hub(spec, P["rotor"])
    build_blade(spec, P["blade"])
    return P, heli


def world_verts(part, M):
    V = np.asarray(part.verts, float)
    return V @ M[:3, :3].T + M[:3, 3]


def check(spec, P):
    ytop = val(spec["tower"]["top_y"])
    N = T(0.0, ytop, 0.0)
    Rn = N @ rotor_node(spec)
    tips, lows = [], []
    for i in (1, 2, 3):
        W = world_verts(P["blade"], Rn @ pitch_node(spec, i))
        far = W[np.argmax(np.linalg.norm(W - Rn[:3, 3], axis=1))]
        tips.append(far)
        lows.append(W[:, 1].min())
    hub = Rn[:3, 3]
    rad = [float(np.linalg.norm(t - hub)) for t in tips]
    D = 2.0 * max(rad)
    print(f"[turbine] hub ({hub[0]:.2f}, {hub[1]:.2f}), tip radius {min(rad):.2f}..{max(rad):.2f} m, "
          f"rotor D {D:.1f} m, lowest blade point y {min(lows):.2f} m")
    assert abs(D - val(spec["rotor"]["diameter"])) < 0.02 * val(spec["rotor"]["diameter"]), D
    assert abs(hub[1] - val(spec["rotor"]["hub_height"])) < 1e-6
    tw = np.asarray(P["tower"].verts)
    assert abs(tw[:, 1].max() - ytop) < 0.01 and abs(2 * np.hypot(tw[:, 0], tw[:, 2]).max() - 10.1) < 0.1
    b1 = tips[0]
    assert b1[1] < hub[1] - 100.0 and b1[0] > 11.35, ("blade_1 not at 6 o'clock in front of the tower", b1)
    Wb = world_verts(P["blade"], Rn @ pitch_node(spec, 1))
    r_tw = 0.5 * np.interp(Wb[:, 1].clip(15, ytop), [15, ytop], [10.0, 6.5])
    clear = float((np.hypot(Wb[:, 0], Wb[:, 2]) - r_tw).min())
    print(f"[turbine] blade_1 tower clearance {clear:.2f} m")
    assert clear > 2.0, clear
    return min(lows)


def build_blender(spec, P, heli):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    root = new_object("turbine", None, mats, None, I, props={
        "frame": spec["frame"], "spec": "turbine_spec.json", "state": "inspection: rotor locked, blade_1 at 6 o'clock, feathered"})
    pile = new_object("monopile", P["monopile"], mats, root, I, tile=2.0)
    new_object("marine_growth", P["marine_growth"], mats, pile, I, sharp_deg=80.0, tile=1.0)
    new_object("anode_cage", P["anode_cage"], mats, pile, I)
    new_object("transition_piece", P["transition_piece"], mats, root, I, tile=2.0)
    new_object("tower", P["tower"], mats, root, I, tile=4.0)
    ytop = val(spec["tower"]["top_y"])
    nac = new_object("nacelle", P["nacelle"], mats, root, T(0.0, ytop, 0.0), props={"axis": "+Y (yaw)"})
    rot = new_object("rotor", P["rotor"], mats, nac, rotor_node(spec),
                     props={"axis": "+X (shaft)", "locked": True, "shaft_tilt_deg": val(spec["rotor"]["shaft_tilt_deg"])})
    for i in (1, 2, 3):
        pn = new_object(f"pitch_{i}", None, mats, rot, pitch_node(spec, i),
                        props={"axis": "+Y (span)", "pitch_deg": val(spec["rotor"]["pitch_deg"]), "rest": "feathered"})
        new_object(f"blade_{i}", P["blade"], mats, pn, I, sharp_deg=60.0)
    new_object("helihoist", None, mats, nac, T(*heli))
    ce = spec["empties"]["cable_entry"]
    a = math.radians(ce["azimuth_deg"])
    rp = 0.5 * val(spec["monopile"]["outer_diameter"])
    new_object("cable_entry", None, mats, root,
               T(*pol(rp + 0.55, a, ce["y"])) @ frame_from_forward_up(radial(a), (0.0, 1.0, 0.0)),
               props={"convention": "look -Z (out along the cable), up +Y", "hole_diameter": ce["hole_diameter"]})
    TPs = spec["transition_piece"]
    ab = math.radians(val(TPs["boat_landing_azimuth_deg"]))
    rf = 0.5 * val(TPs["outer_diameter"]) + TPs["boat_landing"]["standoff"] + TPs["boat_landing"]["fender_radius"]
    new_object("boat_landing", None, mats, root,
               T(*pol(rf, ab, 1.0)) @ frame_from_forward_up(radial(ab), (0.0, 1.0, 0.0)),
               props={"convention": "look -Z (out from the fenders), up +Y"})
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="turbine_spec.json")
    ap.add_argument("--out", default="turbine.glb")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(_HERE, p)
    t0 = time.perf_counter()
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    P, heli = build_geometry(spec)
    used = set(m for p in P.values() for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    check(spec, P)
    ntri = {k: sum(len(f) - 2 for f in p.faces) for k, p in P.items()}
    ntri["blade"] *= 3
    print(f"[turbine] geometry in {time.perf_counter() - t0:.1f} s, triangles {sum(ntri.values())} {ntri}")
    if a.check:
        return
    import bpy
    build_blender(spec, P, heli)
    bpy.context.view_layer.update()
    bpy.ops.export_scene.gltf(filepath=path(a.out), export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True)
    print(f"[turbine] wrote {path(a.out)} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

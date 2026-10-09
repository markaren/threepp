"""Build the DJI Matrice 350 RTK carrying a Zenmuse H20T from m350_spec.json: m350.glb (via Blender).

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_m350_blender.py -- --spec m350_spec.json --out m350.glb

Geometry only (plain Python + numpy, no Blender; prints the checks and fails on a gate):
    python build_m350_blender.py --check
    python build_m350_blender.py --check --write-areas    # and the measured projected areas into the spec

The mesh accumulator, the primitives and the Blender export are build_common's (../build_common.py,
shared by every generator); this script adds the aircraft. All geometry is built with numpy in the
model frame (X forward, Y up, Z starboard, metres; the origin on the centreline midway between the
four motor axes, at the height of the arm tubes' axes at the motor mounts) and converted to
Blender's Z-up only when a mesh or a transform is handed to bpy. Every number is the spec's
(`layout` for where things are, `model` for the shapes, `materials`).

Node hierarchy (all names exact):
    m350                         root, identity
      airframe                   body, batteries, arms, the hanging motors, the antennas on the arm
                                 ends, landing gear, the camera's frame and damper
      lamp_front, lamp_rear      the arm lamps under the front (red) and the rear (green) motors:
      lamp_beacon                each one mesh with one emissive material (led_front, led_rear, beacon)
      prop_fr, prop_fl, prop_rl, prop_rr
                                 at the rotor disc centres, UNDER the arm ends (the motors hang, shaft
                                 down); hub and two blades; spins about local +Y (`spin` +1 =
                                 counter-clockwise seen from above), thrust up
      gimbal_pan                 at layout.gimbal_pivot, turns about local +Y
        gimbal_roll              at the head's centre, turns about local +X
          gimbal_tilt            turns about local +Z, a positive turn raises the lens; the head
            camera_wide, camera_zoom, camera_thermal      at their windows
      camera_fpv, imu, rtk_antenna_l, rtk_antenna_r, beacon_top, beacon_bottom

The empties look down local -Z with local +Y up (threepp camera convention).
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
from build_common import (Part, R, T, Z_TO_X, Z_TO_Y, annulus, box, cylinder, export, frame_from_forward_up,  # noqa: E402
                          frame_y_toward, loft, make_materials, new_object, revolve, strut, table)

I4 = np.eye(4)
ROTORS = ("fr", "fl", "rl", "rr")


# ---------------------------------------------------------------- primitives
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rrect(hy, hz, r, seg, eps=0.0012):
    """Rounded rectangle in (y, z), counter-clockwise. Each corner is its arc and a point just
    beyond either end of it, so that the smooth shading's turn stays in the fillet and the flats
    stay flat."""
    r = max(min(r, hy - 1e-5, hz - 1e-5), 1e-5)
    pts = []
    for (cy, cz, a0) in ((hy - r, hz - r, 0.0), (-(hy - r), hz - r, 90.0), (-(hy - r), -(hz - r), 180.0), (hy - r, -(hz - r), 270.0)):
        a, b = math.radians(a0), math.radians(a0 + 90.0)
        flat = (hz - r) if a0 in (0.0, 180.0) else (hy - r)        # half the flat this arc starts from
        nxt = (hy - r) if a0 in (0.0, 180.0) else (hz - r)
        e0, e1 = min(eps, 0.45 * flat), min(eps, 0.45 * nxt)
        pts.append((cy + r * math.cos(a) + e0 * math.sin(a), cz + r * math.sin(a) - e0 * math.cos(a)))
        for k in range(seg + 1):
            t = a + (b - a) * k / seg
            pts.append((cy + r * math.cos(t), cz + r * math.sin(t)))
        pts.append((cy + r * math.cos(b) - e1 * math.sin(b), cz + r * math.sin(b) + e1 * math.cos(b)))
    return pts


def rbox(hx, hy, hz, rc, re=None, seg=6, eseg=4):
    """Box of half sizes (hx, hy, hz) about the origin: its (y, z) section rounded by rc, the edges
    of its two x faces by re. One closed loft along x."""
    rc = min(rc, hy - 1e-4, hz - 1e-4)
    re = min(0.9 * rc if re is None else re, 0.95 * rc, 0.95 * hx)
    st = []
    for k in range(eseg + 1):
        a = 0.5 * math.pi * k / eseg
        st.append((-hx + re * (1.0 - math.cos(a)), re * (1.0 - math.sin(a))))
    st.append((-hx + re + min(0.0012, 0.4 * (hx - re)), 0.0))
    st += [(-x, d) for (x, d) in reversed(st)]
    return loft([[(x, y, z) for (y, z) in rrect(hy - d, hz - d, rc - d, seg)] for (x, d) in st])


def along(axis):
    """Rotation that puts rbox's x on the model's `axis` (its y and z follow cyclically)."""
    M = np.eye(4)
    if axis == "Y":
        M[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    elif axis == "Z":
        M[:3, :3] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    return M


def add_box(part, mat, x, y, z, rc, re=None, axis="X", M=None, seg=6, eseg=4):
    """Rounded box between x = (x0, x1), y = (y0, y1), z = (z0, z1): the four edges along `axis`
    rounded by rc, the two faces across it by re."""
    c = (0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (z[0] + z[1]))
    h = (0.5 * abs(x[1] - x[0]), 0.5 * abs(y[1] - y[0]), 0.5 * abs(z[1] - z[0]))
    k = "XYZ".index(axis)
    v, f = rbox(h[k], h[(k + 1) % 3], h[(k + 2) % 3], rc, re, seg, eseg)
    A = T(*c) @ along(axis)
    part.add(v, f, mat, A if M is None else M @ A)


def toward(origin, d):
    """Placement that puts a solid of revolution's +Z along d at origin."""
    return T(*origin) @ frame_y_toward(unit(d)) @ Z_TO_Y()


def spun(part, profile, mat, origin, d, n=32, M=None):
    v, f = revolve(profile, n)
    A = toward(origin, d)
    part.add(v, f, mat, A if M is None else M @ A)


def ridges(part, origin, d, r, z0, z1, count, mat, w=0.0030, h=0.0014, M=None):
    """Knurling: `count` bars along the axis on the radius r."""
    F = toward(origin, d)
    if M is not None:
        F = M @ F
    v, f = box(w, 2.0 * h, z1 - z0)
    for k in range(count):
        part.add(v, f, mat, F @ R("Z", 360.0 * k / count) @ T(0.0, r, 0.5 * (z0 + z1)))


def ball(part, centre, r, mat, n=16):
    prof = [(r * math.sin(a), -r * math.cos(a)) for a in np.linspace(0.0, math.pi, 9)]
    prof[0], prof[-1] = (0.0, -r), (0.0, r)
    spun(part, prof, mat, centre, (0, 1, 0), n)


def lens(part, at, d, barrel_r, glass_r, proud, glass="lens_glass", ring="lens_ring", n=32, M=None):
    """A window: a bezel standing `proud` of the surface at `at`, a cone inside it down to the
    domed glass (the hood a lens sits in: it reads as a lens, where a flush dark disc reads as a button)."""
    gi, d0 = 0.72 * glass_r, proud - 0.30 * glass_r
    spun(part, [(barrel_r, -0.004), (barrel_r, proud - 0.0008), (barrel_r - 0.0008, proud), (glass_r, proud), (gi, d0)], ring, at, d, n, M)
    spun(part, [(0.985 * gi, d0 - 0.0015), (0.985 * gi, d0 + 0.0003), (0.62 * gi, d0 + 0.10 * glass_r), (0.0, d0 + 0.14 * glass_r)],
         glass, at, d, n, M)


def merged(*parts):
    out = Part()
    for p in parts:
        b = len(out.verts)
        out.verts.extend(p.verts)
        out.faces.extend(tuple(b + k for k in f) for f in p.faces)
        out.mats.extend(p.mats)
    return out


# ---------------------------------------------------------------- the airframe
def rotor_pos(spec, name):
    return np.array(spec["layout"]["rotors"][name]["pos"], float)


def build_body(spec, A, lamps):
    B, S = spec["model"]["body"], spec["model"]
    ch, tw, ts = B["chassis"], B["tower"], B["top_shell"]
    add_box(A, "chassis_grey", ch["x"], ch["y"], (-ch["half_width"], ch["half_width"]), ch["corner_r"], 0.006, axis="Y", seg=10, eseg=6)
    add_box(A, "body_grey", tw["x"], tw["y"], (-tw["half_width"], tw["half_width"]), tw["corner_r"], tw["edge_r"], seg=8, eseg=6)
    add_box(A, "body_grey", ts["x"], ts["y"], (-ts["half_width"], ts["half_width"]), ts["chamfer"], ts["edge_r"], axis="Y", seg=1, eseg=3)
    y_top = ts["y"][1]
    # the cooling ribs on the top shell, the slots in the nose
    rb = B["rib"]
    for x in np.linspace(rb["x"][0], rb["x"][1], rb["count"]):
        add_box(A, "trim_black", (x - 0.5 * rb["thickness"], x + 0.5 * rb["thickness"]), (y_top - 0.002, y_top + rb["height"]),
                (-rb["half_width"], rb["half_width"]), 0.0018, 0.0015, axis="Z", seg=3, eseg=2)
    for s in (-1.0, 1.0):                                # a panel seam along either side of the tower
        v, f = box(tw["x"][1] - tw["x"][0] - 0.050, 0.0026, 0.0016)
        A.add(v, f, "trim_black", T(0.5 * (tw["x"][0] + tw["x"][1]), B["seam_y"], s * (tw["half_width"] + 0.0001)))
    nv = B["nose_vent"]
    x_nose = tw["x"][1]
    for k in range(nv["count"]):
        z = (k - 0.5 * (nv["count"] - 1)) * nv["pitch"]
        add_box(A, "trim_black", (x_nose - 0.003, x_nose + 0.0006), nv["y"], (z - 0.5 * nv["width"], z + 0.5 * nv["width"]),
                0.003, 0.0004, seg=3, eseg=1)
    # the batteries, their level panels, the ribs on their outer sides, the locking knob between them
    BT = S["battery"]
    sx, sy, sz = BT["size"]
    x0, x1 = BT["centre_x"] - 0.5 * sx, BT["centre_x"] + 0.5 * sx
    y0, y1 = BT["centre_y"] - 0.5 * sy, BT["centre_y"] + 0.5 * sy
    pn, sr = BT["panel"], BT["side_ribs"]
    for s in (-1.0, 1.0):
        zc = s * BT["centre_z"]
        add_box(A, "battery_grey", (x0, x1), (y0, y1), (zc - 0.5 * sz, zc + 0.5 * sz), BT["corner_r"], 0.010)
        pw, ph = pn["size"]
        add_box(A, "trim_black", (x0 - 0.0012, x0 + 0.003), (pn["y"] - 0.5 * ph, pn["y"] + 0.5 * ph), (zc - 0.5 * pw, zc + 0.5 * pw),
                0.004, 0.0006, seg=3, eseg=1)
        for k in range(pn["leds"]):
            z = zc + (k - 0.5 * (pn["leds"] - 1)) * 0.007 - s * 0.006
            v, f = box(0.002, 0.0055, 0.003)
            A.add(v, f, "indicator_green", T(x0 - 0.0014, pn["y"], z))
        spun(A, [(0.0045, 0.0), (0.0045, 0.0028), (0.0035, 0.0034)], "chassis_grey", (x0 - 0.001, pn["y"], zc + s * 0.017), (-1, 0, 0), 16)
        for x in np.linspace(sr["x"][0], sr["x"][1], sr["count"]):
            zo = zc + s * 0.5 * sz
            add_box(A, "battery_grey", (x - 0.5 * sr["thickness"], x + 0.5 * sr["thickness"]), sr["y"],
                    (min(zo - s * 0.002, zo + s * sr["proud"]), max(zo - s * 0.002, zo + s * sr["proud"])), 0.0018, 0.0015,
                    axis="Y", seg=2, eseg=2)
    kn = BT["knob"]
    spun(A, [(0.012, -0.012), (kn["radius"] - 0.003, -0.004), (kn["radius"], 0.0), (kn["radius"], kn["proud"] - 0.003),
             (kn["radius"] - 0.003, kn["proud"]), (0.0, kn["proud"] + 0.001)], "trim_black", (x0, kn["y"], 0.0), (-1, 0, 0), 32)
    for k in range(kn["lobes"]):
        a = 2.0 * math.pi * k / kn["lobes"]
        v, f = cylinder(0.0048, 0.001, kn["proud"] - 0.002, 12)
        A.add(v, f, "trim_black", toward((x0, kn["y"] + kn["radius"] * math.cos(a), kn["radius"] * math.sin(a)), (-1, 0, 0)))
    # the vision windows (pairs), the FPV camera and its auxiliary window in the nose
    V = S["vision"]
    for name, row in (("front", (0, 0, 1)), ("rear", (0, 0, 1)), ("top", (0, 0, 1)), ("bottom", (0, 0, 1))):
        c, d, half, r = V[name]
        for s in (-1.0, 1.0):
            p = np.array(c, float) + s * half * np.array(row, float)
            lens(A, p, d, r + 0.0035, r, 0.0022)
    c, d, half, r = V["side"]
    for sz_ in (-1.0, 1.0):
        for s in (-1.0, 1.0):
            lens(A, (c[0] + s * half, c[1], sz_ * c[2]), (0, 0, sz_), r + 0.0035, r, 0.0022)
    # the rear corners of the top shell: one window each, looking out across the cut corner
    ym = 0.5 * (ts["y"][0] + ts["y"][1])
    cf = ts["chamfer"]
    for s in (-1.0, 1.0):
        lens(A, (ts["x"][0] + 0.5 * cf, ym, s * (ts["half_width"] - 0.5 * cf)), (-1, 0, s), 0.0115, 0.008, 0.0018)
    fp = np.array(S["sensors"]["camera_fpv"]["position"], float)
    lens(A, (x_nose, fp[1], fp[2]), (1, 0, 0), 0.0165, 0.012, 0.004)
    add_box(A, "lens_ring", (x_nose - 0.002, x_nose + 0.0016), (fp[1] - 0.013, fp[1] + 0.013), (0.021, 0.033), 0.0055, 0.0008, seg=4, eseg=1)
    for dy in (-0.006, 0.006):
        spun(A, [(0.0036, 0.0), (0.0036, 0.0022), (0.0, 0.0026)], "lens_glass", (x_nose, fp[1] + dy, 0.027), (1, 0, 0), 16)
    for s in (-1.0, 1.0):
        spun(A, [(0.0042, 0.0), (0.0042, 0.0012), (0.0, 0.0016)], "lens_glass", (x_nose, fp[1] - 0.050, s * 0.046), (1, 0, 0), 16)
    # the beacons: a dark seat on the airframe, the lamp itself on its own node
    for key, up in (("beacon_top", 1.0), ("beacon_bottom", -1.0)):
        p = np.array(S["sensors"][key]["position"], float)
        y_seat = y_top if up > 0 else ch["y"][0]
        spun(A, [(0.013, -0.002), (0.013, 0.0015), (0.011, 0.0025)], "trim_black", (p[0], y_seat, p[2]), (0, up, 0), 24)
        h = abs(p[1] - y_seat)
        spun(lamps["lamp_beacon"], [(0.0098, 0.0005), (0.0098, 0.45 * h), (0.0075, 0.85 * h), (0.0, h)], "beacon", (p[0], y_seat, p[2]), (0, up, 0), 24)


def build_arm(spec, A, lamps, name, cans):
    S = spec["model"]
    AR, MO = S["arm"], S["motor"]
    tip = rotor_pos(spec, name).copy()
    tip[1] = 0.0
    sx, sz = (1.0 if tip[0] > 0 else -1.0), (1.0 if tip[2] > 0 else -1.0)
    root = np.array(AR["root_front"] if sx > 0 else AR["root_rear"], float) * (1.0, 1.0, sz)
    d = unit(tip - root)
    L = float(np.linalg.norm(tip - root))
    dh = unit((d[0], 0.0, d[2]))
    ch = S["body"]["chassis"]
    up, down = (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)
    # the hinge block on the chassis' corner, the locking sleeve, the tube
    add_box(A, "chassis_grey", (root[0] - 0.030, root[0] + 0.030), (ch["y"][0] + 0.004, root[1] + 0.024),
            (min(sz * (ch["half_width"] - 0.030), sz * (ch["half_width"] + 0.016)), max(sz * (ch["half_width"] - 0.030), sz * (ch["half_width"] + 0.016))),
            0.012, 0.008, axis="Y", seg=10, eseg=6)
    spun(A, [(0.021, -0.012), (0.024, -0.008), (0.024, 0.010), (0.018, 0.016)], "chassis_grey", root, d, 32)
    sl = AR["sleeve"]
    r, a, ln = sl["radius"], sl["start"], sl["length"]
    spun(A, [(r - 0.004, a), (r, a + 0.003), (r, a + 0.56 * ln), (r - 0.0022, a + 0.60 * ln), (r - 0.0022, a + 0.92 * ln), (r - 0.006, a + ln)],
         "trim_black", root, d, 36)
    ridges(A, root, d, r, a + 0.006, a + 0.54 * ln, 20, "trim_black")
    strut(A, root, tip - 0.012 * d, AR["tube_radius"], "carbon", 32)
    if sx > 0:
        rbd = AR["red_band"]
        v, f = annulus(AR["tube_radius"] - 0.001, AR["tube_radius"] + 0.0007, L - rbd["from_motor"], L - rbd["from_motor"] + rbd["length"], 32)
        A.add(v, f, "marking_red", toward(root, d))
    # the clamp at the tip and the motor's base on the arm's end: the base reaches above the tube, where
    # the antenna stands, and below it, where the motor hangs (its heights are measured DOWN from the axis)
    cl = AR["clamp"]
    spun(A, [(cl["radius"] - 0.004, L - cl["length"]), (cl["radius"], L - cl["length"] + 0.004), (cl["radius"], L - 0.004)], "trim_black", root, d, 32)
    y0, y1 = MO["base_y"]
    br = MO["base_radius"]
    spun(A, [(br - 0.006, y0), (br, y0 + 0.004), (br, y1 - 0.003), (br - 0.003, y1)], "trim_black", tip, down, 40)
    top = -y0
    if sx > 0:                                           # a slim upright antenna on the front arms
        an = AR["antenna_front"]
        side = np.cross(dh, up)
        rings = []
        for t, k in ((0.0, 1.0), (0.3, 1.0), (0.6, 1.0), (0.85, 1.0), (0.94, 0.86), (0.985, 0.6), (1.0, 0.3)):
            c = 0.5 * k * (an["chord"][0] + (an["chord"][1] - an["chord"][0]) * t)
            th = 0.5 * k * (an["thickness"][0] + (an["thickness"][1] - an["thickness"][0]) * t)
            rings.append([tip + dh * (c * math.cos(q)) + side * (th * math.sin(q)) + np.array([0.0, top - 0.003 + an["height"] * t, 0.0])
                          for q in np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)])
        v, f = loft(rings)
        A.add(v, f, "trim_black")
    else:                                                # a short wide cylinder with a domed cap on the rear ones
        rd = AR["radome_rear"]
        rr, hh = rd["radius"], rd["height"]
        spun(A, [(rr - 0.004, top - 0.003), (rr, top + 0.002), (rr, top + hh - 0.017), (rr - 0.003, top + hh - 0.008),
                 (0.62 * rr, top + hh - 0.002), (0.0, top + hh)], "trim_black", tip, up, 40)
    # the arm lamp: a small lens on the outer side of the base
    lp = MO["lamp"]
    spun(lamps["lamp_front" if sx > 0 else "lamp_rear"],
         [(lp["radius"], -0.003), (lp["radius"], 0.35 * lp["proud"]), (0.7 * lp["radius"], 0.8 * lp["proud"]), (0.0, lp["proud"])],
         "led_front" if sx > 0 else "led_rear", tip + dh * (br - 0.002) + np.array([0.0, 0.5 * (top - y1), 0.0]), dh, 24)
    # the motor, hanging: the stator in the gap, the bell under it
    spun(A, [(MO["stator_radius"], y1 - 0.001), (MO["stator_radius"], MO["can_y"][0] + 0.001)], "motor_steel", tip, down, 40)
    cr = MO["can_radius"]
    c0, c1 = MO["can_y"]
    prof = [(cr - 0.0015, c0), (cr, c0 + 0.002), (cr, c1 - 0.003), (cr - 0.002, c1), (0.58 * cr, c1 + 0.004), (0.34 * cr, MO["bell_top_y"])]
    v, f = revolve(prof, 56)
    Mc = toward(tip, down)
    A.add(v, f, "motor_anodised", Mc)
    cans[name] = np.asarray(v, float) @ Mc[:3, :3].T + Mc[:3, 3]
    for k in range(6):                                   # the bell's spokes
        a = 2.0 * math.pi * k / 6
        v, f = box(0.016, 0.0016, 0.005)
        A.add(v, f, "motor_steel", T(*tip) @ R("Y", math.degrees(a)) @ T(0.60 * cr, -(c1 + 0.0026), 0.0) @ R("Z", 11.0))


def build_gear(spec, A):
    S = spec["model"]
    G = S["gear"]
    ch = S["body"]["chassis"]
    for s in (-1.0, 1.0):
        top = np.array(G["top"], float) * (1.0, 1.0, s)
        foot = np.array(G["foot"], float) * (1.0, 1.0, s)
        d = unit(foot - top)
        L = float(np.linalg.norm(foot - top))
        add_box(A, "chassis_grey", (-0.030, 0.030), (ch["y"][0] - 0.012, ch["y"][0] + 0.004), (min(s * (abs(top[2]) - 0.026), s * (abs(top[2]) + 0.014)), max(s * (abs(top[2]) - 0.026), s * (abs(top[2]) + 0.014))),
                0.010, 0.005, axis="Y")
        strut(A, top - 0.006 * d, top + G["upper_share"] * L * d, G["upper_radius"], "trim_black", 28)
        co = G["collar"]
        a, ln, r = co["at"] * L, co["length"], co["radius"]
        spun(A, [(r - 0.005, a), (r, a + 0.003), (r, a + ln - 0.003), (r - 0.005, a + ln)], "trim_black", top, d, 32)
        ridges(A, top, d, r, a + 0.005, a + ln - 0.005, 18, "trim_black")
        spun(A, [(G["upper_radius"], G["upper_share"] * L - 0.001), (G["tube_radius"] + 0.0005, G["upper_share"] * L + 0.012)], "trim_black", top, d, 28)
        fs = G["foot_sleeve"]                             # the thicker lower end the photograph shows
        spun(A, [(G["tube_radius"] + 0.001, L - fs["length"]), (fs["radius"], L - fs["length"] + 0.008), (fs["radius"], L - 0.004)], "trim_black", top, d, 28)
        strut(A, top + (G["upper_share"] * L - 0.004) * d, foot, G["tube_radius"], "carbon", 28)
        sk = G["skid"]
        hl = 0.5 * sk["length"]
        up = np.array([0.0, 0.0045, 0.0])                 # the T piece sits a little high on the bar: the rubber ends are what lands
        spun(A, [(0.012, -0.030), (0.015, -0.026), (0.015, 0.026), (0.012, 0.030)], "trim_black", foot + up, (1, 0, 0), 28)
        ball(A, foot + up, 0.0148, "trim_black", 20)
        strut(A, foot - np.array([hl - 0.02, 0.0, 0.0]), foot + np.array([hl - 0.02, 0.0, 0.0]), sk["radius"], "carbon", 24)
        rr, rl = sk["rubber_radius"], sk["rubber_length"]
        for e in (-1.0, 1.0):
            spun(A, [(rr - 0.002, 0.0), (rr, 0.003), (rr, rl - 0.008), (0.75 * rr, rl - 0.002), (0.0, rl)], "rubber",
                 foot + np.array([e * (hl - rl), 0.0, 0.0]), (e, 0, 0), 24)


def build_damper(spec, A):
    D = spec["model"]["damper"]
    px, py, pz = spec["layout"]["gimbal_pivot"]
    h = D["half_size"]
    for (y0, y1) in (D["upper_y"], D["lower_y"]):
        add_box(A, "trim_black", (px - h, px + h), (y0, y1), (pz - h, pz + h), 0.014, 0.0015, axis="Y", seg=5, eseg=2)
    yb = 0.5 * (D["upper_y"][0] + D["lower_y"][1])
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            ball(A, (px + sx * D["ball_offset"], yb, pz + sz * D["ball_offset"]), D["ball_radius"], "rubber", 16)
    F = D["frame"]                                       # the tubular frame the plates hang in: wide at the body, narrow below
    yl = 0.5 * (D["lower_y"][0] + D["lower_y"][1])
    for x in F["x"]:
        for s in (-1.0, 1.0):
            strut(A, (x, F["top_y"], s * F["top_half_width"]), (x, yl, s * h), F["tube_radius"], "trim_black", 14)
        strut(A, (x, yl, -h - 0.003), (x, yl, h + 0.003), F["tube_radius"], "trim_black", 14)
    for s in (-1.0, 1.0):
        strut(A, (F["x"][0] - 0.004, yl, s * h), (F["x"][1] + 0.004, yl, s * h), F["tube_radius"], "trim_black", 14)
    spun(A, [(D["connector_radius"], D["lower_y"][0] + 0.001), (D["connector_radius"], py + 0.0008), (D["connector_radius"] - 0.001, py)],
         "lens_ring", (px, 0.0, pz), (0, 1, 0), 36)


def build_airframe(spec):
    A = Part()
    lamps = {"lamp_front": Part(), "lamp_rear": Part(), "lamp_beacon": Part()}
    cans = {}
    build_body(spec, A, lamps)
    for name in ROTORS:
        build_arm(spec, A, lamps, name, cans)
    build_gear(spec, A)
    build_damper(spec, A)
    return A, lamps, cans


# ---------------------------------------------------------------- the propellers
def build_propeller(spec, spin):
    """(hub, blades) in the propeller node's frame: the disc centre at the origin, the axis local +Y,
    the blades along local +-X. The propeller hangs under its motor and still pushes the air down:
    `spin` +1 turns counter-clockwise seen from above, so the blade on +X moves toward -Z and carries
    its leading edge there, raised above its trailing edge."""
    P, PR = spec["model"]["propeller"], spec["aircraft"]["propeller"]
    Rt, r0 = 0.5 * PR["diameter_m"], P["root_r"]
    chord = table(P["chord"])
    hub, blades = Part(), Part()
    fs = set(np.round(np.linspace(0.0, 1.0, 27), 5).tolist()) | {0.97, 0.99}
    for b in P["bands"]:
        fs |= {b[0], b[1]}
    fs = sorted(fs)
    rings = []
    n = 12
    for f in fs:
        r = r0 + (Rt - r0) * f
        c = chord(f)
        th = P["thickness"][0] + (P["thickness"][1] - P["thickness"][0]) * f
        beta = min(math.atan(PR["pitch_m"] / (2.0 * math.pi * r)), math.radians(P["max_twist_deg"]))
        ch = np.array([0.0, math.sin(beta), -spin * math.cos(beta)])          # trailing edge -> leading edge
        nn = np.array([0.0, math.cos(beta), spin * math.sin(beta)])
        ring = []
        for k in range(n):
            q = 2.0 * math.pi * k / n
            u, w = 0.5 * math.cos(q), 0.5 * math.sin(q) * (1.0 + 0.35 * math.cos(q))
            p = u * c * ch + w * th * nn
            ring.append((r, p[1], p[2]))
        rings.append(ring)
    v, f = loft(rings)

    def band(fm):
        for b in P["bands"]:
            if b[0] <= fm <= b[1]:
                return b[2]
        return "prop_black"
    mats = []
    for i in range(len(fs) - 1):
        mats += [band(0.5 * (fs[i] + fs[i + 1]))] * n
    mats += ["prop_black", band(1.0)]
    for ang in (0.0, 180.0):
        blades.add_multi(v, f, mats, R("Y", ang))
    # the hub, under the motor: the cap below the bell, the bar the two blades fold on, their pivots
    # (hub_y is measured DOWN from the disc centre)
    down = (0.0, -1.0, 0.0)
    h0, h1 = P["hub_y"]
    hr = P["hub_radius"]
    spun(hub, [(0.72 * hr, h0), (hr, h0 + 0.004), (hr, h1 - 0.006), (0.8 * hr, h1 - 0.0015), (0.0, h1)], "motor_anodised", (0, 0, 0), down, 32)
    cx, cy, cz = P["clamp"]
    add_box(hub, "prop_black", (-0.5 * cx, 0.5 * cx), (-0.001, cy - 0.001), (-0.5 * cz, 0.5 * cz), 0.0105, 0.002, axis="Y", seg=6, eseg=2)
    for e in (-1.0, 1.0):
        spun(hub, [(0.0062, -cy), (0.0062, 0.0042), (0.0045, 0.0052)], "motor_steel", (e * r0, 0.0025, 0.0), down, 16)
    return hub, blades


# ---------------------------------------------------------------- the gimbal and the camera
def build_gimbal(spec):
    """(pan, roll, tilt) parts, each in its node's frame, and the three camera frames in the tilt node's."""
    G = spec["model"]["gimbal"]
    drop = G["head_drop"]
    pan, roll, tilt = Part(), Part(), Part()
    pm = G["pan_motor"]
    spun(pan, [(pm["radius"] - 0.004, -pm["height"]), (pm["radius"], -pm["height"] + 0.004), (pm["radius"], -0.003), (pm["radius"] - 0.002, 0.0)],
         "payload_grey", (0, 0, 0), (0, 1, 0), 40)
    rm = G["roll_motor"]
    add_box(pan, "payload_grey", (rm["x"][0] + 0.002, -0.008), (-pm["height"] + 0.002, -0.010), (-0.015, 0.015), 0.007, 0.004)
    add_box(pan, "payload_grey", (rm["x"][0], rm["x"][0] + 0.022), (-drop - 0.012, -0.010), (-0.015, 0.015), 0.007, 0.004, axis="Y")
    ln = rm["x"][1] - rm["x"][0]
    spun(pan, [(rm["radius"] - 0.003, 0.0), (rm["radius"], 0.003), (rm["radius"], ln - 0.003), (rm["radius"] - 0.003, ln)], "payload_grey",
         (rm["x"][0], -drop, 0.0), (1, 0, 0), 36)
    Y = G["yoke"]
    hh, (z0, z1) = Y["half_height"], Y["arm_z"]
    add_box(roll, "payload_grey", Y["bar_x"], (-hh, hh), (-z1, z1), 0.006, 0.004, axis="Z")
    tm = G["tilt_motor"]
    for s in (-1.0, 1.0):
        add_box(roll, "payload_grey", (Y["bar_x"][0], 0.012), (-hh + 0.001, hh - 0.001), (min(s * z0, s * z1), max(s * z0, s * z1)), 0.005, 0.004)
        w = tm["z"][1] - tm["z"][0]
        spun(roll, [(tm["radius"] - 0.003, 0.0), (tm["radius"], 0.003), (tm["radius"], w - 0.004), (tm["radius"] - 0.004, w)], "payload_grey",
             (0.0, 0.0, s * tm["z"][0]), (0, 0, s), 36)
        spun(roll, [(0.011, w - 0.001), (0.011, w + 0.0012), (0.009, w + 0.002)], "lens_ring", (0.0, 0.0, s * tm["z"][0]), (0, 0, s), 24)
    H = G["head"]
    add_box(tilt, "payload_grey", H["x"], (-H["half_height"], H["half_height"]), (-H["half_width"], H["half_width"]), H["corner_r"], H["edge_r"],
            seg=7, eseg=5)
    xf = H["x"][1]
    add_box(tilt, "lens_ring", (xf - 0.004, xf + 0.0012), (-H["half_height"] + 0.009, H["half_height"] - 0.009),
            (-H["half_width"] + 0.007, H["half_width"] - 0.007), 0.009, 0.0008, seg=5, eseg=1)
    W = G["windows"]
    cams = {}
    for key, glass in (("zoom", "lens_glass"), ("thermal", "thermal_lens"), ("wide", "lens_glass")):
        w = W[key]
        lens(tilt, w["at"], (1, 0, 0), w["barrel_r"], w["glass_r"], w["proud"], glass=glass, n=40)
        cams["camera_" + key] = T(w["at"][0] + w["proud"], w["at"][1], w["at"][2]) @ frame_from_forward_up((1, 0, 0), (0, 1, 0))
    lr = W["rangefinder"]
    ax, ay, az = lr["at"]
    hy_, hz_ = 0.5 * lr["size"][0], 0.5 * lr["size"][1]
    add_box(tilt, "lens_glass", (ax - 0.002, ax + lr["proud"] + 0.0012), (ay - hy_, ay + hy_), (az - hz_, az + hz_), 0.005, 0.0008, seg=4, eseg=1)
    return pan, roll, tilt, cams


# ---------------------------------------------------------------- the nodes
def build_nodes(spec):
    """[(name, parent, part or None, local matrix, props)] in creation order, and what the checks need."""
    S, Lay = spec["model"], spec["layout"]
    A, lamps, cans = build_airframe(spec)
    nodes = [("m350", None, None, I4, {
        "frame": "X forward, Y up, Z starboard; origin on the centreline midway between the motor axes, at the arm tubes' axes at the motor mounts",
        "spec": "m350_spec.json"})]
    nodes.append(("airframe", "m350", A, I4, {}))
    for k in ("lamp_front", "lamp_rear", "lamp_beacon"):
        nodes.append((k, "m350", lamps[k], I4, {"material": {"lamp_front": "led_front", "lamp_rear": "led_rear", "lamp_beacon": "beacon"}[k]}))
    hubs, blades = {}, {}
    for name in ROTORS:
        spin = Lay["rotors"][name]["spin"]
        hub, bl = build_propeller(spec, float(spin))
        hubs[name], blades[name] = hub, bl
        nodes.append(("prop_" + name, "m350", merged(hub, bl), T(*Lay["rotors"][name]["pos"]),
                      {"axis": "+Y", "spin": spin, "positive": "counter-clockwise seen from above"}))
    pan, roll, tilt, cams = build_gimbal(spec)
    nodes.append(("gimbal_pan", "m350", pan, T(*Lay["gimbal_pivot"]), {"axis": "+Y"}))
    nodes.append(("gimbal_roll", "gimbal_pan", roll, T(0.0, -S["gimbal"]["head_drop"], 0.0), {"axis": "+X"}))
    nodes.append(("gimbal_tilt", "gimbal_roll", tilt, I4, {"axis": "+Z", "positive": "raises the lens"}))
    for k in ("camera_wide", "camera_zoom", "camera_thermal"):
        nodes.append((k, "gimbal_tilt", None, cams[k], {"convention": "look -Z, up +Y"}))
    SE = S["sensors"]
    fp = SE["camera_fpv"]
    nodes.append(("camera_fpv", "m350", None, T(*fp["position"]) @ frame_from_forward_up(fp["look"], (0, 1, 0)), {"convention": "look -Z, up +Y"}))
    for k in ("imu", "rtk_antenna_l", "rtk_antenna_r", "beacon_top", "beacon_bottom"):
        nodes.append((k, "m350", None, T(*SE[k]["position"]), {}))
    return nodes, {"cans": cans, "hubs": hubs, "blades": blades}


def world_matrices(nodes):
    W = {}
    for name, parent, _, M, _ in nodes:
        W[name] = M if parent is None else W[parent] @ M
    return W


# ---------------------------------------------------------------- checks
def triangles(part, M):
    V = np.asarray(part.verts, float) @ M[:3, :3].T + M[:3, 3]
    idx = [(f[0], f[i], f[i + 1]) for f in part.faces for i in range(1, len(f) - 1)]
    return V[np.asarray(idx, int)]


def projected_area(tris, drop, cell):
    """Area of the union of the triangles seen along axis `drop`: cell centres of a grid inside any of them."""
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


def check(spec, nodes, extra):
    S, Lay = spec["model"], spec["layout"]
    W = world_matrices(nodes)
    print("[m350] nodes:")
    for name, parent, part, _, _ in nodes:
        kind = "empty" if part is None else f"mesh, {sum(len(f) - 2 for f in part.faces)} triangles, " + ", ".join(sorted(set(part.mats)))
        print(f"[m350]   {name:<16} < {parent or '-':<12} {kind}")
    tris_n = sum(len(f) - 2 for _, _, p, _, _ in nodes if p is not None for f in p.faces)
    cans = extra["cans"]
    axes = {k: v.mean(axis=0) for k, v in cans.items()}
    diag = [float(np.linalg.norm((axes[a] - axes[b])[[0, 2]])) for a, b in (("fr", "rl"), ("fl", "rr"))]
    allc = np.concatenate(list(cans.values()))
    width = float(allc[:, 2].max() - allc[:, 2].min())
    solid, fixed = [], []                                # everything but the propeller blades; and but the propellers
    for name, _, part, _, _ in nodes:
        if part is None:
            continue
        if name.startswith("prop_"):
            solid.append(triangles(extra["hubs"][name[5:]], W[name]))
        else:
            solid.append(triangles(part, W[name]))
            fixed.append(solid[-1])
    solid, fixed = np.concatenate(solid), np.concatenate(fixed)
    hub_low = min(float(triangles(extra["hubs"][k], W["prop_" + k])[:, :, 1].min()) for k in ROTORS)
    air = triangles(nodes[1][2], I4)
    skid = float(air[:, :, 1].min())
    top_fixed = float(fixed[:, :, 1].max())              # the highest fixed point: the propellers hang under the arms
    height = top_fixed - skid
    arm_top = float(air[np.abs(air[:, :, 2]).min(axis=1) > 0.25][:, :, 1].max())
    for k in ROTORS:                                     # every blade climbs toward its leading edge: thrust up
        spin = Lay["rotors"][k]["spin"]
        V = np.asarray(extra["blades"][k].verts)
        sec = V[(V[:, 0] > 0.12) & (V[:, 0] < 0.16)]
        hi, lo = sec[np.argmax(sec[:, 1])], sec[np.argmin(sec[:, 1])]
        assert hi[2] * spin < 0.0 < lo[2] * spin and hi[1] - lo[1] > 0.002, ("blade pitched the wrong way", k)
    length = float(fixed[:, :, 0].max() - fixed[:, :, 0].min())
    tip = 2.0 * max(float(np.hypot(np.asarray(extra["blades"][k].verts)[:, 0], np.asarray(extra["blades"][k].verts)[:, 2]).max()) for k in ROTORS)
    for name, _, _, M, _ in nodes:
        if name.startswith("prop_"):
            p = Lay["rotors"][name[5:]]["pos"]
            assert np.allclose(M[:3, 3], p) and np.allclose(axes[name[5:]][[0, 2]], [p[0], p[2]], atol=1e-6), ("rotor off the layout", name)
    assert np.allclose(W["gimbal_pan"][:3, 3], Lay["gimbal_pivot"])
    G = S["gates"]
    vals = {"diagonal_m": max(diag, key=lambda x: abs(x - G["diagonal_m"][0])), "width_over_cans_m": width, "height_m": height,
            "length_m": length, "tip_circle_m": tip}
    errs = []
    for k, val in vals.items():
        ref, tol = G[k]
        ok = abs(val - ref) <= tol
        print(f"[m350] gate {k:<18} {val:.4f}  ({ref} +- {tol})  {'ok' if ok else 'FAILED'}")
        if not ok:
            errs.append(k)
    print(f"[m350] length without the propellers {length:.4f} m; the product page's {G['product_length_m']} m is NOT met "
          f"(see model.gates.note)")
    print(f"[m350] skids' underside y {skid:.4f} (layout {Lay['skid_bottom_y_m']}); highest fixed point y {top_fixed:.4f} (the top beacon); "
          f"antennas on the arm ends up to y {arm_top:.4f}")
    print(f"[m350] propellers under the arms: disc centres y {Lay['rotor_disc_y_m']}, hubs down to y {hub_low:.4f}, "
          f"{hub_low - skid:.3f} m above the skids' underside; every blade's leading edge is its higher one in the sense it turns")
    print(f"[m350] product numbers met: wheelbase 0.895, width 0.670 (over the motor cans), height 0.430 (skids to the highest fixed "
          f"point: {height:.3f}), propeller 21 in; NOT met: length 0.810 (the model is {length:.3f})")
    print(f"[m350] triangles {tris_n}; every closed primitive closed and consistently wound (Part.add's Newell sum)")
    cell = 0.002
    areas = {"front": projected_area(solid, 0, cell), "side": projected_area(solid, 2, cell), "top": projected_area(solid, 1, cell)}
    print(f"[m350] projected areas without the blades, {1000 * cell:.0f} mm grid: front {areas['front']:.4f}, side {areas['side']:.4f}, "
          f"top {areas['top']:.4f} m^2 (spec: {spec['aero']['projected_area_m2']})")
    assert abs(skid - Lay["skid_bottom_y_m"]) < 0.002, ("skids off the layout", skid)
    assert not errs, ("gates failed", errs)
    return areas, tris_n


def write_areas(spec_path, areas):
    """The three measured numbers into the spec's text, in place: nothing else of the file is touched."""
    with open(spec_path, "r", encoding="utf-8", newline="") as fh:
        txt = fh.read()
    new, n = re.subn(r'"projected_area_m2": \{[^}]*\}',
                     '"projected_area_m2": {"front": %.4f, "side": %.4f, "top": %.4f}' % (areas["front"], areas["side"], areas["top"]), txt, count=1)
    assert n == 1, "projected_area_m2 not found in the spec"
    if new != txt:
        with open(spec_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    print(f"[m350] projected areas written to {os.path.basename(spec_path)}")


# ---------------------------------------------------------------- Blender side
def build_blender(spec, nodes):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    obj = {}
    for name, parent, part, M, props in nodes:
        obj[name] = new_object(name, part, mats, obj.get(parent), M, props=props, sharp_deg=32.0)
    return obj["m350"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="m350_spec.json")
    ap.add_argument("--out", default="m350.glb")
    ap.add_argument("--check", action="store_true", help="build the geometry and print the checks, no Blender")
    ap.add_argument("--write-areas", action="store_true", help="write the measured projected areas into the spec")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    nodes, extra = build_nodes(spec)
    used = set(m for _, _, p, _, _ in nodes if p is not None for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    areas, _ = check(spec, nodes, extra)
    if a.write_areas:
        write_areas(path(a.spec), areas)
    if a.check:
        return
    build_blender(spec, nodes)
    tris_n = export(path(a.out))
    print(f"[m350] wrote {path(a.out)}  triangles={tris_n}  bytes={os.path.getsize(path(a.out))}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

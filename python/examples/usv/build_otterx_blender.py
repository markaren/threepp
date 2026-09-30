"""Build the Otter X USV from otterx_spec.json: otterx.glb (via Blender) and otterx_hydro.json.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_otterx_blender.py -- --spec otterx_spec.json --out otterx.glb

Hydrostatics only (plain Python + numpy, no Blender; writes otterx_hydro.json):
    python build_otterx_blender.py --hydro-only

The mesh primitives, the hydrostatic integrator and the Blender export are the
Mariner generator's (build_mariner_blender.py, next to this file); this script
adds the catamaran. All geometry is built with numpy in the spec's vessel frame
(X forward, Y up, Z starboard, metres) and converted to Blender's Z-up only when
a mesh or a transform is handed to bpy. The glTF exporter's Y-up conversion maps
it straight back, so every number in the spec is the number in the .glb.

Node hierarchy (all names exact):
    otterx                       root, identity, origin = centreline x keel line x mid-LOA
      pontoons                   both demihulls, the skegs and their heel boxes, the gondola
      body                       head block, midbody, stern block, hatch covers, beam strip,
                                 stern side panels
      deck_fittings              cleats, lifting eye, GNSS antennas, forward camera, livery,
                                 stern handles
      gantry                     ladder plates, top plate, radome, masthead lights, antennas,
                                 equipment boxes, sidelights
      thruster_port, thruster_stbd          azimuth pods, turn about local +Y
        thruster_<side>_rotor               rim-drive rotor, spins about local +X
        thrust_<side>                       empty at the duct centre, thrust along local +X
      camera_main, radar, gnss_port, gnss_stbd, sat_compass, ais_vhf_antenna, mimo_antenna,
      lte_antenna, imu, transducer, mbes_mount                       sensor empties
      nav_light_port, nav_light_stbd, ram_light_top, ram_light_mid, ram_light_bottom, floodlight

Sensor empties look down local -Z with local +Y up (threepp camera convention).

The hydrostatics integrate the buoyant closed meshes column by column (the
Mariner's winding-number integrator), solve the floating position for each
loading condition, and assert the design draft against the brochure's
400 +/- 100 mm.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mariner_blender import (Hydro, Part, R, T, Z_TO_X, Z_TO_Y, box, box_between, clip_slab,  # noqa: E402
                                   cylinder, extrude_xy, frame_from_forward_up, loft, make_materials,
                                   mass_condition, new_object, pchip, revolve, sensor_frame)


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


def lerp2(p, q, y):
    """x on the segment p-q at height y (extrapolated past the ends)."""
    return p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1])


# ---------------------------------------------------------------- pontoon lines
class PontoonLines:
    """One demihull as a function of x.

    The section template (starboard, offsets outboard from the keel line) is an
    asymmetric U: round bilges of different radii into a vertical outer wall and an
    inner side that flares out to the tunnel roof. Forward of taper_start_x the
    offsets shrink on an ellipse toward the stem, and the section's height is
    mapped between the keel profile and the stem profile. The transom ring is
    raked: each point sits on the line from transom bottom to transom top."""

    def __init__(self, spec):
        P = spec["pontoons"]
        self.P = P
        kp = np.asarray(P["keel_profile"], float)
        sp = np.asarray(P["stem_profile"], float)
        self._keel, self._stem = pchip(kp[:, 0], kp[:, 1]), pchip(sp[:, 0], sp[:, 1])
        self.x_tip, self.x_stem = float(kp[-1, 0]), float(sp[0, 0])
        self.tb, self.tt = P["transom"]["bottom"], P["transom"]["top"]
        self.template = self._template()

    def _template(self, n_o=8, n_i=10):
        P = self.P
        a_o = P["outer_z"] - P["keel_z"]
        a_i = P["keel_z"] - P["inner_z_bilge"]
        a_t = P["keel_z"] - P["inner_z_deck"]
        b_o, b_i, d = P["bilge_y_outer"], P["bilge_y_inner"], P["deck_y"]
        pts = [(0.0, 0.0)]
        for k in range(1, n_o + 1):
            t = 0.5 * math.pi * k / n_o
            pts.append((a_o * math.sin(t), b_o * (1.0 - math.cos(t))))
        pts += [(a_o, d), (-a_t, d)]
        for k in range(n_i, 0, -1):
            t = 0.5 * math.pi * k / n_i
            pts.append((-a_i * math.sin(t), b_i * (1.0 - math.cos(t))))
        return pts

    def keel(self, x):
        return self._keel(x)

    def top(self, x):
        return self.P["deck_y"] if x <= self.x_stem else self._stem(x)

    def width(self, x):
        P = self.P
        x0 = P["taper_start_x"]
        if x <= x0:
            return 1.0
        t = min(1.0, (x - x0) / (self.x_tip - x0))
        return max(P["taper_min"], math.sqrt(max(0.0, 1.0 - t * t)))

    def transom_x(self, y):
        (xb, yb), (xt, yt) = self.tb, self.tt
        return xb + (xt - xb) * min(1.0, max(0.0, (y - yb) / (yt - yb)))

    def ring(self, x, side, transom=False):
        P = self.P
        yk, yt, w, d = self.keel(x), self.top(x), self.width(x), P["deck_y"]
        out = []
        for (dz, y) in self.template:
            yy = yk + (y / d) * (yt - yk)
            out.append((self.transom_x(yy) if transom else x, yy, side * (P["keel_z"] + w * dz)))
        return out

    def stations(self):
        S = self.P["stations"]
        x0, x1 = self.tb[0], self.x_tip - 0.002
        xs = set(np.round(np.arange(x0, x1, S["dx"]), 6).tolist())
        for (a, b, dx) in S["fine"]:
            xs |= set(np.round(np.arange(a, b, dx), 6).tolist())
        xs |= {x0, x1, self.P["taper_start_x"], self.x_stem}
        return sorted(set(round(x, 6) for x in xs if x0 <= x <= x1))


def build_pontoons(spec, L, part):
    xs = L.stations()
    for side in (1, -1):
        rings = [L.ring(x, side, transom=(i == 0)) for i, x in enumerate(xs)]
        part.add(*loft(rings), "pontoon_pe_black", buoyant=True)
    A, P = spec["appendages"], spec["pontoons"]
    sk = A["skeg"]
    for side in (1, -1):
        zc = side * P["keel_z"]
        hw = 0.5 * sk["thickness"]
        part.add(*extrude_xy([tuple(p) for p in sk["profile_xy"]], zc - hw, zc + hw), "pontoon_pe_black", buoyant=True)
        h = sk["heel"]
        v, f, M = box_between((h["x_range"][0], h["y_range"][0], zc - h["halfwidth"]),
                              (h["x_range"][1], h["y_range"][1], zc + h["halfwidth"]))
        part.add(v, f, "pontoon_pe_black", M, buoyant=True)
    g = A["gondola"]
    (x0, x1), (y0, y1), nb = g["x_range"], g["y_range"], g["nose_bottom_x"]
    ht, hb = g["halfwidth_top"], g["halfwidth_bottom"]

    def gring(x, yb):
        return [(x, yb, -hb), (x, yb, hb), (x, y1, ht), (x, y1, -ht)]
    rings = [gring(x0, y0), gring(nb, y0), gring(x1, y1 - 0.012)]
    part.add(*loft(rings), "fairing_grey", buoyant=True)


# ---------------------------------------------------------------- the body
def cover_section(B):
    """Hatch cover section (z, y): skirt, rounded shoulders, flat top, closed along the skirt line."""
    C = B["covers"]
    hw, r, top, sk = B["halfwidth"] + C["proud"], C["shoulder_r"], C["top_y"], C["skirt_y"]
    pts = [(-hw, sk)]
    for k in range(7):
        a = math.pi - 0.5 * math.pi * k / 6
        pts.append((-hw + r + r * math.cos(a), top - r + r * math.sin(a)))
    for k in range(7):
        a = 0.5 * math.pi - 0.5 * math.pi * k / 6
        pts.append((hw - r + r * math.cos(a), top - r + r * math.sin(a)))
    pts.append((hw, sk))
    return pts


def body_side_z(B, x, y):
    """|z| of the body's outer side surface at (x, y) below the cover shoulders."""
    C = B["covers"]
    on_cover = any(a <= x <= b for (a, b) in C["x_splits"]) and y >= C["skirt_y"]
    return B["halfwidth"] + (C["proud"] if on_cover else 0.0)


def clip_band(poly, y0, y1):
    """clip_slab in y."""
    return [(x, y) for (y, x) in clip_slab([(y, x) for (x, y) in poly], y0, y1)]


def body_decal(part, B, poly, mat, off=0.003):
    """Paint a side-view polygon onto both body sides, piece by piece so each piece is flat."""
    C = B["covers"]
    xcuts = sorted({p[0] for p in poly} | {c for s in C["x_splits"] for c in s})
    ycuts = [-1.0, C["skirt_y"], C["top_y"] - C["shoulder_r"], 9.0]
    for x0, x1 in zip(xcuts[:-1], xcuts[1:]):
        slab = clip_slab(poly, x0, x1)
        if len(slab) < 3:
            continue
        xm = 0.5 * (x0 + x1)
        on_cover = any(a <= xm <= b for (a, b) in C["x_splits"])
        for y0, y1 in zip(ycuts[:-1], ycuts[1:]):
            piece = clip_band(slab, y0, y1)
            if len(piece) < 3 or (on_cover and y0 >= C["top_y"] - C["shoulder_r"]):
                continue                                 # nothing paints the cover shoulders
            z = body_side_z(B, xm, 0.5 * (y0 + y1)) + off
            for side in (1, -1):
                part.add([(x, y, side * z) for (x, y) in piece], [tuple(range(len(piece)))], mat,
                         closed=False, outward=(0.0, 0.0, side))


def build_body(spec, part):
    B = spec["body"]
    hw, yb = B["halfwidth"], B["bottom_y"]

    def block(profile, mat, face_mats):
        """Side profile extruded across the body; face_mats overrides the faces of given profile edges.
        The tunnel roof (every edge along bottom_y) is the hull's black PE."""
        poly = [tuple(p) for p in profile]
        v, f = extrude_xy(poly, -hw, hw)
        mats = []
        for i, (p, q) in enumerate(zip(poly, poly[1:] + poly[:1])):
            mats.append("pontoon_pe_black" if abs(p[1] - yb) + abs(q[1] - yb) < 1e-9 else face_mats.get(i, mat))
        part.add_multi(v, f, mats + [mat, mat], buoyant=True)
    block(B["head_profile"], "livery_orange", {i: "body_white" for i in B["head_white_edges"]})
    mb = B["midbody"]
    v, f, M = box_between((mb["x_range"][0], yb, -hw), (mb["x_range"][1], mb["top_y"], hw))
    part.add_multi(v, f, ["body_white", "body_white", "pontoon_pe_black", "body_white", "body_white", "body_white"], M,
                   buoyant=True)
    block(B["stern_profile"], "livery_orange", {})
    sect = cover_section(B)
    for (xa, xb) in B["covers"]["x_splits"]:
        rings = [[(x, y, z) for (z, y) in sect] for x in (xa, xb)]
        part.add(*loft(rings), "hatch_white", buoyant=True)
    bs = B["beam_strip"]
    v, f, M = box_between((bs["x_range"][0], mb["top_y"] - 0.01, -hw + 0.02), (bs["x_range"][1], bs["top_y"], hw - 0.02))
    part.add(v, f, "beam_grey", M)
    sp = B["stern_side_panels"]
    poly = [tuple(p) for p in sp["profile_xy"]]
    for side in (1, -1):
        za, zb = sorted((side * (hw - 0.01), side * (hw + sp["proud"])))
        part.add(*extrude_xy(poly, za, zb), "panel_orange")


# ---------------------------------------------------------------- deck fittings, livery
def cleat(part, x, y, z, length):
    v, f, M = box_between((x - 0.06, y - 0.004, z - 0.025), (x + 0.06, y + 0.008, z + 0.025))
    part.add(v, f, "cleat_alu", M)
    for dx in (-0.04, 0.04):
        v, f, M = box_between((x + dx - 0.014, y, z - 0.012), (x + dx + 0.014, y + 0.04, z + 0.012))
        part.add(v, f, "cleat_alu", M)
    hl = 0.5 * length
    prof = [(x - hl, y + 0.043), (x - hl + 0.03, y + 0.036), (x + hl - 0.03, y + 0.036), (x + hl, y + 0.043),
            (x + hl - 0.03, y + 0.058), (x - hl + 0.03, y + 0.058)]
    part.add(*extrude_xy(prof, z - 0.011, z + 0.011), "cleat_alu")


def build_deck_fittings(spec, part):
    B, D, LV, SE = spec["body"], spec["deck"], spec["livery"], spec["sensors"]
    P = spec["pontoons"]
    c = D["cleats"]
    for side in (1, -1):
        cleat(part, c["x"], P["deck_y"], side * c["z"], c["length"])
    le = D["lifting_eye"]
    pin, cl = le["pin"], le["clamp"]
    v, f = cylinder(pin["radius"], pin["x_range"][0], pin["x_range"][1], 20)
    part.add(v, f, "fitting_grey", T(0.0, pin["y"], 0.0) @ Z_TO_X())
    v, f, M = box_between((cl["x_range"][0], cl["y_range"][0], -cl["halfwidth"]), (cl["x_range"][1], cl["y_range"][1], cl["halfwidth"]))
    part.add(v, f, "gantry_black", M)
    hw, w = 0.5 * le["thickness"], 0.5 * le["arm_width"]
    arm = [np.asarray(p, float) for p in le["arm_xy"]]
    for p0, p1 in zip(arm[:-1], arm[1:]):             # the bent flat bar, one straight piece per leg
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]]) * w
        e = d * w                                     # run each piece on past the bend
        quad = [tuple(p0 - e - n), tuple(p1 + e - n), tuple(p1 + e + n), tuple(p0 - e + n)]
        part.add(*extrude_xy(quad, -hw, hw), "fitting_grey")
    ex, ey = le["eye_centre_xy"]
    circ = [(ex + le["eye_r"] * math.cos(2 * math.pi * k / 24), ey + le["eye_r"] * math.sin(2 * math.pi * k / 24)) for k in range(24)]
    part.add(*extrude_xy(circ, -hw, hw), "fitting_grey")
    v, f = cylinder(le["hole_r"], -hw - 0.002, hw + 0.002, 24)
    part.add(v, f, "tunnel_dark", T(ex, ey, 0.0))
    # GNSS antennas on the head top
    y_head = max(p[1] for p in B["head_profile"])
    for name in ("gnss_port", "gnss_stbd"):
        g = SE[name]
        gx, _, gz = g["position"]
        d0, d1 = g["disc_y"]
        v, f = cylinder(g["post_radius"], y_head - 0.005, d0 + 0.005, 16)
        part.add(v, f, "gnss_white", T(gx, 0.0, gz) @ Z_TO_Y())
        R0 = g["disc_radius"]
        prof = [(0.0, d0), (R0 - 0.004, d0), (R0, d0 + 0.012), (R0, d0 + 0.02), (0.8 * R0, d0 + 0.042),
                (0.45 * R0, d1 - 0.004), (0.0, d1)]
        v, f = revolve(prof, 32)
        part.add(v, f, "gnss_white", T(gx, 0.0, gz) @ Z_TO_Y())
    # forward camera housing on the head top: dark foot, glass ring, white body
    cam = SE["camera_main"]
    h = cam["housing"]
    hx, hz = h["centre_xz"]
    y0, y1 = h["y_range"]
    r = h["radius"]
    for (ra, ya, yb, mat) in ((r + 0.003, y0 - 0.003, y0 + 0.03, "gantry_black"), (r - 0.002, y0 + 0.03, y0 + 0.05, "lens_glass"),
                              (r, y0 + 0.05, y1 - 0.01, "camera_white"), (r - 0.006, y1 - 0.011, y1, "camera_white")):
        v, f = cylinder(ra, ya, yb, 24)
        part.add(v, f, mat, T(hx, 0.0, hz) @ Z_TO_Y())
    v, f = cylinder(0.011, 0.0, 0.006, 20)
    part.add(v, f, "lens_glass", T(hx + r - 0.004, cam["position"][1], hz) @ Z_TO_X())
    # livery and the stern handles
    body_decal(part, B, [tuple(p) for p in LV["orange_polygon_xy"]], "livery_orange")
    sh = B["stern_handles"]
    (cx, cy), (ax, ay) = sh["centre_xy"], sh["half_axes"]
    z_panel = B["halfwidth"] + B["stern_side_panels"]["proud"]
    n = 32
    for side in (1, -1):
        rim = []
        for s in (1.0, 1.25):
            rim.append([(cx + s * ax * math.cos(2 * math.pi * k / n), cy + s * ay * math.sin(2 * math.pi * k / n)) for k in range(n)])
        v = [(x, y, side * (z_panel + 0.002)) for (x, y) in rim[0] + rim[1]]
        f = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
        part.add(v, f, "steel", closed=False, outward=(0.0, 0.0, side))
        v = [(x, y, side * (z_panel + 0.0025)) for (x, y) in rim[0]]
        part.add(v, [tuple(range(n))], "tunnel_dark", closed=False, outward=(0.0, 0.0, side))


# ---------------------------------------------------------------- the gantry
def build_gantry(spec, part):
    G, SE, LI = spec["gantry"], spec["sensors"], spec["lights"]
    fr, ar = G["front_rail"], G["aft_rail"]
    za, zb = G["plate_z"]
    ff, fh, af, ah = fr["foot"], fr["head"], ar["foot"], ar["head"]
    front_inner = ((ff[0] - fr["width"], ff[1]), (fh[0] - fr["width"], fh[1]))
    aft_inner = ((af[0] + ar["width"], af[1]), (ah[0] + ar["width"], ah[1]))
    pieces = [[tuple(ff), tuple(fh), front_inner[1], front_inner[0]],
              [tuple(af), aft_inner[0], aft_inner[1], tuple(ah)]]
    for (ya, yb) in G["rungs_y"]:
        pieces.append([(lerp2(*aft_inner, ya) - 0.02, ya), (lerp2(*front_inner, ya) + 0.02, ya),
                       (lerp2(*front_inner, yb) + 0.02, yb), (lerp2(*aft_inner, yb) - 0.02, yb)])
    for side in (1, -1):
        z0, z1 = sorted((side * za, side * zb))
        for poly in pieces:
            part.add(*extrude_xy(poly, z0, z1), "gantry_black")
    tp = G["top_plate"]
    (px0, px1), (py0, py1) = tp["x_range"], tp["y_range"]
    v, f, M = box_between((px0, py0, -tp["halfwidth"]), (px1, py1, tp["halfwidth"]))
    part.add(v, f, "gantry_black", M)
    # radome: pedestal, blue band, dome
    rd = G["radome"]
    rx, rz = rd["centre_xz"]
    pr, p0, p1 = rd["pedestal"]
    b0, b1 = rd["band_y"]
    rr, top = rd["radius"], rd["top_y"]
    at = T(rx, 0.0, rz) @ Z_TO_Y()
    v, f = cylinder(pr, p0 - 0.002, p1 + 0.002, 32)
    part.add(v, f, "radome_white", at)
    v, f = cylinder(rr, b0, b1, 64)
    part.add(v, f, "radome_band_blue", at)
    v, f = revolve([(rr, b1 - 0.001), (rr, top - 0.065), (rr - 0.01, top - 0.035), (rr - 0.045, top - 0.015),
                    (rr - 0.15, top - 0.003), (0.0, top)], 64)
    part.add(v, f, "radome_white", at)
    # masthead: pole, three lanterns, cap, cage
    mh = G["masthead"]
    mx, mz = mh["centre_xz"]
    at = T(mx, 0.0, mz) @ Z_TO_Y()
    pr, p0, p1 = mh["pole"]
    v, f = cylinder(pr, p0 - 0.002, mh["top_y"] - 0.01, 16)
    part.add(v, f, "gantry_black", at)
    lr = mh["lantern_r"]
    for (mat, y0, y1) in mh["lanterns"]:
        v, f = cylinder(lr + 0.006, y0 - 0.008, y0, 24)
        part.add(v, f, "gantry_black", at)
        v, f = cylinder(lr, y0, y1, 24)
        part.add(v, f, mat, at)
        v, f = cylinder(lr + 0.006, y1, y1 + 0.006, 24)
        part.add(v, f, "gantry_black", at)
    v, f = cylinder(lr + 0.006, mh["top_y"] - 0.012, mh["top_y"], 24)
    part.add(v, f, "gantry_black", at)
    for k in range(4):
        a = math.radians(45.0 + 90.0 * k)
        v, f = cylinder(0.003, p1, mh["top_y"] - 0.005, 8)
        part.add(v, f, "gantry_black", T(mx + mh["cage_r"] * math.cos(a), 0.0, mz + mh["cage_r"] * math.sin(a)) @ Z_TO_Y())
    # equipment boxes under the plate, with the lit window
    for ub in G["under_boxes"]:
        (x0, x1), (y0, y1), (z0, z1) = ub["x_range"], ub["y_range"], ub["z_range"]
        v, f, M = box_between((x0, y0, z0), (x1, y1 + 0.004, z1))
        part.add(v, f, "box_grey", M)
        if "window_z" in ub:
            (wz0, wz1), (wy0, wy1) = ub["window_z"], ub["window_y"]
            v, f, M = box_between((x1 - 0.002, wy0, wz0), (x1 + 0.004, wy1, wz1))
            part.add(v, f, "floodlight_lens", M)
    sc = G["sat_compass"]
    cx, cz = sc["centre_xz"]
    sx, sy, sz = sc["size"]
    v, f, M = box_between((cx - 0.5 * sx, py1 - 0.002, cz - 0.5 * sz), (cx + 0.5 * sx, py1 + sy, cz + 0.5 * sz))
    part.add(v, f, "antenna_white", M)
    kx, ky, kz = sc["cap"]
    v, f, M = box_between((cx - 0.5 * kx, py1 + sy - 0.002, cz - 0.5 * kz), (cx + 0.5 * kx, py1 + sy + ky, cz + 0.5 * kz))
    part.add(v, f, "antenna_white", M)
    # antennas
    mi = SE["mimo_antenna"]
    bx, by, bz = mi["base"]
    r, top = mi["radius"], by + mi["length"]
    v, f = revolve([(0.0, by - 0.002), (r, by - 0.002), (r, top - 0.012), (0.7 * r, top - 0.003), (0.0, top)], 20)
    part.add(v, f, "antenna_white", T(bx, 0.0, bz) @ Z_TO_Y())
    v, f = cylinder(r + 0.003, mi["below"][0], mi["below"][1] + 0.002, 20)
    part.add(v, f, "box_grey", T(bx, 0.0, bz) @ Z_TO_Y())
    lt = SE["lte_antenna"]
    x2, y2, z2 = lt["position"]
    r2 = lt["radius"]
    v, f = revolve([(0.0, y2 - 0.002), (r2, y2 - 0.002), (0.95 * r2, y2 + 0.015), (0.5 * r2, y2 + 0.023), (0.0, y2 + lt["height"])], 24)
    part.add(v, f, "antenna_white", T(x2, 0.0, z2) @ Z_TO_Y())
    av = SE["ais_vhf_antenna"]
    wx, wy, wz = av["base"]
    side = 1.0 if wz > 0 else -1.0
    zi = side * za
    v, f, M = box_between((wx - 0.03, wy - 0.02, min(zi, wz - side * 0.015)), (wx + 0.03, wy + 0.02, max(zi, wz - side * 0.015)))
    part.add(v, f, "gantry_black", M)
    v, f = revolve([(0.0, wy - 0.03), (0.014, wy - 0.03), (0.014, wy + 0.08), (av["radius"] * 1.6, wy + 0.1),
                    (av["radius"], wy + av["length"] - 0.01), (0.0, wy + av["length"])], 10)
    part.add(v, f, "box_grey", T(wx, 0.0, wz) @ Z_TO_Y())
    # sidelights on the outer faces of the front rails
    for name, mat in (("nav_light_stbd", "nav_green"), ("nav_light_port", "nav_red")):
        lx, ly, lz = LI[name]["position"]
        s = 1.0 if lz > 0 else -1.0
        v, f, M = box_between((lx - 0.035, ly - 0.03, s * (zb - 0.002)), (lx + 0.035, ly + 0.03, s * (zb + 0.022)))
        part.add(v, f, "gantry_black", M)
        v, f, M = box_between((lx - 0.028, ly - 0.022, s * (zb + 0.021)), (lx + 0.03, ly + 0.022, s * (zb + 0.03)))
        part.add(v, f, mat, M)


# ---------------------------------------------------------------- the thrusters
def build_thruster(spec):
    """Pod (node frame: on the azimuth axis at the hull, vessel axes) and rotor (duct centre)."""
    TH = spec["propulsion"]["thrusters"]
    pod, rotor = Part(), Part()
    ox, oy = TH["axis_x"], TH["mount_y"]
    c = TH["collar"]
    v, f = cylinder(c["radius"], c["y_range"][0] - oy, c["y_range"][1] - oy, 24)
    pod.add(v, f, "pod_black", Z_TO_Y())
    d = TH["duct"]
    dx, dy = d["centre_xy"][0] - ox, d["centre_xy"][1] - oy
    at = T(dx, dy, 0.0) @ Z_TO_X()
    hl = 0.5 * d["length"]
    pod.add(*annulus(d["inner_r"], d["outer_r"], -hl, hl, 48), "pod_black", at)
    pod.add(*annulus(d["outer_r"] - 0.004, d["band_r"], -0.5 * d["band_length"], 0.5 * d["band_length"], 48), "pod_band_grey", at)
    # the strut from the collar to the duct: a short streamlined web
    web = [(-0.05, dy + d["outer_r"] - 0.01), (0.04, dy + d["outer_r"] - 0.01), (0.03, c["y_range"][0] - oy + 0.01),
           (-0.035, c["y_range"][0] - oy + 0.01)]
    pod.add(*extrude_xy(web, -0.012, 0.012), "pod_black")
    rt = TH["rotor"]
    hl = 0.5 * rt["hub_length"]
    v, f = revolve([(0.0, -hl), (0.6 * rt["hub_r"], -hl), (rt["hub_r"], -0.5 * hl), (rt["hub_r"], 0.5 * hl),
                    (0.5 * rt["hub_r"], hl), (0.0, hl + 0.006)], 24)
    rotor.add(v, f, "rotor_grey", Z_TO_X())
    span = rt["blade_tip_r"] - 0.8 * rt["hub_r"]
    for k in range(rt["blades"]):
        v, f = box(rt["blade_chord"], span, 0.005)
        M = R("X", 360.0 * k / rt["blades"]) @ T(0.0, 0.8 * rt["hub_r"] + 0.5 * span, 0.0) @ R("Y", rt["pitch_deg"])
        rotor.add(v, f, "rotor_grey", M)
    return pod, rotor, T(dx, dy, 0.0)


# ---------------------------------------------------------------- hydrostatics
def check_spec(spec):
    M = spec["mass"]
    tot = sum(b["mass"] for b in M["budget"])
    assert abs(tot - M["dry"]) < 1e-9, tot
    m, com = mass_condition(spec, "lightship")
    assert all(abs(com[i] - M["com_dry"][i]) < 0.005 for i in range(3)), com
    P = spec["pontoons"]
    assert abs(2 * P["outer_z"] - spec["principal"]["beam"]) < 1e-6
    assert abs(P["outer_z"] - spec["body"]["halfwidth"]) < 0.03, "the body should sit inside the pontoon walls"
    L = PontoonLines(spec)
    for x in L.stations():
        assert L.keel(x) < L.top(x) - 0.02, ("keel above the pontoon top", x)
    return com


def hydrostatics(spec, tris):
    HS = spec["hydrostatics"]
    hy = Hydro(tris, HS["grid"])
    assert hy.closure < 1e-6, ("buoyant meshes are not closed", hy.closure)
    out = {"schema": "threepp.usv_hydro/1", "spec": "otterx_spec.json", "rho": HS["rho"], "g": HS["g"],
           "frame": "vessel frame of the spec (X forward, Y up, Z starboard, keel line y = 0)", "conditions": {}}
    for cond in spec["mass"]["conditions"]:
        m, c = mass_condition(spec, cond)
        h, trim = hy.solve(m, c, HS["rho"])
        out["conditions"][cond] = hy.report(h, trim, m, c, HS["rho"], HS["g"])
    out["design_condition"] = spec["mass"]["design_condition"]
    out["bonjean"] = hy.bonjean(spec)
    return out


# ---------------------------------------------------------------- Blender side
def build_geometry(spec):
    """All parts in the vessel frame (numpy), plus the buoyant triangles."""
    L = PontoonLines(spec)
    pontoons, body, fit, gantry = Part(), Part(), Part(), Part()
    build_pontoons(spec, L, pontoons)
    build_body(spec, body)
    build_deck_fittings(spec, fit)
    build_gantry(spec, gantry)
    tris = np.concatenate(pontoons.tris + body.tris)
    return {"pontoons": pontoons, "body": body, "deck_fittings": fit, "gantry": gantry}, tris


def build_blender(spec, parts, hydro):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    d = hydro["conditions"][hydro["design_condition"]]
    m_dry, com = mass_condition(spec, "lightship")
    root = new_object("otterx", None, mats, None, I, props={
        "frame": "X forward, Y up, Z starboard; origin = centreline x keel line x mid-LOA",
        "spec": "otterx_spec.json", "hydro": "otterx_hydro.json",
        "mass_dry": m_dry, "com_dry": [round(c, 4) for c in com],
        "design_waterline_y": d["waterline_y_at_x0"], "design_trim_deg": d["trim_deg"]})
    new_object("pontoons", parts["pontoons"], mats, root, I, sharp_deg=35.0)
    new_object("body", parts["body"], mats, root, I, sharp_deg=35.0)
    new_object("deck_fittings", parts["deck_fittings"], mats, root, I)
    new_object("gantry", parts["gantry"], mats, root, I)

    TH = spec["propulsion"]["thrusters"]
    kz = spec["pontoons"]["keel_z"]
    for side, name in ((-1, "port"), (1, "stbd")):
        pod, rotor, M_rotor = build_thruster(spec)
        node = new_object(f"thruster_{name}", pod, mats, root, T(TH["axis_x"], TH["mount_y"], side * kz),
                          props={"axis": "+Y", "limit_deg": TH["limit_deg"],
                                 "positive": "thrust line turns toward port, the stern is pushed to port, "
                                             "the boat turns to starboard"})
        new_object(f"thruster_{name}_rotor", rotor, mats, node, M_rotor, props={"axis": "+X"})
        new_object(f"thrust_{name}", None, mats, node, M_rotor,
                   props={"thrust_axis": "+X", "bollard_thrust_n": TH["bollard_thrust_n"], "power_kw": TH["power_kw"],
                          "assumed": TH["assumed"]})

    SE = spec["sensors"]
    cam = SE["camera_main"]
    new_object("camera_main", None, mats, root, sensor_frame(cam["position"], cam["look"], cam["pitch_down_deg"]),
               props={"convention": "look -Z, up +Y", "hfov_deg": cam["hfov_deg"],
                      "resolution": cam["resolution"], "fps": cam["fps"]})
    rd = SE["radar"]
    new_object("radar", None, mats, root, sensor_frame(rd["position"], (1, 0, 0)),
               props={k: rd[k] for k in ("spin_axis", "rpm", "h_beam_deg", "v_beam_deg", "range_max")})
    for name in ("gnss_port", "gnss_stbd"):
        new_object(name, None, mats, root, T(*SE[name]["position"]), props={"role": "phase centre", "baseline": SE["gnss_baseline"]})
    new_object("sat_compass", None, mats, root, T(*SE["sat_compass"]["position"]))
    av = SE["ais_vhf_antenna"]
    new_object("ais_vhf_antenna", None, mats, root, T(*av["base"]), props={"length": av["length"]})
    mi = SE["mimo_antenna"]
    new_object("mimo_antenna", None, mats, root, T(*mi["base"]), props={"length": mi["length"]})
    new_object("lte_antenna", None, mats, root, T(*SE["lte_antenna"]["position"]))
    new_object("imu", None, mats, root, T(*SE["imu"]["position"]))
    td = SE["transducer"]
    new_object("transducer", None, mats, root, T(*td["position"]) @ frame_from_forward_up(td["look"], (1.0, 0.0, 0.0)),
               props={"convention": "look -Z, up +Y"})
    new_object("mbes_mount", None, mats, root, T(*SE["mbes_mount"]["position"]))
    LI = spec["lights"]
    for name, d_ in LI.items():
        look = d_.get("look", (1.0, 0.0, 0.0))
        props = {"convention": "look -Z, up +Y"}
        props.update({k: v for k, v in d_.items() if k in ("colour", "cone_deg")})
        new_object(name, None, mats, root, sensor_frame(d_["position"], look), props=props)
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="otterx_spec.json")
    ap.add_argument("--out", default="otterx.glb")
    ap.add_argument("--hydro-out", default="otterx_hydro.json")
    ap.add_argument("--hydro-only", action="store_true")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def path(p):
        return p if os.path.isabs(p) else os.path.join(here, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    com = check_spec(spec)
    print(f"[otterx] spec ok; dry CoM from budget = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")

    parts, tris = build_geometry(spec)
    pod, rotor, _ = build_thruster(spec)
    used = set(m for p in list(parts.values()) + [pod, rotor] for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    hydro = hydrostatics(spec, tris)
    for cond, r in hydro["conditions"].items():
        print(f"[otterx] {cond:9s} {r['mass']:6.0f} kg  draft {r['draft_moulded_x0']:.3f}  max draft {r['draft_max']:.3f}  "
              f"trim {r['trim_deg']:+.2f} deg  LCB {r['lcb']:+.3f}  GM_T {r['gm_t']:.2f}  GM_L {r['gm_l']:.1f}  "
              f"LWL {r['lwl']:.2f}  BWL {r['bwl']:.2f}  Awp {r['waterplane_area']:.2f}")
    with open(path(a.hydro_out), "w", encoding="utf-8") as fh:
        json.dump(hydro, fh, indent=1)
    print(f"[otterx] wrote {path(a.hydro_out)}")
    HS, P = spec["hydrostatics"], spec["principal"]
    d = hydro["conditions"][spec["mass"]["design_condition"]]
    assert abs(d["draft_moulded_x0"] - P["draft_brochure"]) <= P["draft_tolerance"], d["draft_moulded_x0"]
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
    print(f"[otterx] wrote {path(a.out)}  triangles={tris_n}  y range [{zmin:.3f}, {zmax:.3f}]")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

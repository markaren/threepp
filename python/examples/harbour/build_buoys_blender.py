"""Build the Norwegian floating marks and fishing floats from buoys_spec.json: buoys.glb (via
Blender) and buoys_hydro.json.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless, from this folder:
    blender --background --factory-startup --python build_buoys_blender.py -- --spec buoys_spec.json --out buoys.glb

Hydrostatics only (plain Python + numpy, no Blender; writes buoys_hydro.json):
    python build_buoys_blender.py --hydro-only

The mesh primitives and the Blender material/object helpers are the Mariner generator's
(../usv/build_mariner_blender.py); this script adds the marks. Geometry is built with numpy in
the frame X forward, Y up, Z starboard (metres) and converted to Blender's Z-up only when handed
to bpy; the glTF exporter's Y-up conversion maps it back.

One root per mark, spaced layout.spacing_x apart along X for previewing (a demo clones a root by
name and places it). Each root's origin is on the mark's axis at the lowest point of its buoyant
body. Node names (exact), for every mark <m>:
    <m>                     root empty; extras: kind, waterline_y, freeboard, mass, gm, ...
      <m>_body              float / pipe, paint zones, retroreflective bands, mooring hardware
      <m>_topmark           topmark(s) and post (spars), flag and radar reflector (garnblaase)
      <m>_light             lantern mesh, node origin at the lamp (focal plane); spars only
        <m>_lamp            empty at the lamp for a demo light; extras: colour
      <m>_number            lateral number geometry (wrapped on the body), laterals only
      <m>_mooring_eye       empty at the chain attachment
      <m>_top               empty at the topmark centre (or the top of the body) for labels

Hydrostatics: every mark is a set of coaxial solids of revolution, so the displaced volume is
integrated directly: at each height the section is a disc whose radius is the largest radius of
the buoyant solids there. The result is checked against the closed buoyant mesh's signed volume
(times the n-gon area factor). The mooring line / chain enters as a vertical load at the eye.
Gates (the build fails): every closed primitive's Newell normals sum to ~0 (Part.add), mesh
volume agrees, every mark floats upright with GM > 0 at its design condition, and the solved
freeboard is within gates.freeboard_tol of the sourced (spars) or ASSUMED design value.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "usv"))
from build_mariner_blender import (Part, R, T, Z_TO_Y, box, box_between, make_materials,  # noqa: E402
                                   new_object, revolve, signed_volume)


# ---------------------------------------------------------------- extra primitives
def torus(Rm, r, n=24, m=12):
    """Closed torus about +Z (ring in the XY plane), major radius Rm, tube radius r."""
    v, f = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        for j in range(m):
            b = 2 * math.pi * j / m
            rr = Rm + r * math.cos(b)
            v.append((rr * math.cos(a), rr * math.sin(a), r * math.sin(b)))
    for i in range(n):
        i1 = (i + 1) % n
        for j in range(m):
            j1 = (j + 1) % m
            f.append((i * m + j, i1 * m + j, i1 * m + j1, i * m + j1))
    return v, f


def solid_y(part, profile, mat, n, buoyant=False, M=None):
    """Solid of revolution about +Y from a [(r, y), ...] profile."""
    v, f = revolve(profile, n)
    part.add(v, f, mat, (M if M is not None else np.eye(4)) @ Z_TO_Y(), buoyant=buoyant)


def banded_tube(r, ys, n, colour):
    """Closed cylinder about +Y with rings at every height in ys; colour(y_mid, angle_mid) names
    each side face's material; the caps take colour at their end. Returns v, f, mats."""
    ys = sorted(set(float(y) for y in ys))
    v, f, mats = [], [], []
    for y in ys:
        for i in range(n):
            a = 2 * math.pi * i / n
            v.append((r * math.cos(a), y, -r * math.sin(a)))
    for k in range(len(ys) - 1):
        ym = 0.5 * (ys[k] + ys[k + 1])
        for i in range(n):
            j = (i + 1) % n
            f.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
            mats.append(colour(ym, 2 * math.pi * (i + 0.5) / n))
    last = (len(ys) - 1) * n
    f.append(tuple(range(n - 1, -1, -1)))
    mats.append(colour(ys[0] + 1e-6, 0.0))
    f.append(tuple(last + i for i in range(n)))
    mats.append(colour(ys[-1] - 1e-6, 0.0))
    return v, f, mats


def fix_winding_outward(v, f):
    """banded_tube's quads run with the angle as built; flip all faces if the volume is negative."""
    V = np.asarray(v, float)
    if signed_volume(V, f) < 0:
        f = [tuple(reversed(x)) for x in f]
    return f


def sphere_profile(d, y0, k=24):
    """[(r, y)] of a sphere of diameter d with its lowest point at y0, pole to pole."""
    R_ = 0.5 * d
    return [(R_ * math.sin(math.pi * i / k), y0 + R_ - R_ * math.cos(math.pi * i / k)) for i in range(k + 1)]


# ---------------------------------------------------------------- the marks
class Mark:
    """One mark: parts, buoyant profiles (coaxial), mass items, line load and empties."""

    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self.body, self.topmark, self.lantern = Part(), Part(), None
        self.profiles = []            # buoyant solids of revolution [(r, y), ...]
        self.segments = []            # their n (for the mesh volume check)
        self.masses = []              # dicts: mass, y0, y1, k
        self.line_load, self.eye_y = 0.0, 0.0
        self.lamp_y, self.light, self.top_y = None, None, 0.0
        self.number = None            # (text, placements)
        self.design_freeboard, self.freeboard_ref_y, self.freeboard_to = None, None, ""
        self.props = {}

    def buoyant(self, part, profile, mat, n, M=None):
        self.profiles.append([(float(r), float(y)) for (r, y) in profile])
        self.segments.append(n)
        solid_y(part, profile, mat, n, buoyant=True, M=M)


def add_eye(part, R_, r, cy, mat="steel_galv"):
    """Ring in the XY plane (a vertical eye, the chain shackles through it)."""
    v, f = torus(R_, r)
    part.add(v, f, mat, T(0.0, cy, 0.0))


def build_spar(name, S, spec):
    P, TM, LA = spec["spar"], spec["topmarks"], spec["lantern"]
    mk = Mark(name, "spar_buoy (bøyestake)")
    r, L, n = 0.5 * P["diameter"], P["length"], P["segments"]
    wl, fb = P["draft_nominal"], P["freeboard"]
    bh = P["retro_band_h"]

    def at(frac):
        return wl + frac * fb

    zones = [(at(a), at(b), m) for (a, b, m) in S["zones"]]
    zones[0] = (-1e9, zones[0][1], zones[0][2])
    zones[-1] = (zones[-1][0], 1e9, zones[-1][2])
    bands = [(at(c) - 0.5 * bh, at(c) + 0.5 * bh, m) for (c, m) in S["retro"]]
    st = S.get("stripes")

    def colour(y, a):
        for (y0, y1, m) in bands:
            if y0 <= y <= y1:
                return m
        for (y0, y1, m) in zones:
            if y0 <= y <= y1:
                if m == "stripes":
                    k = int(a / (2 * math.pi) * st["n"]) % len(st["mats"])
                    return st["mats"][k]
                return m
        raise ValueError((name, y))

    ys = [0.0, L] + [z for (a, b, _) in zones for z in (a, b) if 0.0 < z < L] + [z for (a, b, _) in bands for z in (a, b)]
    ys += list(np.linspace(0.0, L, 15))
    v, f, mats = banded_tube(r, ys, n, colour)
    f = fix_winding_outward(v, f)
    mk.body.add_multi(v, f, mats, buoyant=True)
    mk.profiles.append([(r, 0.0), (r, L)])
    mk.segments.append(n)
    bp = P["bottom_plate"]
    bv, bf, bM = box_between((-0.5 * bp["size"], -bp["thick"], -0.5 * bp["size"]), (0.5 * bp["size"], 0.0, 0.5 * bp["size"]))
    mk.body.add(bv, bf, "steel_galv", bM)
    e = P["eye"]
    add_eye(mk.body, e["R"], e["r"], e["centre_y"])
    mk.eye_y = e["centre_y"] - e["R"]

    # topmark post, topmark(s), lantern on top
    post = P["post"]
    base = L + post["clearance"]
    kind, tmat = S["topmark"]
    nt = TM["segments"]
    tp_ = mk.topmark
    if kind.startswith("cones") or kind == "cone_up":
        c = TM["cone_single"] if kind == "cone_up" else TM["cone"]
        rb, h, g = 0.5 * c["base_d"], c["h"], TM["gap"]
        ups = {"cone_up": [True], "cones_up_up": [True, True], "cones_down_down": [False, False],
               "cones_base_to_base": [False, True], "cones_point_to_point": [True, False]}[kind]
        y = base
        for up in ups:
            prof = [(0.0, y), (rb, y), (0.0, y + h)] if up else [(0.0, y), (rb, y + h), (0.0, y + h)]
            solid_y(tp_, prof, tmat, nt)
            y += h + g
        top = y - g
    elif kind == "cylinder":
        c = TM["cylinder"]
        solid_y(tp_, [(0.0, base), (0.5 * c["d"], base), (0.5 * c["d"], base + c["h"]), (0.0, base + c["h"])], tmat, nt)
        top = base + c["h"]
    elif kind in ("balls_two", "ball_one"):
        d = TM["ball"]["d"] if kind == "balls_two" else TM["ball_single"]["d"]
        y = base
        for _ in range(2 if kind == "balls_two" else 1):
            solid_y(tp_, sphere_profile(d, y, 16), tmat, nt)
            y += d + TM["ball_gap"]
        top = y - TM["ball_gap"]
    elif kind == "x":
        X = TM["x"]
        cy = base + 0.5 * X["bar_len"] * math.sqrt(0.5) + 0.02
        for yaw in (0.0, 90.0):
            for roll in (45.0, -45.0):
                bv, bf = box(X["bar_w"], X["bar_len"], X["bar_t"])
                tp_.add(bv, bf, tmat, T(0.0, cy, 0.0) @ R("Y", yaw) @ R("Z", roll))
        top = cy + 0.5 * X["bar_len"] * math.sqrt(0.5) + 0.02
    else:
        raise ValueError(kind)
    mk.top_y = 0.5 * (base + top)
    post_top = top + post["above_topmark"]
    solid_y(tp_, [(0.0, L - 0.02), (0.5 * post["diameter"], L - 0.02), (0.5 * post["diameter"], post_top), (0.0, post_top)],
            "steel_black", 16)
    # lantern: base housing, lens, cap; node origin at the lens centre
    lb, ll, lc = LA["base"], LA["lens"], LA["cap"]
    y0 = post_top
    lamp = y0 + lb["h"] + 0.5 * ll["h"]
    lt = Part()
    light = spec["lights"][S["light"]]
    for (d, h, m) in ((lb["d"], lb["h"], "lantern_black"), (ll["d"], ll["h"], light["lens"]), (lc["d"], lc["h"], "lantern_black")):
        solid_y(lt, [(0.0, y0 - lamp), (0.5 * d, y0 - lamp), (0.5 * d, y0 + h - lamp), (0.0, y0 + h - lamp)], m, 24)
        y0 += h
    mk.lantern, mk.lamp_y, mk.light = lt, lamp, S["light"]
    mk.masses = [{"name": "pipe", "mass": P["mass_pipe"], "y0": 0.0, "y1": L},
                 {"name": "topmark_post", "mass": P["mass_topmark"], "y0": mk.top_y, "y1": mk.top_y},
                 {"name": "lantern", "mass": LA["mass"], "y0": lamp, "y1": lamp}]
    mk.line_load = P["chain_load"]
    mk.design_freeboard, mk.freeboard_ref_y, mk.freeboard_to = fb, L, "pipe top (sourced focal plane at MSL)"
    if "number" in S:
        mk.number = (str(S["number"]), [(r, at(P["number_frac"]), 0.0, P["number_size"], a) for a in (0.0, math.pi)])
    mk.props = {"note": S["note"], "topmark": kind, "light_colour": S["light"]}
    return mk


def build_marina(name, S, spec):
    M_ = spec["marina"]
    mk = Mark(name, "lateral buoy (" + ("sylindrisk" if "can" in name else "konisk") + ")")
    n = M_["segments"]
    prof = [(float(r), float(y)) for (r, y) in S["profile"]]
    y0b, y1b, bm = S["retro"]
    # split the side into paint / band / paint by building three stacked solids is not closed-safe;
    # instead colour the faces of one revolve by height
    ys = sorted(set([y for (_, y) in prof] + [y0b, y1b]))
    prof2 = []
    for (a, b) in zip(prof[:-1], prof[1:]):
        prof2.append(a)
        if b[1] > a[1]:
            for y in ys:
                if a[1] < y < b[1]:
                    prof2.append((a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y))
    prof2.append(prof[-1])
    v, f = revolve(prof2, n)
    V = np.asarray(v, float) @ Z_TO_Y()[:3, :3].T
    mats = []
    for face in f:
        ym = float(np.mean([V[k][1] for k in face]))
        mats.append(bm if (y0b <= ym <= y1b and len(face) == 4) else S["paint"])
    mk.body.add_multi(V, f, mats, buoyant=True)
    mk.profiles.append(prof)
    mk.segments.append(n)
    pd = M_["pad"]
    bv, bf, bM = box_between((-0.5 * pd["size"], -pd["thick"], -0.5 * pd["size"]), (0.5 * pd["size"], 0.0, 0.5 * pd["size"]))
    mk.body.add(bv, bf, "steel_galv", bM)
    e = M_["eye"]
    add_eye(mk.body, e["R"], e["r"], e["centre_y"])
    mk.eye_y = e["centre_y"] - e["R"]
    mk.top_y = S["top_y"]
    mk.masses = S["masses"]
    mk.line_load = S["line_load"]
    mk.design_freeboard, mk.freeboard_ref_y, mk.freeboard_to = S["design_freeboard"], S["top_y"], S["freeboard_to"]
    ny = S["number_y"]
    # radius and slope of the body at the number's height
    rr, slope = None, 0.0
    for (a, b) in zip(prof[:-1], prof[1:]):
        if b[1] > a[1] and a[1] <= ny <= b[1]:
            rr = a[0] + (b[0] - a[0]) * (ny - a[1]) / (b[1] - a[1])
            slope = (b[0] - a[0]) / (b[1] - a[1])
    mk.number = (str(S["number"]), [(rr, ny, slope, S["number_size"], a) for a in (0.0, math.pi)])
    mk.props = {"line_note": S["line_note"]}
    return mk


def build_mooring(name, S, spec):
    mk = Mark(name, "mooring buoy")
    d = S["diameter"]
    mk.buoyant(mk.body, sphere_profile(d, 0.0, 24), S["paint"], 40)
    # retro band: a thin proud ring (open to the sphere it hugs, but a closed torus-like tube is simpler)
    y0, y1, bm = S["band"]
    Rs = 0.5 * d
    ra = math.sqrt(max(Rs * Rs - (y0 - Rs) ** 2, 0.0)) + 0.004
    rb = math.sqrt(max(Rs * Rs - (y1 - Rs) ** 2, 0.0)) + 0.004
    solid_y(mk.body, [(0.0, y0), (ra, y0), (rb, y1), (0.0, y1)], bm, 40)
    solid_y(mk.body, [(0.0, S["rod_bottom"]), (0.5 * S["rod_d"], S["rod_bottom"]), (0.5 * S["rod_d"], S["rod_top"]),
                      (0.0, S["rod_top"])], "steel_galv", 16)
    t = S["top_ring"]
    add_eye(mk.body, t["R"], t["r"], t["centre_y"])
    e = S["eye"]
    add_eye(mk.body, e["R"], e["r"], e["centre_y"])
    mk.eye_y = e["centre_y"] - e["R"]
    mk.top_y = t["centre_y"]
    mk.masses = S["masses"]
    mk.line_load = S["line_load"]
    mk.design_freeboard, mk.freeboard_ref_y, mk.freeboard_to = S["design_freeboard"], d, S["freeboard_to"]
    mk.props = {"line_note": S["line_note"], "pickup_ring_y": t["centre_y"]}
    return mk


def build_float(name, S, spec):
    F = spec["floats"]
    mk = Mark(name, "fishing float (blåse)")
    d, n = S["diameter"], F["segments"]
    mk.buoyant(mk.body, sphere_profile(d, 0.0, 28), F["paint"], n)
    # neck and eye hang below (eye down: the line pulls it under)
    e = S["eye"]
    eye_c = -(S["overall"] - d) + e["R"] + e["r"]
    neck_bot = eye_c + 0.3 * e["R"]
    nr = 0.5 * S["neck_d"]
    solid_y(mk.body, [(0.0, neck_bot), (0.6 * nr, neck_bot), (nr, 0.5 * (neck_bot + 0.0)), (nr, 0.03), (0.0, 0.03)],
            F["paint"], n)
    add_eye(mk.body, e["R"], e["r"], eye_c, F["paint"])
    mk.eye_y = eye_c - e["R"]
    mk.top_y = d
    mk.masses = [{"name": "float", "mass": S["mass"], "y0": 0.5 * d, "y1": 0.5 * d, "k": math.sqrt(2.0 / 3.0) * 0.5 * d}]
    mk.line_load = S["line_load"]
    mk.design_freeboard, mk.freeboard_ref_y, mk.freeboard_to = S["design_freeboard"], d, F["freeboard_to"]
    mk.props = {"buoyancy_note": S["buoyancy_note"], "overall_with_eye": S["overall"]}
    return mk


def build_garn(name, S, spec):
    mk = Mark(name, "gillnet end marker (garnblåse)")
    n = S["segments"]
    b = S["ballast"]
    mk.buoyant(mk.body, [(0.0, 0.0), (0.5 * b["d"], 0.0), (0.5 * b["d"], b["h"]), (0.0, b["h"])], "ballast_lead", n)
    rp = 0.5 * S["pole_d"]
    mk.buoyant(mk.topmark, [(0.0, 0.02), (rp, 0.02), (rp, S["pole_top"]), (0.0, S["pole_top"])], "pole_fibreglass", 16)
    fd = S["float_d"]
    mk.buoyant(mk.body, sphere_profile(fd, S["float_centre_y"] - 0.5 * fd, 24), "float_orange", n)
    e = S["eye"]
    add_eye(mk.body, e["R"], e["r"], e["centre_y"])
    mk.eye_y = e["centre_y"] - e["R"]
    rf = S["reflector"]
    p, t = rf["plate"], rf["t"]
    for dims in ((p, p, t), (p, t, p), (t, p, p)):
        bv, bf = box(*dims)
        mk.topmark.add(bv, bf, "reflector_alu", T(0.0, rf["centre_y"], 0.0) @ R("Y", 45.0))
    fl = S["flag"]
    bv, bf = box(fl["w"], fl["h"], fl["t"])
    mk.topmark.add(bv, bf, "flag_black", T(rp + 0.5 * fl["w"], fl["top_y"] - 0.5 * fl["h"], 0.0))
    mk.top_y = fl["top_y"] - 0.5 * fl["h"]
    mk.masses = S["masses"]
    mk.line_load = S["line_load"]
    mk.design_freeboard, mk.freeboard_ref_y, mk.freeboard_to = S["design_freeboard"], S["pole_top"], S["freeboard_to"]
    mk.props = {"line_note": S["line_note"], "note": S["note"]}
    return mk


def build_marks(spec):
    marks = []
    for name in spec["layout"]["order"]:
        if name in spec["spars"]:
            marks.append(build_spar(name, spec["spars"][name], spec))
        elif name in ("lateral_port_can", "lateral_stbd_cone"):
            marks.append(build_marina(name, spec["marina"][name], spec))
        elif name == "mooring_buoy":
            marks.append(build_mooring(name, spec["mooring_buoy"], spec))
        elif name in ("blaase_a3", "blaase_a5"):
            marks.append(build_float(name, spec["floats"][name], spec))
        elif name == "garnblaase":
            marks.append(build_garn(name, spec["garnblaase"], spec))
        else:
            raise KeyError(name)
    return marks


# ---------------------------------------------------------------- hydrostatics (axisymmetric)
def r_at(profile, y):
    best = 0.0
    for (a, b) in zip(profile[:-1], profile[1:]):
        lo, hi = (a, b) if a[1] <= b[1] else (b, a)
        if hi[1] > lo[1] and lo[1] <= y <= hi[1]:
            best = max(best, lo[0] + (hi[0] - lo[0]) * (y - lo[1]) / (hi[1] - lo[1]))
    return best


def section_radius(mk, ys):
    return np.array([max(r_at(p, y) for p in mk.profiles) for y in ys])


def mass_props(mk):
    """Structural mass, its centre y and inertia about its centre (slender items + k)."""
    m = sum(i["mass"] for i in mk.masses)
    yg = sum(i["mass"] * 0.5 * (i["y0"] + i["y1"]) for i in mk.masses) / m
    I = 0.0
    for i in mk.masses:
        L = abs(i["y1"] - i["y0"])
        yc = 0.5 * (i["y0"] + i["y1"])
        I += i["mass"] * (L * L / 12.0 + (yc - yg) ** 2 + i.get("k", 0.0) ** 2)
    return m, yg, I


def hydro_mark(mk, spec):
    W, H = spec["water"], spec["hydro"]
    rho, g, dy = W["rho"], W["g"], H["dy"]
    ylo = min(y for p in mk.profiles for (_, y) in p)
    yhi = max(y for p in mk.profiles for (_, y) in p)
    ys = np.arange(ylo + 0.5 * dy, yhi, dy)
    rr = section_radius(mk, ys)
    A = math.pi * rr * rr
    cumV = np.concatenate([[0.0], np.cumsum(A) * dy])
    cumM = np.concatenate([[0.0], np.cumsum(A * ys) * dy])
    yedge = np.concatenate([[ylo], ys + 0.5 * dy])

    def V_at(h):
        return float(np.interp(h, yedge, cumV)), float(np.interp(h, yedge, cumM))

    m_s, yg_s, I_s = mass_props(mk)
    W_tot = m_s + mk.line_load                          # kg of vertical load
    yG = (m_s * yg_s + mk.line_load * mk.eye_y) / W_tot
    Vneed = W_tot / rho
    assert Vneed < cumV[-1], (mk.name, "sinks", Vneed, cumV[-1])
    lo, hi = ylo, yhi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if V_at(mid)[0] < Vneed:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    V, Mv = V_at(h)
    yB = Mv / V
    r_wl = float(section_radius(mk, [h - 1e-6])[0])
    Awp = math.pi * r_wl ** 2
    BM = (math.pi * r_wl ** 4 / 4.0) / V
    GM = yB + BM - yG
    a33 = H["heave_added_mass_coeff"] * rho * (2.0 / 3.0) * math.pi * r_wl ** 3
    T_heave = 2 * math.pi * math.sqrt((m_s + a33) / (rho * g * Awp)) if Awp > 0 else float("inf")
    sub = ys < h
    I_roll = I_s + m_s * (yg_s - yG) ** 2 + float((rho * A[sub] * (ys[sub] - yG) ** 2).sum() * dy)
    T_roll = 2 * math.pi * math.sqrt(I_roll / (rho * g * V * GM)) if GM > 0 else float("inf")
    rows = []
    for yw in np.linspace(ylo, yhi, H["table_rows"]):
        v_, m_ = V_at(yw)
        rw = float(section_radius(mk, [min(max(yw - 1e-6, ylo), yhi)])[0])
        rows.append([round(float(yw), 4), round(v_, 6), round(m_ / v_ if v_ > 0 else float(yw), 4), round(math.pi * rw * rw, 5)])
    # closed buoyant mesh vs exact volume (per n-gon factor, all solids share one mesh union only
    # when they do not overlap; the solids here are disjoint or nested by construction)
    exact = sum(np.pi * float((section_radius(_Solo(p), ys) ** 2).sum()) * dy * (n / (2 * math.pi)) * math.sin(2 * math.pi / n)
                for p, n in zip(mk.profiles, mk.segments))
    tris = np.concatenate([t for part in (mk.body, mk.topmark) for t in part.tris])
    mesh = float(sum(np.dot(a, np.cross(b, c)) for (a, b, c) in tris) / 6.0)
    freeboard = mk.freeboard_ref_y - h
    out = {
        "kind": mk.kind,
        "mass": round(m_s, 3), "cog_y": round(yg_s, 4),
        "line_load_at_eye": mk.line_load, "eye_y": round(mk.eye_y, 4),
        "design": {
            "total_vertical_load_kg": round(W_tot, 3), "g_y_with_line": round(yG, 4),
            "waterline_y": round(h, 4), "displaced_volume": round(V, 6), "kb_y": round(yB, 4),
            "waterplane_area": round(Awp, 5), "waterplane_radius": round(r_wl, 4), "bm": round(BM, 4), "gm": round(GM, 4),
            "freeboard": round(freeboard, 4), "freeboard_to": mk.freeboard_to,
            "design_freeboard": mk.design_freeboard,
            "heave_period_s": round(T_heave, 3), "heave_added_mass": round(a33, 3),
            "roll_period_s": round(T_roll, 3), "roll_inertia_incl_added": round(I_roll, 3),
        },
        "draft_table": {"columns": ["waterline_y", "displaced_volume", "centroid_y", "waterplane_area"], "rows": rows},
        "mesh_volume_check": {"mesh": round(mesh, 6), "exact_ngon": round(exact, 6)},
        "body_y_range": [round(ylo, 4), round(yhi, 4)],
    }
    if mk.lamp_y is not None:
        out["design"]["focal_plane"] = round(mk.lamp_y - h, 4)
    return out


class _Solo:
    def __init__(self, p):
        self.profiles = [p]


def hydrostatics(spec, marks):
    G = spec["gates"]
    res = {"schema": "threepp.buoy_hydro/1",
           "frame": "X forward, Y up, Z starboard; each mark's origin on its axis at the lowest point of its buoyant body",
           "method": spec["hydro"]["method"] + "; " + spec["hydro"]["note"],
           "rho": spec["water"]["rho"], "g": spec["water"]["g"], "marks": {}}
    for mk in marks:
        r = hydro_mark(mk, spec)
        d = r["design"]
        mv = r["mesh_volume_check"]
        rel = abs(mv["mesh"] - mv["exact_ngon"]) / mv["exact_ngon"]
        print(f"[buoys] {mk.name:18s} mass {r['mass']:6.1f} + line {mk.line_load:5.1f} kg  wl {d['waterline_y']:.3f}  "
              f"freeboard {d['freeboard']:.3f} (design {d['design_freeboard']:.2f})  GM {d['gm']:.3f}  "
              f"T_heave {d['heave_period_s']:.2f} s  T_roll {d['roll_period_s']:.2f} s  mesh vol err {rel:.1e}")
        assert rel <= G["mesh_volume_rel"], (mk.name, "mesh volume", mv)
        assert d["gm"] > G["gm_min"], (mk.name, "GM", d["gm"])
        assert abs(d["freeboard"] - d["design_freeboard"]) <= G["freeboard_tol"], (mk.name, "freeboard", d["freeboard"])
        res["marks"][mk.name] = r
    return res


# ---------------------------------------------------------------- Blender side
def refine(V, F, maxlen):
    """Split triangles at the midpoint of their longest edge until no edge exceeds maxlen, so a
    glyph bent onto a round body follows it (a long flat triangle would cut into the pipe)."""
    V = [tuple(p) for p in V]
    mid = {}

    def midpoint(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = len(V)
            V.append(tuple(0.5 * (np.asarray(V[a]) + np.asarray(V[b]))))
        return mid[k]
    out, todo = [], [tuple(f) for f in F]
    while todo:
        a, b, c = todo.pop()
        e = [(np.linalg.norm(np.subtract(V[a], V[b])), 0), (np.linalg.norm(np.subtract(V[b], V[c])), 1),
             (np.linalg.norm(np.subtract(V[c], V[a])), 2)]
        L, k = max(e)
        if L <= maxlen:
            out.append((a, b, c))
            continue
        if k == 0:
            m = midpoint(a, b)
            todo += [(a, m, c), (m, b, c)]
        elif k == 1:
            m = midpoint(b, c)
            todo += [(a, b, m), (a, m, c)]
        else:
            m = midpoint(c, a)
            todo += [(a, b, m), (m, b, c)]
    return np.asarray(V, float), out


def number_part(text, placements, bl_text_mesh, maxlen=0.01, lift=0.004):
    """Blender font text (triangulated flat) bent onto the body: placements =
    [(r, y, dr/dy, size, angle)]. Glyph x maps to the angle about the axis, glyph y runs up the
    body's generator, and every vertex sits `lift` outside the surface."""
    verts, faces = bl_text_mesh(text)
    part = Part()
    for (r0, yc, slope, size, a0) in placements:
        s = size / max(1e-9, (verts[:, 1].max() - verts[:, 1].min()))
        uvw = np.stack([(verts[:, 0] - 0.5 * (verts[:, 0].max() + verts[:, 0].min())) * s,
                        (verts[:, 1] - 0.5 * (verts[:, 1].max() + verts[:, 1].min())) * s,
                        verts[:, 2] * s], axis=1)
        Vr, Fr = refine(uvw, faces, maxlen)
        u, v, w = Vr[:, 0], Vr[:, 1], Vr[:, 2]
        tl = math.hypot(slope, 1.0)
        tr, ty = slope / tl, 1.0 / tl                  # up the generator
        nr, ny = ty, -tr                                # outward normal in the (r, y) plane
        rad = r0 + v * tr + (w + lift) * nr
        yy = yc + v * ty + (w + lift) * ny
        phi = a0 + u / (r0 + v * tr)
        P = np.stack([rad * np.cos(phi), yy, -rad * np.sin(phi)], axis=1)
        part.add(P, Fr, "number_white", closed=False)
    return part


def make_text_mesh():
    import bmesh
    import bpy

    def fn(text):
        cu = bpy.data.curves.new("num_" + text, "FONT")
        cu.body = text
        cu.size = 1.0
        cu.extrude = 0.0
        cu.align_x, cu.align_y = "CENTER", "CENTER"
        ob = bpy.data.objects.new("num_tmp", cu)
        bpy.context.scene.collection.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        bm = bmesh.new()                      # triangulate while flat: a bent n-gon mis-triangulates
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
        V = np.array([tuple(v.co) for v in me.vertices], float)
        F = [tuple(p.vertices) for p in me.polygons]
        bpy.data.objects.remove(ob)
        # the font mesh faces +Z (toward the reader); keep that as the outward w axis
        n = sum(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])[2] for f in F if len(f) >= 3)
        if n < 0:
            F = [tuple(reversed(f)) for f in F]
        return V, F
    return fn


def build_blender(spec, marks, hydro):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    dx = spec["layout"]["spacing_x"]
    text_mesh = make_text_mesh()
    for i, mk in enumerate(marks):
        h = hydro["marks"][mk.name]
        d = h["design"]
        props = {"kind": mk.kind, "spec": "buoys_spec.json", "hydro": "buoys_hydro.json",
                 "waterline_y": d["waterline_y"], "freeboard": d["freeboard"], "mass": h["mass"],
                 "line_load_at_eye": mk.line_load, "gm": d["gm"], "heave_period_s": d["heave_period_s"],
                 "frame": "origin on the axis at the lowest point of the buoyant body; Y up"}
        props.update({k: v for k, v in mk.props.items() if isinstance(v, (str, int, float))})
        root = new_object(mk.name, None, mats, None, T(i * dx, 0.0, 0.0), props=props)
        new_object(mk.name + "_body", mk.body, mats, root, np.eye(4), sharp_deg=40.0)
        if mk.topmark.faces:
            new_object(mk.name + "_topmark", mk.topmark, mats, root, np.eye(4), sharp_deg=40.0)
        if mk.lantern is not None:
            L = spec["lights"][mk.light]
            lo = new_object(mk.name + "_light", mk.lantern, mats, root, T(0.0, mk.lamp_y, 0.0),
                            props={"colour": mk.light, "rgb": L["colour"], "focal_plane_above_water": d.get("focal_plane", 0.0)})
            new_object(mk.name + "_lamp", None, mats, lo, np.eye(4), props={"colour": mk.light, "rgb": L["colour"]})
        if mk.number is not None:
            text, pl = mk.number
            new_object(mk.name + "_number", number_part(text, pl, text_mesh), mats, root, np.eye(4),
                       props={"number": text}, sharp_deg=30.0)
        new_object(mk.name + "_mooring_eye", None, mats, root, T(0.0, mk.eye_y, 0.0),
                   props={"line_load_kg": mk.line_load})
        new_object(mk.name + "_top", None, mats, root, T(0.0, mk.top_y, 0.0))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="buoys_spec.json")
    ap.add_argument("--out", default="buoys.glb")
    ap.add_argument("--hydro-out", default="buoys_hydro.json")
    ap.add_argument("--hydro-only", action="store_true")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def path(p):
        return p if os.path.isabs(p) else os.path.join(here, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    marks = build_marks(spec)            # every closed primitive passed the Newell gate in Part.add
    used = set(m for mk in marks for p in (mk.body, mk.topmark, mk.lantern or Part()) for m in p.mats)
    used.add("number_white")
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    hydro = hydrostatics(spec, marks)
    with open(path(a.hydro_out), "w", encoding="utf-8") as fh:
        json.dump(hydro, fh, indent=1)
    print(f"[buoys] wrote {a.hydro_out}  ({len(marks)} marks, all gates pass)")
    if a.hydro_only:
        return
    import bpy
    build_blender(spec, marks, hydro)
    bpy.context.view_layer.update()
    tris_n = sum(sum(len(p.vertices) - 2 for p in ob.data.polygons)
                 for ob in bpy.context.scene.objects if ob.type == "MESH")
    bpy.ops.export_scene.gltf(filepath=path(a.out), export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True)
    print(f"[buoys] wrote {a.out}  triangles={tris_n}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

"""Build the Skywalker X8 from x8_spec.json: x8.glb (via Blender).

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_x8_blender.py -- --spec x8_spec.json --out x8.glb

Geometry only (plain Python + numpy, no Blender; prints the checks):
    python build_x8_blender.py --check

The mesh primitives and the Blender export are the Mariner generator's
(../usv/build_mariner_blender.py); this script adds the flying wing. All geometry is
built with numpy in the model frame (X forward, Y up, Z right, metres, origin at the
CG) and converted to Blender's Z-up only when a mesh or a transform is handed to bpy.
The glTF exporter's Y-up conversion maps it straight back, so every number in the
spec is the number in the .glb.

The wing is one closed loft from the port winglet's top through the centre body to
the starboard winglet's top: a section at each station along the spar line, which
runs flat out to the bend and turns up through it into the winglet. A section is an
airfoil (the NACA four-digit thickness form with the open trailing edge, a reflexed
camber line on the wing) of the station's chord, thickness and upper share; where an
elevon is, the section stops at the hinge and the elevon is the rest of it, lofted on
its own node.

Node hierarchy (all names exact):
    x8                          root, identity, origin = CG (the article's body origin)
      airframe                  wing, centre body, winglets
      hatches                   the three hatch covers
      fittings                  pitot tube, GNSS antenna, camera windows, motor fairing and can
      elevon_left, elevon_right on the hinge lines: local +Z along the hinge toward starboard,
                                local +Y up; a positive turn about local +Z puts the
                                trailing edge down
      propeller                 at the disc centre, spins about local +X (blades, hub, spinner)
      imu, gnss, pitot_tip, camera_nadir, camera_fpv        sensor empties

Sensor empties look down local -Z with local +Y up (threepp camera convention).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

TRAPZ = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed it; Blender's may be older
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "usv"))
from build_mariner_blender import (Part, R, T, Z_TO_X, Z_TO_Y, box_between, cylinder, loft,  # noqa: E402
                                   make_materials, new_object, pchip, revolve, sensor_frame)


# ---------------------------------------------------------------- the airfoil
def thickness(x):
    """NACA four-digit half-thickness per unit thickness ratio, open trailing edge (0.0105 t at x = 1)."""
    return 5.0 * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def camber(x, m):
    """Reflexed camber line: positive ahead, slightly negative over the last 23 % of the chord."""
    return m * math.sin(math.pi * x) * (1.0 - 1.3 * x)


def chord_points(x0, x1, n):
    """n + 1 chord fractions from x0 to x1, closer together toward the leading edge."""
    return [x0 + (x1 - x0) * (1.0 - math.cos(0.5 * math.pi * i / n)) for i in range(n + 1)]


class Wing:
    """The planform and sections of x8_spec.json's geometry block, as functions of the
    distance l outboard of the centreline (flat part) and the spar line's arc length beyond it."""

    N_CHORD = 24                      # chordwise intervals per surface

    def __init__(self, spec):
        G = spec["geometry"]
        self.G = G
        self.x_nose = G["cg_from_nose"]
        le, te = np.asarray(G["leading_edge"]), np.asarray(G["trailing_edge"])
        th, us = np.asarray(G["thickness"]), np.asarray(G["upper_share"])
        self._le, self._te = pchip(le[:, 0], le[:, 1]), pchip(te[:, 0], te[:, 1])
        self._t, self._u = pchip(th[:, 0], th[:, 1]), pchip(us[:, 0], us[:, 1])
        self.m = G["camber"]
        W = G["winglet"]
        self.W = W
        self.lb, self.rb = W["bend_l"], W["bend_r"]
        self.cant = math.radians(W["cant_deg"])
        E = G["elevons"]
        self.e_l0, self.e_l1 = E["l_range"]
        self.e_c0, self.e_c1 = E["chord"]
        self.gap = E["gap"]

    # ---- the flat part
    def s_le(self, l):
        return self._le(min(l, self.lb))

    def s_te(self, l):
        return self._te(min(l, self.lb))

    def hinge_s(self, l):
        """s of the straight hinge line at l (between the elevon's ends, extended)."""
        s0 = self.s_te(self.e_l0) - self.e_c0
        s1 = self.s_te(self.e_l1) - self.e_c1
        return s0 + (s1 - s0) * (l - self.e_l0) / (self.e_l1 - self.e_l0)

    def flat(self, l, cut=False):
        """Section parameters at l on the flat part: s of the LE, chord, absolute thickness,
        upper share, camber, the chord fraction it stops at."""
        sl, c = self.s_le(l), self.s_te(l) - self.s_le(l)
        m = self.m * min(1.0, max(0.0, (l - 0.10) / 0.08))
        x_cut = 1.0
        if cut:
            x_cut = (self.hinge_s(l) - sl) / c
        return {"s_le": sl, "c": c, "t": self._t(l), "k": self._u(l), "m": m, "x_cut": x_cut}

    # ---- the stations
    def stations(self):
        """(spar point (lateral, height), cant angle, section) from the centreline to the
        starboard winglet's top."""
        G = self.G
        l0c, l1c = self.e_l0 - self.gap, self.e_l1 + self.gap      # where the wing gives way to the elevon
        ls = set(np.round(np.arange(0.0, 0.20, 0.02), 6).tolist())
        ls |= set(np.round(np.arange(0.20, self.lb, 0.04), 6).tolist())
        ls = sorted(x for x in ls if not (l0c - 0.004 <= x <= l1c + 0.004))
        cut_ls = np.linspace(l0c, l1c, 12).tolist()
        ls = sorted(set(ls + [l0c - 0.003, l1c + 0.003, self.lb]))
        out = []
        for l in sorted(set(ls) | set(cut_ls)):
            out.append(((l, 0.0), 0.0, self.flat(l, cut=l0c - 1e-9 <= l <= l1c + 1e-9), "wing"))
        # the bend: the tip section carried round the arc, thinning toward the winglet's
        tip = self.flat(self.lb)
        W = self.W
        nb = 6
        for k in range(1, nb + 1):
            a = self.cant * k / nb
            f = k / nb
            sec = dict(tip)
            sec["t"] = tip["t"] + (W["thickness"] - tip["t"]) * f
            sec["k"] = tip["k"] + (0.5 - tip["k"]) * f
            sec["m"] = tip["m"] * (1.0 - f)
            out.append(((self.lb + self.rb * math.sin(a), self.rb * (1.0 - math.cos(a))), a, sec,
                        "winglet" if f > 0.49 else "wing"))
        # the winglet: straight, swept back, tapering
        base = ((self.lb + self.rb * math.sin(self.cant), self.rb * (1.0 - math.cos(self.cant))))
        h_top = W["top_height"]
        rise = h_top - base[1]
        nw = 5
        for k in range(1, nw + 1):
            f = k / nw
            hh = rise * f
            c = tip["c"] + (W["top_chord"] - tip["c"]) * f
            sec = {"s_le": tip["s_le"] + math.tan(math.radians(W["le_sweep_deg"])) * hh, "c": c, "t": W["thickness"],
                   "k": 0.5, "m": 0.0, "x_cut": 1.0}
            out.append(((base[0] + hh * math.cos(self.cant), base[1] + hh * math.sin(self.cant)), self.cant, sec, "winglet"))
        return out

    def ring(self, station, side):
        """The section's loop, model frame: the upper surface from the cut (or the trailing
        edge) to the leading edge, the lower back to the cut; the hinge face closes it."""
        (lat, hgt), a, sec, _ = station
        n = self.N_CHORD
        xs = chord_points(0.0, sec["x_cut"], n)
        t, c, k, m = sec["t"], sec["c"], sec["k"], sec["m"]
        tr = t / c
        sn, cs = math.sin(a), math.cos(a)
        pts = []
        for x, up in [(x, True) for x in reversed(xs)] + [(x, False) for x in xs[1:]]:
            yt = thickness(x) * tr * c
            y = camber(x, m) * c + (2.0 * k * yt if up else -2.0 * (1.0 - k) * yt)
            X = self.x_nose - (sec["s_le"] + x * c)
            pts.append((X, hgt + cs * y, side * (lat - sn * y)))
        return pts

    def surface_y(self, l, s, upper=True):
        """Height of the flat part's upper (or lower) surface at (l, s)."""
        sec = self.flat(l)
        x = (s - sec["s_le"]) / sec["c"]
        x = min(max(x, 0.0), 1.0)
        yt = thickness(x) * sec["t"]
        return camber(x, sec["m"]) * sec["c"] + (2.0 * sec["k"] * yt if upper else -2.0 * (1.0 - sec["k"]) * yt)

    def X(self, s):
        return self.x_nose - s


# ---------------------------------------------------------------- the airframe
def build_airframe(spec, wing, part):
    st = wing.stations()
    rings = [wing.ring(s, -1.0) for s in reversed(st[1:])] + [wing.ring(s, 1.0) for s in st]
    kinds = [s[3] for s in reversed(st[1:])] + [s[3] for s in st]
    v, f = loft(rings)
    N = len(rings[0])
    mats = []
    for i in range(len(rings) - 1):
        m = "winglet_orange" if kinds[i] == "winglet" and kinds[i + 1] == "winglet" else "foam_grey"
        mats += [m] * N
    mats += ["winglet_orange", "winglet_orange"]
    part.add_multi(v, f, mats)


def rounded_halfwidth(w, r, d_end):
    """Half-width of a rounded-corner rectangle at distance d_end from its nearer end."""
    if d_end >= r:
        return w
    return w - r + math.sqrt(max(0.0, r * r - (r - d_end) ** 2))


def build_hatches(spec, wing, part):
    H = spec["geometry"]["hatches"]
    proud, r = H["proud"], H["corner_r"]
    for (s0, s1, w) in H["covers"]:
        ss = np.linspace(s0, s1, 26)
        rings = []
        for s in ss:
            hw = max(0.004, rounded_halfwidth(w, r, min(s - s0, s1 - s)))
            ls = np.linspace(-hw, hw, 13)
            top = [(wing.X(s), wing.surface_y(abs(l), s) + proud, l) for l in ls]
            bot = [(wing.X(s), wing.surface_y(abs(l), s) - 0.004, l) for l in ls]
            rings.append(bot + top[::-1])
        v, f = loft(rings)
        n = len(rings[0])
        mats = []
        for i in range(len(rings) - 1):
            for k in range(n):
                mats.append("hatch_orange" if 13 <= k < n - 1 else "seam_dark")
        mats += ["seam_dark", "seam_dark"]
        part.add_multi(v, f, mats)


def build_fittings(spec, wing, part):
    G = spec["geometry"]
    # pitot tube from the nose
    pt = G["pitot"]
    xn = wing.x_nose
    v, f = cylinder(pt["radius"], xn - 0.02, xn + pt["length"], 12)
    part.add(v, f, "pitot_steel", Z_TO_X())
    # motor fairing and can, on the body's x axis
    M = G["motor"]
    fs0, fs1 = M["fairing_s"]
    v, f = revolve([(0.0, wing.X(fs0)), (0.6 * M["fairing_r"], wing.X(fs0) - 0.03), (M["fairing_r"], wing.X(fs0) - 0.07),
                    (M["fairing_r"], wing.X(fs1))], 24)
    part.add(v, f, "foam_grey", Z_TO_X())
    cs0, cs1 = M["can_s"]
    v, f = cylinder(M["can_r"], wing.X(cs1), wing.X(cs0), 28)
    part.add(v, f, "motor_grey", Z_TO_X())
    v, f = cylinder(0.004, wing.X(M["spinner_s"][0]) + 0.002, wing.X(cs1) + 0.001, 10)
    part.add(v, f, "spinner_alu", Z_TO_X())
    # GNSS puck on the rear hatch
    SE = G["sensors"]
    gx, gy, gz = SE["gnss"]["position"]
    y0 = wing.surface_y(0.0, wing.x_nose - gx) + G["hatches"]["proud"]
    v, f = revolve([(0.0, y0 - 0.002), (0.028, y0 - 0.002), (0.028, y0 + 0.008), (0.022, y0 + 0.014), (0.0, y0 + 0.016)], 24)
    part.add(v, f, "antenna_white", T(gx, 0.0, gz) @ Z_TO_Y())
    # camera windows: forward in the nose, nadir in the belly
    cf = SE["camera_fpv"]["position"]
    v, f = cylinder(0.012, -0.003, 0.002, 20)
    part.add(v, f, "lens_glass", T(*cf) @ Z_TO_X())
    cn = SE["camera_nadir"]["position"]
    y_b = wing.surface_y(0.0, wing.x_nose - cn[0], upper=False)
    v, f = cylinder(0.018, y_b - 0.002, y_b + 0.003, 24)
    part.add(v, f, "lens_glass", T(cn[0], 0.0, cn[2]) @ Z_TO_Y())


# ---------------------------------------------------------------- elevons
def hinge_frame(p_from, p_to):
    """4x4 node matrix: origin p_from, local +Z along p_from -> p_to, local +Y up (orthogonalised)."""
    z = np.asarray(p_to, float) - np.asarray(p_from, float)
    z /= np.linalg.norm(z)
    y = np.array([0.0, 1.0, 0.0]) - z[1] * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, p_from
    return M


def build_elevon(spec, wing, side):
    """The elevon on one side: its node matrix and the mesh in the node's frame."""
    l0, l1 = wing.e_l0, wing.e_l1
    ls = np.linspace(l0, l1, 12)
    n = wing.N_CHORD // 2
    rings, hinge = [], []
    for l in ls:
        sec = wing.flat(l, cut=True)
        xc = sec["x_cut"]
        xs = chord_points(xc, 1.0, n)
        c, tr, k, m = sec["c"], sec["t"] / sec["c"], sec["k"], sec["m"]
        pts = []
        for x, up in [(x, True) for x in reversed(xs)] + [(x, False) for x in xs]:
            yt = thickness(x) * tr * c
            y = camber(x, m) * c + (2.0 * k * yt if up else -2.0 * (1.0 - k) * yt)
            pts.append((wing.X(sec["s_le"] + x * c), y, side * l))
        rings.append(pts)
        hinge.append((wing.X(sec["s_le"] + xc * c), camber(xc, m) * c + (2.0 * k - 1.0) * thickness(xc) * tr * c, side * l))
    # node: on the hinge line's starboard-going direction
    a, b = (hinge[0], hinge[-1]) if side > 0 else (hinge[-1], hinge[0])
    M = hinge_frame(a, b)
    part = Part()
    v, f = loft(rings)
    part.add(v, f, "foam_grey", np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the propeller
def build_propeller(spec):
    """Blades, hub and spinner in the propeller node's frame (spin axis local +X)."""
    P = spec["geometry"]["propeller"]
    Rt = 0.5 * P["diameter"]
    r0 = P["root_r"]
    part = Part()
    rings = []
    rs = np.linspace(r0, Rt, 12)
    for r in rs:
        f = (r - r0) / (Rt - r0)
        c = P["chord_root"] + (P["chord_tip"] - P["chord_root"]) * f ** 0.8
        if f > 0.9:
            c *= math.sqrt(max(0.15, 1.0 - ((f - 0.9) / 0.1) ** 2))     # a rounded tip
        beta = math.atan(P["pitch"] / (2.0 * math.pi * r))
        th = P["thickness"] * (1.0 - 0.5 * f)
        # the section in the plane perpendicular to the blade (local Y): chord from the trailing
        # edge to the leading edge along cos(beta) +Z + sin(beta) +X, thickness across it
        ch = np.array([math.sin(beta), 0.0, math.cos(beta)])
        nn = np.array([math.cos(beta), 0.0, -math.sin(beta)])
        pts = []
        for (u, w) in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            p = u * c * ch + w * th * nn
            pts.append((p[0], r, p[2]))
        rings.append(pts)
    v, f = loft(rings)
    for ang in (0.0, 180.0):
        part.add(v, f, "prop_black", R("X", ang))
    # the folding yoke and the spinner
    hb = P["hub_r"]
    vv, ff, MM = box_between((-0.008, -r0 - 0.004, -0.009), (0.008, r0 + 0.004, 0.009))
    part.add(vv, ff, "prop_black", MM)
    v, f = revolve([(0.0, 0.03), (0.4 * hb, 0.026), (0.8 * hb, 0.016), (hb, 0.004), (hb, -0.008), (0.0, -0.008)], 24)
    part.add(v, f, "spinner_alu", R("Y", -90.0))
    return part


# ---------------------------------------------------------------- checks
def planform_area(wing, n=2000):
    """Projected area of the flat part (both sides), m^2, and its mean aerodynamic chord."""
    ls = np.linspace(0.0, wing.lb, n)
    c = np.array([wing.s_te(l) - wing.s_le(l) for l in ls])
    S = 2.0 * TRAPZ(c, ls)
    mac = 2.0 * TRAPZ(c * c, ls) / S
    return S, mac


def check(spec, wing, parts):
    G = spec["geometry"]
    V = np.array(parts["airframe"].verts)
    span = V[:, 2].max() - V[:, 2].min()
    S, mac = planform_area(wing)
    nose = wing.x_nose
    P = G["propeller"]
    spinner_end = wing.X(G["motor"]["spinner_s"][1])
    length = nose - spinner_end
    print(f"[x8] span over the winglets {span:.4f} m (spec {G['span']}), nose to spinner {length:.3f} m "
          f"(product page {G['length']}), winglet top {V[:, 1].max():.3f} m above the chord plane")
    print(f"[x8] planform area to the bend {S:.3f} m^2 (article S {spec['physical']['S']}, product page 0.80 with the winglets), "
          f"mean aerodynamic chord {mac:.3f} m (article cbar {spec['physical']['cbar']}), S / b {S / span:.3f}")
    ls = np.linspace(wing.e_l0, wing.e_l1, 200)
    e_area = 2.0 * TRAPZ([wing.s_te(l) - wing.hinge_s(l) for l in ls], ls)
    print(f"[x8] elevons {e_area:.4f} m^2 both ({100 * e_area / S:.1f} % of the planform); propeller disc at X "
          f"{wing.X(P['disc_s']):.3f}, {P['disc_s'] - max(wing.s_te(l) for l in np.linspace(0.0, 0.5 * P['diameter'], 50)):.3f} m "
          f"behind the trailing edge it sweeps")
    assert abs(span - G["span"]) < 0.005, span
    assert abs(length - G["length"]) < 0.02, length
    assert abs(S - spec["physical"]["S"]) < 0.08, S


# ---------------------------------------------------------------- Blender side
def build_geometry(spec):
    wing = Wing(spec)
    airframe, hatches, fittings = Part(), Part(), Part()
    build_airframe(spec, wing, airframe)
    build_hatches(spec, wing, hatches)
    build_fittings(spec, wing, fittings)
    return wing, {"airframe": airframe, "hatches": hatches, "fittings": fittings}


def build_blender(spec, wing, parts):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    P, A, G = spec["physical"], spec["reference"]["article"], spec["geometry"]
    root = new_object("x8", None, mats, None, I, props={
        "frame": "X forward, Y up, Z right; origin = CG, cg_from_nose behind the nose, in the chord plane",
        "spec": "x8_spec.json", "body_axes": "body (x fwd, y right, z down) = (X, Z, -Y)",
        "mass": P["mass"], "cg_from_nose": G["cg_from_nose"], "model": A["doi"]})
    new_object("airframe", parts["airframe"], mats, root, I, sharp_deg=40.0)
    new_object("hatches", parts["hatches"], mats, root, I, sharp_deg=50.0)
    new_object("fittings", parts["fittings"], mats, root, I)
    lim = spec["actuators"]["elevon_limit_deg"]
    for name, side in (("elevon_left", -1.0), ("elevon_right", 1.0)):
        part, M = build_elevon(spec, wing, side)
        new_object(name, part, mats, root, M, sharp_deg=40.0,
                   props={"axis": "+Z (hinge, toward starboard)", "limit_deg": lim,
                          "positive": "trailing edge down", "model_angle": "delta_el" if side < 0 else "delta_er"})
    PR = G["propeller"]
    new_object("propeller", build_propeller(spec), mats, root, T(wing.X(PR["disc_s"]), 0.0, 0.0),
               props={"axis": "+X", "diameter": PR["diameter"], "pitch": PR["pitch"],
                      "positive": "the model's Omega_p: spin vector along body +x, clockwise seen from behind"})
    SE = G["sensors"]
    new_object("imu", None, mats, root, T(*SE["imu"]["position"]))
    new_object("gnss", None, mats, root, T(*SE["gnss"]["position"]), props={"role": "phase centre"})
    new_object("pitot_tip", None, mats, root, T(*SE["pitot_tip"]["position"]))
    for name in ("camera_nadir", "camera_fpv"):
        c = SE[name]
        new_object(name, None, mats, root, sensor_frame(c["position"], c["look"]),
                   props={"convention": "look -Z, up +Y", "assumed": c["note"]})
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="x8_spec.json")
    ap.add_argument("--out", default="x8.glb")
    ap.add_argument("--check", action="store_true", help="build the geometry and print the checks, no Blender")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    wing, parts = build_geometry(spec)
    el, _ = build_elevon(spec, wing, 1.0)
    prop = build_propeller(spec)
    used = set(m for p in list(parts.values()) + [el, prop] for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    check(spec, wing, parts)
    if a.check:
        return
    import bpy
    build_blender(spec, wing, parts)
    bpy.context.view_layer.update()
    tris_n = 0
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            tris_n += sum(len(p.vertices) - 2 for p in ob.data.polygons)
    bpy.ops.export_scene.gltf(filepath=path(a.out), export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True)
    print(f"[x8] wrote {path(a.out)}  triangles={tris_n}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

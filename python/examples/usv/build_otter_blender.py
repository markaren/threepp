"""Build the Otter USV from otter_spec.json: otter.glb (via Blender) and otter_hydro.json.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_otter_blender.py -- --spec otter_spec.json --out otter.glb

Hydrostatics only (plain Python + numpy, no Blender; writes otter_hydro.json):
    python build_otter_blender.py --hydro-only

The mesh primitives, the hydrostatic integrator and the Blender export are the
Mariner generator's (build_mariner_blender.py) and the demihull lines are the
Otter X's (build_otterx_blender.py), both next to this file. Geometry is built
with numpy in the spec's vessel frame (X forward, Y up, Z starboard, metres).

Node hierarchy (all names exact):
    otter                        root, identity, origin = centreline x keel line x mid-LOA
      pontoons                   both demihulls, the tow eyes
      body                       pod, deck plate, orange side panels
      frame                      fore and aft tube frames, thruster struts and guards, cable loops
      deck_fittings              GNSS antennas, whip, LTE puck
      gantry                     leg plates, top plate, box, posts, bracket, sidelights
      thruster_port, thruster_stbd          fixed pods
        thruster_<side>_rotor               propeller, spins about local +X
        thrust_<side>                       empty at the propeller, thrust along local +X
      camera_main, gnss_fore, gnss_aft, ais_vhf_antenna, lte_antenna, imu, transducer   sensor empties
      nav_light_port, nav_light_stbd

Sensor empties look down local -Z with local +Y up (threepp camera convention).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mariner_blender import (Hydro, Part, R, T, Z_TO_X, Z_TO_Y, box, box_between, cylinder,  # noqa: E402
                                   extrude_xy, frame_from_forward_up, loft, make_materials, mass_condition,
                                   new_object, revolve, sensor_frame, strut)
from build_otterx_blender import PontoonLines, lerp2  # noqa: E402

XZ_PRISM = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # extrude_xy's (a, b, c) -> (x=a, y=c, z=b)


class OtterLines(PontoonLines):
    """The Otter X's demihull with a plan form that also narrows toward the stern."""

    def width(self, x):
        w = super().width(x)
        a = self.P["aft_taper"]
        if x < a["start_x"]:
            t = min(1.0, (a["start_x"] - x) / (a["start_x"] - self.tb[0]))
            w = 1.0 - (1.0 - a["min"]) * t * t
        return w


def lean_prism(part, poly, y0, z0, y1, z1, thickness, side, mat):
    """A plate whose side-view outline is poly (x, y), lying in the plane that leans from |z| = z0
    at y0 to z1 at y1, `thickness` thick toward the centreline."""
    def z_at(y):
        return z0 + (z1 - z0) * (y - y0) / (y1 - y0)
    rings = [[(x, y, side * (z_at(y) - d)) for (x, y) in poly] for d in (0.0, thickness)]
    part.add(*loft(rings), mat)


def plan_prism(part, poly_xz, y0, y1, mat):
    """A prism whose plan outline is poly (x, z), from y0 up to y1."""
    part.add(*extrude_xy(poly_xz, y0, y1), mat, XZ_PRISM)


def tube_path(part, pts, r, mat, n=12):
    for p0, p1 in zip(pts[:-1], pts[1:]):
        strut(part, p0, p1, r, mat, n)


# ---------------------------------------------------------------- pontoons, body
def build_pontoons(spec, L, part):
    P = spec["pontoons"]
    xs = L.stations()
    for side in (1, -1):
        rings = [L.ring(x, side, transom=(i == 0)) for i, x in enumerate(xs)]
        part.add(*loft(rings), "pontoon_pe_black", buoyant=True)
        te = P["tow_eyes"]
        for (x, y) in te["xy"]:
            z = side * (P["keel_z"] + L.width(x) * (P["outer_z"] - P["keel_z"]))
            v, f = cylinder(te["radius"], -0.004, 0.002, 16)
            part.add(v, f, "tunnel_dark", T(x, y, z))


def build_body(spec, part):
    B = spec["body"]
    for key, mat in (("pod", "pod_dark"), ("deck_plate", "deck_alu")):
        d = B[key]
        v, f, M = box_between((d["x_range"][0], d["y_range"][0], -d["halfwidth"]), (d["x_range"][1], d["y_range"][1], d["halfwidth"]))
        part.add(v, f, mat, M, buoyant=(key == "pod"))
    sp = B["side_panel"]
    poly = [tuple(p) for p in sp["profile_xy"]]
    ys = [p[1] for p in poly]
    for side in (1, -1):
        lean_prism(part, poly, min(ys), sp["z_bottom"], max(ys), sp["z_top"], sp["thickness"], side, "livery_orange")


# ---------------------------------------------------------------- the tube frames
def build_frame(spec, part):
    F, P, TH = spec["frame_tubes"], spec["pontoons"], spec["propulsion"]["thrusters"]
    s, y1 = F["size"], F["top_y"]
    y0, h = y1 - s, 0.5 * s

    def bar_x(x, za, zb):                                  # athwartships
        v, f, M = box_between((x - h, y0, za), (x + h, y1, zb))
        part.add(v, f, "frame_black", M)

    def bar_z(z, xa, xb):                                  # fore and aft
        v, f, M = box_between((xa, y0, z - h), (xb, y1, z + h))
        part.add(v, f, "frame_black", M)
    # fore: cross beam on two feet, arms converging to the front bar, centre spine
    fo = F["fore"]
    bx, fx, fw = fo["beam_x"], fo["front_x"], fo["front_halfwidth"]
    bar_x(bx, -fo["beam_halfwidth"], fo["beam_halfwidth"])
    bar_x(fx - h, -fw, fw)
    bar_z(0.0, bx, fx - s)
    for side in (1, -1):
        v, f, M = box_between((bx - h, P["deck_y"] - 0.005, side * fo["feet_z"] - h), (bx + h, y0 + 0.002, side * fo["feet_z"] + h))
        part.add(v, f, "frame_black", M)
        a, b = np.array([bx + h, side * fo["arm_root_z"]]), np.array([fx - s, side * fw])
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]) * h
        plan_prism(part, [tuple(a - n), tuple(b - n + d * h), tuple(b + n + d * h), tuple(a + n)], y0, y1, "frame_black")
    # aft: cross beam, an arm over each pontoon, the antenna bar
    af = F["aft"]
    bar_x(af["beam_x"], -af["beam_halfwidth"], af["beam_halfwidth"])
    bar_x(af["bar_x"], -af["arm_z"], af["arm_z"])
    cl = F["cable_loops"]
    for side in (1, -1):
        z = side * af["arm_z"]
        bar_z(z, af["arm_end_x"], af["beam_x"])
        # thruster strut from the arm down through the pontoon to the pod, and the guard
        strut(part, (TH["strut_x"], TH["pod_y"], z), (TH["strut_x"], y0 + 0.002, z), TH["strut_radius"], "strut_steel")
        g = TH["guard"]
        tube_path(part, [(x, y, z) for (x, y) in g["path_xy"]], g["radius"], "frame_black")
        # cable loop: two legs and a half circle
        xa, xb = cl["x"]
        rc, zc = 0.5 * abs(xb - xa), side * cl["z"]
        yc = cl["top_y"] - rc
        pts = [(xa, P["deck_y"] - 0.005, zc)]
        pts += [(0.5 * (xa + xb) + rc * math.cos(math.pi * k / 10) * (1 if xa > xb else -1), yc + rc * math.sin(math.pi * k / 10), zc)
                for k in range(11)]
        pts.append((xb, P["deck_y"] - 0.005, zc))
        tube_path(part, pts, cl["radius"], "frame_black", 10)


# ---------------------------------------------------------------- deck fittings
def build_deck_fittings(spec, part):
    SE, F = spec["sensors"], spec["frame_tubes"]
    for name in ("gnss_fore", "gnss_aft"):
        g = SE[name]
        gx, _, gz = g["position"]
        d0, d1 = g["disc_y"]
        at = T(gx, 0.0, gz) @ Z_TO_Y()
        v, f = cylinder(g["post_radius"], F["top_y"] - 0.002, d0 + 0.005, 16)
        part.add(v, f, "frame_black", at)
        R0 = g["disc_radius"]
        v, f = revolve([(0.0, d0), (0.55 * R0, d0), (R0, d0 + 0.02), (R0, d0 + 0.028), (0.8 * R0, d0 + 0.046),
                        (0.45 * R0, d1 - 0.004), (0.0, d1)], 32)
        part.add(v, f, "gnss_white", at)
    av = SE["ais_vhf_antenna"]
    wx, wy, wz = av["base"]
    v, f = revolve([(0.0, wy - 0.002), (0.012, wy - 0.002), (0.012, wy + 0.03), (av["radius"] * 1.6, wy + 0.06),
                    (av["radius"], wy + av["length"] - 0.01), (0.0, wy + av["length"])], 10)
    part.add(v, f, "frame_black", T(wx, 0.0, wz) @ Z_TO_Y())
    lt = SE["lte_antenna"]
    x2, y2, z2 = lt["position"]
    r2 = lt["radius"]
    v, f = revolve([(0.0, y2 - 0.002), (r2, y2 - 0.002), (r2, y2 + 0.7 * lt["height"]), (0.85 * r2, y2 + lt["height"]), (0.0, y2 + lt["height"])], 24)
    part.add(v, f, "frame_black", T(x2, 0.0, z2) @ Z_TO_Y())


# ---------------------------------------------------------------- the gantry
def build_gantry(spec, part):
    G, SE, LI = spec["gantry"], spec["sensors"], spec["lights"]
    lg = G["leg"]
    front, aft = [tuple(p) for p in lg["front_xy"]], [tuple(p) for p in lg["aft_xy"]]
    ya, yb = lg["rung_y"]
    fi, ai = (front[1], front[2]), (aft[0], aft[3])       # the strips' facing edges
    rung = [(lerp2(*ai, ya) - 0.01, ya), (lerp2(*fi, ya) + 0.01, ya), (lerp2(*fi, yb) + 0.01, yb), (lerp2(*ai, yb) - 0.01, yb)]
    y_foot, y_head = front[0][1], front[2][1]
    for side in (1, -1):
        for poly in (front, aft, rung):
            lean_prism(part, poly, y_foot, lg["z_foot"], y_head, lg["z_head"], lg["thickness"], side, "gantry_black")
    tp = G["top_plate"]
    v, f, M = box_between((tp["x_range"][0], tp["y_range"][0], -tp["halfwidth"]), (tp["x_range"][1], tp["y_range"][1], tp["halfwidth"]))
    part.add(v, f, "gantry_black", M)
    bx = G["box"]
    v, f, M = box_between((bx["x_range"][0], bx["y_range"][0], -bx["halfwidth"]), (bx["x_range"][1], bx["y_range"][1] + 0.002, bx["halfwidth"]))
    part.add(v, f, "gantry_black", M)
    cam = SE["camera_main"]
    v, f = cylinder(cam["lens_radius"] + 0.006, 0.0, 0.006, 24)
    part.add(v, f, "bracket_alu", T(bx["x_range"][1], cam["position"][1], cam["position"][2]) @ Z_TO_X())
    v, f = cylinder(cam["lens_radius"], 0.0, 0.008, 24)
    part.add(v, f, "lens_glass", T(bx["x_range"][1], cam["position"][1], cam["position"][2]) @ Z_TO_X())
    top = tp["y_range"][1]
    po = G["posts"]
    for side in (1, -1):
        v, f = revolve([(0.0, top - 0.002), (po["radius"] + 0.004, top - 0.002), (po["radius"] + 0.004, top + 0.02), (po["radius"], top + 0.02),
                        (po["radius"], po["top_y"] - 0.004), (0.0, po["top_y"])], 16)
        part.add(v, f, "antenna_white", T(po["x"], 0.0, side * po["z"]) @ Z_TO_Y())
    # upright bracket: two bars up to the apex, and the small round camera between them
    br = G["bracket"]
    hw, t = br["base_halfwidth"], 0.5 * br["thickness"]
    for side in (1, -1):
        a, b = np.array([side * hw, top]), np.array([0.0, br["apex_y"]])
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]) * 0.012
        quad = [tuple(a - n), tuple(b - n), tuple(b + n), tuple(a + n)]
        v = [(br["x"] + dx, y, z) for dx in (-t, t) for (z, y) in quad]
        f = [(0, 1, 2, 3), (7, 6, 5, 4)] + [(k, 4 + k, 4 + (k + 1) % 4, (k + 1) % 4) for k in range(4)]
        part.add(v, f, "bracket_alu")
    strut(part, (br["x"], top, 0.0), (br["x"], br["eye_y"], 0.0), 0.006, "gantry_black")
    v, f = cylinder(br["eye_r"], -0.02, 0.025, 24)
    part.add(v, f, "gantry_black", T(br["x"], br["eye_y"], 0.0) @ Z_TO_X())
    v, f = cylinder(0.6 * br["eye_r"], 0.024, 0.027, 20)
    part.add(v, f, "lens_glass", T(br["x"], br["eye_y"], 0.0) @ Z_TO_X())
    for name, mat in (("nav_light_stbd", "nav_green"), ("nav_light_port", "nav_red")):
        lx, ly, lz = LI[name]["position"]
        s = 1.0 if lz > 0 else -1.0
        v, f, M = box_between((lx - 0.022, ly - 0.02, lz - s * 0.02), (lx + 0.022, ly + 0.02, lz + s * 0.012))
        part.add(v, f, "gantry_black", M)
        v, f = cylinder(0.014, -0.012, 0.014, 16)
        part.add(v, f, mat, T(lx + 0.012, ly, lz + s * 0.012) @ Z_TO_Y())


# ---------------------------------------------------------------- the thrusters
def build_thruster(spec):
    """Pod (node frame: on the strut axis at the pod's centreline, vessel axes) and rotor."""
    TH = spec["propulsion"]["thrusters"]
    pod, rotor = Part(), Part()
    (xa, xb), r = TH["pod"]["x_range"], TH["pod"]["radius"]
    v, f = revolve([(0.0, xa), (0.5 * r, xa), (r, xa + 0.03), (r, xb - 0.07), (0.75 * r, xb - 0.025), (0.0, xb)], 24)
    pod.add(v, f, "pod_black", Z_TO_X())
    pod.add(*extrude_xy([(0.01, -r - 0.04), (0.11, -r - 0.04), (0.14, -0.5 * r), (0.0, -0.5 * r)], -0.005, 0.005), "pod_black")   # the fin
    rt = TH["rotor"]
    hl = 0.5 * rt["hub_length"]
    v, f = revolve([(0.0, -hl - 0.012), (0.6 * rt["hub_r"], -hl), (rt["hub_r"], -0.3 * hl), (rt["hub_r"], hl), (0.0, hl)], 20)
    rotor.add(v, f, "prop_orange", Z_TO_X())
    span = rt["blade_tip_r"] - 0.8 * rt["hub_r"]
    for k in range(rt["blades"]):
        v, f = box(rt["blade_chord"], span, 0.005)
        M = R("X", 360.0 * k / rt["blades"]) @ T(0.0, 0.8 * rt["hub_r"] + 0.5 * span, 0.0) @ R("Y", rt["pitch_deg"])
        rotor.add(v, f, "prop_orange", M)
    return pod, rotor, T(rt["x"], 0.0, 0.0)


# ---------------------------------------------------------------- hydrostatics
def check_spec(spec):
    M = spec["mass"]
    tot = sum(b["mass"] for b in M["budget"])
    assert abs(tot - M["dry"]) < 1e-9, tot
    m, com = mass_condition(spec, "lightship")
    assert all(abs(com[i] - M["com_dry"][i]) < 0.005 for i in range(3)), com
    P, pr = spec["pontoons"], spec["principal"]
    assert abs(2 * P["outer_z"] - pr["beam"]) < 1e-6
    kp = P["keel_profile"]
    assert abs(kp[-1][0] - P["transom"]["top"][0] - pr["loa"]) < 1e-6
    g, G = spec["propulsion"]["thrusters"]["guard"], spec["gantry"]
    low = min(y for (_, y) in g["path_xy"]) - g["radius"]
    assert abs(low - pr["lowest_point_y"]) < 1e-6, low
    assert abs(G["posts"]["top_y"] - low - pr["height"]) < 1e-6, G["posts"]["top_y"] - low
    L = OtterLines(spec)
    for x in L.stations():
        assert L.keel(x) < L.top(x), ("keel above the pontoon top", x)
    return com


def hydrostatics(spec, tris):
    HS = spec["hydrostatics"]
    hy = Hydro(tris, HS["grid"])
    assert hy.closure < 1e-6, ("buoyant meshes are not closed", hy.closure)
    out = {"schema": "threepp.usv_hydro/1", "spec": "otter_spec.json", "rho": HS["rho"], "g": HS["g"],
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
    L = OtterLines(spec)
    parts = {k: Part() for k in ("pontoons", "body", "frame", "deck_fittings", "gantry")}
    build_pontoons(spec, L, parts["pontoons"])
    build_body(spec, parts["body"])
    build_frame(spec, parts["frame"])
    build_deck_fittings(spec, parts["deck_fittings"])
    build_gantry(spec, parts["gantry"])
    tris = np.concatenate(parts["pontoons"].tris + parts["body"].tris)
    return parts, tris


def build_blender(spec, parts, hydro):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    d = hydro["conditions"][hydro["design_condition"]]
    m_dry, com = mass_condition(spec, "lightship")
    root = new_object("otter", None, mats, None, I, props={
        "frame": "X forward, Y up, Z starboard; origin = centreline x keel line x mid-LOA",
        "spec": "otter_spec.json", "hydro": "otter_hydro.json",
        "mass_dry": m_dry, "com_dry": [round(c, 4) for c in com],
        "design_waterline_y": d["waterline_y_at_x0"], "design_trim_deg": d["trim_deg"]})
    for name, part in parts.items():
        new_object(name, part, mats, root, I, sharp_deg=35.0 if name in ("pontoons", "body") else 30.0)

    TH = spec["propulsion"]["thrusters"]
    kz = spec["frame_tubes"]["aft"]["arm_z"]
    for side, name in ((-1, "port"), (1, "stbd")):
        pod, rotor, M_rotor = build_thruster(spec)
        node = new_object(f"thruster_{name}", pod, mats, root, T(TH["strut_x"], TH["pod_y"], side * kz), props={"fixed": True})
        new_object(f"thruster_{name}_rotor", rotor, mats, node, M_rotor, props={"axis": "+X"})
        new_object(f"thrust_{name}", None, mats, node, M_rotor,
                   props={"thrust_axis": "+X", "bollard_thrust_n": TH["bollard_thrust_n"], "power_kw": TH["power_kw"],
                          "assumed": TH["assumed"]})

    SE = spec["sensors"]
    cam = SE["camera_main"]
    new_object("camera_main", None, mats, root, sensor_frame(cam["position"], cam["look"], cam["pitch_down_deg"]),
               props={"convention": "look -Z, up +Y", "hfov_deg": cam["hfov_deg"],
                      "resolution": cam["resolution"], "fps": cam["fps"]})
    for name in ("gnss_fore", "gnss_aft"):
        new_object(name, None, mats, root, T(*SE[name]["position"]), props={"role": "phase centre", "baseline": SE["gnss_baseline"]})
    av = SE["ais_vhf_antenna"]
    new_object("ais_vhf_antenna", None, mats, root, T(*av["base"]), props={"length": av["length"]})
    new_object("lte_antenna", None, mats, root, T(*SE["lte_antenna"]["position"]))
    new_object("imu", None, mats, root, T(*SE["imu"]["position"]))
    td = SE["transducer"]
    new_object("transducer", None, mats, root, T(*td["position"]) @ frame_from_forward_up(td["look"], (1.0, 0.0, 0.0)),
               props={"convention": "look -Z, up +Y"})
    for name, d_ in spec["lights"].items():
        new_object(name, None, mats, root, sensor_frame(d_["position"], d_["look"]),
                   props={"convention": "look -Z, up +Y", "colour": d_["colour"]})
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="otter_spec.json")
    ap.add_argument("--out", default="otter.glb")
    ap.add_argument("--hydro-out", default="otter_hydro.json")
    ap.add_argument("--hydro-only", action="store_true")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def path(p):
        return p if os.path.isabs(p) else os.path.join(here, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    com = check_spec(spec)
    print(f"[otter] spec ok; dry CoM from budget = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")

    parts, tris = build_geometry(spec)
    pod, rotor, _ = build_thruster(spec)
    used = set(m for p in list(parts.values()) + [pod, rotor] for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    hydro = hydrostatics(spec, tris)
    for cond, r in hydro["conditions"].items():
        print(f"[otter] {cond:9s} {r['mass']:6.0f} kg  draft {r['draft_moulded_x0']:.3f}  max draft {r['draft_max']:.3f}  "
              f"trim {r['trim_deg']:+.2f} deg  LCB {r['lcb']:+.3f}  GM_T {r['gm_t']:.2f}  GM_L {r['gm_l']:.1f}  "
              f"LWL {r['lwl']:.2f}  BWL {r['bwl']:.2f}  Awp {r['waterplane_area']:.2f}")
    with open(path(a.hydro_out), "w", encoding="utf-8") as fh:
        json.dump(hydro, fh, indent=1)
    print(f"[otter] wrote {path(a.hydro_out)}")
    HS = spec["hydrostatics"]
    for cond, r in hydro["conditions"].items():
        assert abs(r["trim_deg"]) <= HS["trim_limit_deg"], (cond, r["trim_deg"])
        assert r["gm_t"] > 0.3, (cond, r["gm_t"])
        assert r["waterline_y_at_x0"] < spec["pontoons"]["deck_y"] - 0.05, (cond, "the pontoon decks are awash")
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
    print(f"[otter] wrote {path(a.out)}  triangles={tris_n}  y range [{zmin:.3f}, {zmax:.3f}]")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

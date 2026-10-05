"""Build the REMUS 100 from remus100_spec.json: remus100.glb (via Blender).

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_remus_blender.py -- --spec remus100_spec.json --out remus100.glb

Geometry only (plain Python + numpy, no Blender; prints the checks):
    python build_remus_blender.py --check

The mesh primitives and the Blender export are build_common's
(../build_common.py, shared by every generator); this script adds the vehicle. All geometry is built
with numpy in the model frame (X forward, Y up, Z right, metres, origin at the centre of
buoyancy) and converted to Blender's Z-up only when a mesh or a transform is handed to
bpy. The glTF exporter's Y-up conversion maps it straight back, so every number in the
spec is the number in the .glb.

The hull is the thesis's Myring profile, eqs. (2.1)-(2.3) with Table 2.1's parameters,
revolved: s is the distance aft of the nose tip, X = cb_from_nose - s. The four fins are
Fig. 2-3's trapezoids with NACA 0012 sections, each on its own node at the fin post
(Table 2.2's x_fin); the propeller, the transducers and the antenna are ASSUMED where the
spec says so.

Node hierarchy (all names exact):
    remus100                    root, identity, origin = centre of buoyancy (the thesis's body origin)
      hull                      the Myring hull, the nose cap and the section seams
      fittings                  LBL transducer, nose pockets, side-scan and ADCP transducers, antenna
      stern_port, stern_starboard   the stern-plane pair: local +Z = body +y (model +Z), so a
                                positive turn about local +Z puts the trailing edge down (delta_s > 0)
      rudder_upper, rudder_lower    the rudder pair: local +Z = body +z (model -Y), so a positive
                                turn about local +Z puts the trailing edge to port (delta_r > 0)
      propeller                 at the hub, spins about local +X (blades, hub, cap)
      imu, depth, lbl, adcp_down, adcp_up, sss_port, sss_starboard, gnss      sensor empties

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
sys.path.insert(0, os.path.dirname(HERE))          # examples/ (build_common)
from build_common import (Part, R, T, Z_TO_X, box, cylinder, loft, make_materials, new_object, revolve,  # noqa: E402
                          sensor_frame)


# ---------------------------------------------------------------- the hull
class Hull:
    """The Myring profile of the spec's geometry.myring block: radius r(s) at s metres aft of
    the nose tip, and the model-frame X of s."""

    def __init__(self, spec):
        M = spec["geometry"]["myring"]
        self.M = M
        self.a, self.ao, self.b, self.c, self.co = M["a"], M["a_offset"], M["b"], M["c"], M["c_offset"]
        self.n, self.th, self.d = M["n"], M["theta"], M["d"]
        self.lf = self.a + self.b - self.ao                       # (2.3)
        self.length = self.a + self.b + self.c - self.ao - self.co
        self.x0 = M["cb_from_nose"]

    def r(self, s):
        d, a, c = self.d, self.a, self.c
        if s <= 0.0:
            s = 0.0
        if s < self.a - self.ao:                                  # (2.1) the nose
            k = (s + self.ao - a) / a
            return 0.5 * d * max(0.0, 1.0 - k * k) ** (1.0 / self.n)
        if s <= self.lf:                                          # the parallel mid-body
            return 0.5 * d
        t = min(s, self.length) - self.lf                         # (2.2) with (s - lf)^2: the spec's note
        tn = math.tan(self.th)
        return 0.5 * d - (1.5 * d / (c * c) - tn / c) * t * t + (d / c ** 3 - tn / (c * c)) * t ** 3

    def X(self, s):
        return self.x0 - s

    def stations(self):
        sn = self.a - self.ao
        s = [sn * (1.0 - math.cos(0.5 * math.pi * k / 28)) for k in range(29)]   # closer together toward the tip
        s += list(np.linspace(self.a - self.ao, self.lf, 12)[1:])
        s += list(np.linspace(self.lf, self.length, 40)[1:])
        return sorted(set(round(v, 7) for v in s))


def ring(X, r, n):
    return [(X, r * math.cos(2.0 * math.pi * k / n), r * math.sin(2.0 * math.pi * k / n)) for k in range(n)]


def build_hull(spec, hull, part):
    G = spec["geometry"]
    N = 64
    ss = hull.stations()
    nose_end = G["seams"]["s"][0]
    rings = [ring(hull.X(s), hull.r(s), N) for s in ss]
    v, f = loft(rings)
    mats = []
    for i in range(len(ss) - 1):
        mats += ["nose_black" if 0.5 * (ss[i] + ss[i + 1]) < nose_end else "hull_yellow"] * N
    mats += ["nose_black", "hull_yellow"]
    part.add_multi(v, f, mats)
    # seams: thin rings proud of the skin at the section joints
    S = G["seams"]
    for s in S["s"]:
        rr = hull.r(s) + S["proud"]
        v, f = cylinder(rr, hull.X(s) - 0.5 * S["width"], hull.X(s) + 0.5 * S["width"], N)
        part.add(v, f, "seam_dark", Z_TO_X())


# ---------------------------------------------------------------- fittings
def surface_frame(X, ang, r):
    """4x4 at the hull surface point (X, r cos ang, r sin ang): local +Y the outward normal
    (radial), local +X forward."""
    nrm = np.array([0.0, math.cos(ang), math.sin(ang)])
    x = np.array([1.0, 0.0, 0.0])
    z = np.cross(x, nrm)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, nrm, z, (X, r * nrm[1], r * nrm[2])
    return M


def build_fittings(spec, hull, part):
    G = spec["geometry"]
    # the LBL transducer under the nose (Fig. 2-2): a cylinder, axis vertical
    L = G["lbl_transducer"]
    X = hull.X(L["s"])
    v, f = cylinder(L["radius"], -L["bottom"], -0.5 * hull.r(L["s"]), 40)
    part.add(v, f, "transducer_grey", T(X, 0.0, 0.0) @ R("X", -90.0))
    # nose pockets
    P = G["nose_pockets"]
    for k in range(P["count"]):
        ang = math.radians(90.0 + 120.0 * k)
        rr = hull.r(P["s"])
        v, f = cylinder(P["radius"], -0.004, 0.0015, 20)
        part.add(v, f, "seam_dark", surface_frame(hull.X(P["s"]), ang, rr) @ R("X", -90.0))
    # side-scan transducers, low on each side of the mid-body
    S = G["side_scan"]
    for side in (-1.0, 1.0):
        ang = math.atan2(side * math.cos(math.radians(S["angle_deg"])), -math.sin(math.radians(S["angle_deg"])))
        xc = hull.X(S["s0"] + 0.5 * S["length"])
        v, f = box(S["length"], 2.0 * S["proud"], S["width"])
        part.add(v, f, "transducer_grey", surface_frame(xc, ang, hull.r(S["s0"])))
    # the ADCP: four transducers looking down, four up, the beams beam_deg off the vertical
    A = G["adcp"]
    Xa = hull.X(A["s"])
    for up in (1.0, -1.0):
        for k in range(4):
            az = math.radians(45.0 + 90.0 * k)
            b = np.array([math.sin(math.radians(A["beam_deg"])) * math.cos(az), up * math.cos(math.radians(A["beam_deg"])),
                          math.sin(math.radians(A["beam_deg"])) * math.sin(az)])
            ang = math.atan2(b[2], b[1])
            rr = hull.r(A["s"])
            p = np.array([Xa + 1.2 * A["radius"] * math.copysign(1.0, math.cos(az)), rr * math.cos(ang), rr * math.sin(ang)])
            yb = b
            xb = np.cross(yb, [0.0, 0.0, 1.0]) if abs(yb[2]) < 0.9 else np.cross(yb, [1.0, 0.0, 0.0])
            xb /= np.linalg.norm(xb)
            zb = np.cross(xb, yb)
            M = np.eye(4)
            M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = xb, yb, zb, p
            v, f = cylinder(A["radius"], -0.006, A["proud"], 24)
            part.add(v, f, "transducer_grey", M @ R("X", -90.0))
    # the antenna mast on top of the tail cone
    N = G["antenna"]
    s = N["s"]
    y0 = hull.r(s) - 0.004
    pts = [(-0.5 * N["chord"], y0), (0.5 * N["chord"], y0), (0.25 * N["chord"], y0 + N["height"]), (-0.45 * N["chord"], y0 + N["height"])]
    rings_ = []
    for zz in (-0.5 * N["thickness"], 0.5 * N["thickness"]):
        rings_.append([(hull.X(s) + px, py, zz) for (px, py) in pts])
    v, f = loft(rings_)
    part.add(v, f, "antenna_white")
    v, f = cylinder(0.008, y0 + N["height"] - 0.002, y0 + N["height"] + 0.012, 16)
    part.add(v, f, "antenna_white", T(hull.X(s) - 0.12 * N["chord"], 0.0, 0.0) @ R("X", -90.0))


# ---------------------------------------------------------------- the fins
def naca00(x):
    """NACA four-digit half-thickness per unit thickness ratio, open trailing edge."""
    return 5.0 * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def fin_mesh(spec, hull, span_dir):
    """One fin in the model frame, spanning along the unit vector span_dir (radial) from inside
    the hull to the tip, chord along X. Fig. 2-3: leading edge from 142 mm (root, at the hull)
    to 112 mm (tip) ahead of the tail end, trailing edge 53 mm ahead of it, tip 131 mm out."""
    F = spec["geometry"]["fins"]
    end = hull.length
    d = np.asarray(span_dir, float)
    nrm = np.cross(np.array([1.0, 0.0, 0.0]), d)                 # thickness direction
    r_root_le = hull.r(end - F["root_le_from_end"])
    r_tip = F["tip_r"]
    r_in = hull.r(end - F["te_from_end"]) - 0.004               # the root, inside the skin along the whole chord
    slope = (F["root_le_from_end"] - F["tip_le_from_end"]) / (r_tip - r_root_le)
    xs = [0.5 * (1.0 - math.cos(math.pi * i / 18)) for i in range(19)]
    rings_ = []
    for r in np.linspace(r_in, r_tip, 10):
        le = F["root_le_from_end"] - slope * (r - r_root_le)       # distance of the LE ahead of the tail end
        te = F["te_from_end"]
        c = le - te
        pts = []
        for x, up in [(x, 1.0) for x in xs] + [(x, -1.0) for x in reversed(xs[1:-1])]:
            s = end - le + x * c
            t = up * naca00(x) * F["thickness_ratio"] * c
            p = np.array([hull.X(s), 0.0, 0.0]) + r * d + t * nrm
            pts.append(tuple(p))
        rings_.append(pts)
    return loft(rings_)


def build_fin(spec, hull, name):
    """The fin `name` as (Part in its node's frame, node matrix). The node is at the fin post on
    the hull's axis; its local +Z is the body axis the pair turns on."""
    xf = spec["fins"]["x_fin"]
    span = {"stern_port": (0.0, 0.0, -1.0), "stern_starboard": (0.0, 0.0, 1.0),
            "rudder_upper": (0.0, 1.0, 0.0), "rudder_lower": (0.0, -1.0, 0.0)}[name]
    M = T(xf, 0.0, 0.0) if name.startswith("stern") else T(xf, 0.0, 0.0) @ R("X", 90.0)
    v, f = fin_mesh(spec, hull, span)
    part = Part()
    part.add(v, f, "fin_black", np.linalg.inv(M))
    return part, M


# ---------------------------------------------------------------- the propeller
def build_propeller(spec):
    """Blades, hub and cap in the propeller node's frame (spin axis local +X). The blade
    sections are the X8's: chord from the trailing to the leading edge along
    sin(beta) +X + cos(beta) +Z at the top blade, so a spin about +X pushes water aft."""
    P = spec["geometry"]["propeller"]
    Rt, hb = 0.5 * P["diameter"], P["hub_r"]
    part = Part()
    rings_ = []
    for r in np.linspace(hb * 0.7, Rt, 12):
        f = (r - hb) / (Rt - hb)
        c = P["chord_root"] + (P["chord_tip"] - P["chord_root"]) * max(f, 0.0)
        if f > 0.85:
            c *= math.sqrt(max(0.2, 1.0 - ((f - 0.85) / 0.15) ** 2))
        beta = math.atan(P["pitch"] / (2.0 * math.pi * r))
        th = P["thickness"] * (1.0 - 0.5 * max(f, 0.0))
        ch = np.array([math.sin(beta), 0.0, math.cos(beta)])
        nn = np.array([math.cos(beta), 0.0, -math.sin(beta)])
        pts = []
        for (u, w) in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            p = u * c * ch + w * th * nn
            pts.append((p[0], r, p[2]))
        rings_.append(pts)
    v, f = loft(rings_)
    for k in range(P["blades"]):
        part.add(v, f, "prop_bronze", R("X", 360.0 * k / P["blades"]))
    L = P["hub_len"]
    v, f = revolve([(0.0, -0.5 * L - 0.012), (0.6 * hb, -0.5 * L - 0.008), (hb, -0.5 * L), (hb, 0.5 * L), (0.0, 0.5 * L)], 32)
    part.add(v, f, "hub_steel", Z_TO_X())
    return part


# ---------------------------------------------------------------- checks
def check(spec, hull, parts, fins):
    G, A1 = spec["geometry"], spec["geometry"]["hull_checks"]
    ss = np.linspace(0.0, hull.length, 20001)
    rs = np.array([hull.r(s) for s in ss])
    vol = TRAPZ(math.pi * rs * rs, ss)
    s_c = TRAPZ(math.pi * rs * rs * ss, ss) / vol
    Ap = TRAPZ(2.0 * rs, ss)
    dr = np.gradient(rs, ss)
    Sw = TRAPZ(2.0 * math.pi * rs * np.sqrt(1.0 + dr * dr), ss) + math.pi * rs[0] ** 2 + math.pi * rs[-1] ** 2
    Af = math.pi * (0.5 * hull.d) ** 2
    print(f"[remus] hull length {hull.length:.4f} m (Table 2.1 l {G['myring']['l']}), nose radius {hull.r(0.0) * 1e3:.1f} mm, "
          f"tail-end radius {hull.r(hull.length) * 1e3:.1f} mm, r at the fin's root LE {hull.r(hull.length - G['fins']['root_le_from_end']) * 1e3:.1f} mm (Fig. 2-3: 62)")
    print(f"[remus] volume {vol:.5f} m^3 (Table A.1 {A1['volume']}), frontal area {Af:.4f} m^2 ({A1['Af']}), projected area {Ap:.4f} m^2 ({A1['Ap']}), "
          f"wetted area {Sw:.4f} m^2 ({A1['Sw']})")
    print(f"[remus] the hull's volume centroid {s_c:.4f} m aft of the nose, {G['myring']['cb_from_nose'] - s_c:+.4f} m ahead of the measured "
          f"centre of buoyancy (Table 2.7: the bare-hull estimate is +0.00554 m); buoyancy rho g V {1030.0 * 9.81 * vol:.1f} N (Table 2.7 B_est 317)")
    F = G["fins"]
    r0 = hull.r(hull.length - F["root_le_from_end"])
    le0, le1, te = F["root_le_from_end"], F["tip_le_from_end"], F["te_from_end"]
    # exposed planform: between the hull's contour and the tip, the leading and trailing edges
    end = hull.length
    sq = np.linspace(end - le0 - 0.02, end - te, 4001)
    width = []
    for s in sq:
        a_ = end - s                                            # ahead of the tail end
        r_le = r0 + (le0 - a_) / (le0 - le1) * (F["tip_r"] - r0) if a_ > le1 else F["tip_r"]
        width.append(max(0.0, F["tip_r"] - max(hull.r(s), r_le if a_ > le1 else hull.r(s))))
    area = TRAPZ(width, sq)
    print(f"[remus] fin: span from the hull at the root's leading edge {F['tip_r'] - r0:.4f} m, from the hull at the trailing edge "
          f"{F['tip_r'] - hull.r(end - te):.4f} m (Table 2.2 b_fin 0.0857), mean chord {0.5 * ((le0 - te) + (le1 - te)):.4f} m (c_mean 0.0747), "
          f"exposed area {area:.5f} m^2 (S_fin 0.00665), taper {(le1 - te) / (le0 - te):.3f} (0.654); the post {spec['fins']['x_fin']:+.3f} m is "
          f"{hull.X(hull.length) - spec['fins']['x_fin']:+.4f} m from the tail end, inside the chord ({-te:+.3f}..{-le0:+.3f})")
    for name, (part, M) in fins.items():
        V = np.array(part.verts) @ M[:3, :3].T + M[:3, 3]
        print(f"[remus]   {name}: tip at {np.abs(V[:, 1:]).max():.4f} m from the axis, X {V[:, 0].min():+.3f}..{V[:, 0].max():+.3f}")
    assert abs(hull.length - 1.3327) < 1e-3, hull.length
    assert abs(vol - A1["volume"]) / A1["volume"] < 0.02, vol
    assert abs(Ap - A1["Ap"]) / A1["Ap"] < 0.02, Ap


# ---------------------------------------------------------------- Blender side
def build_geometry(spec):
    hull = Hull(spec)
    hp, fit = Part(), Part()
    build_hull(spec, hull, hp)
    build_fittings(spec, hull, fit)
    fins = {n: build_fin(spec, hull, n) for n in ("stern_port", "stern_starboard", "rudder_upper", "rudder_lower")}
    return hull, {"hull": hp, "fittings": fit}, fins


def build_blender(spec, hull, parts, fins):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    I = np.eye(4)
    P, Th = spec["physical"], spec["reference"]["thesis"]
    root = new_object("remus100", None, mats, None, I, props={
        "frame": "X forward, Y up, Z right; origin = centre of buoyancy, cb_from_nose behind the nose tip",
        "spec": "remus100_spec.json", "body_axes": "body (x fwd, y stbd, z down) = (X, Z, -Y)",
        "W": P["W"], "B": P["B"], "model": Th["handle"]})
    new_object("hull", parts["hull"], mats, root, I, sharp_deg=50.0)
    new_object("fittings", parts["fittings"], mats, root, I, sharp_deg=40.0)
    lim = spec["fins"]["delta_max_deg"]
    for name, (part, M) in fins.items():
        stern = name.startswith("stern")
        new_object(name, part, mats, root, M, sharp_deg=40.0,
                   props={"axis": "+Z = body +y (stern planes)" if stern else "+Z = body +z (rudders)", "limit_deg": lim,
                          "positive": "trailing edge down (delta_s)" if stern else "trailing edge to port (delta_r)"})
    PR = spec["geometry"]["propeller"]
    xp = hull.X(hull.length) - PR["gap"] - 0.5 * PR["hub_len"]
    new_object("propeller", build_propeller(spec), mats, root, T(xp, 0.0, 0.0),
               props={"axis": "+X", "diameter": PR["diameter"], "pitch": PR["pitch"],
                      "positive": "spin vector along body +x (clockwise seen from behind)"})
    G = spec["geometry"]
    SE = G["sensors"]
    new_object("imu", None, mats, root, T(*SE["imu"]["position"]))
    new_object("depth", None, mats, root, T(*SE["depth"]["position"]))
    L = G["lbl_transducer"]
    new_object("lbl", None, mats, root, sensor_frame((hull.X(L["s"]), -L["bottom"], 0.0), SE["lbl"]["look"]))
    A = G["adcp"]
    ra = hull.r(A["s"])
    new_object("adcp_down", None, mats, root, sensor_frame((hull.X(A["s"]), -ra, 0.0), SE["adcp_down"]["look"]))
    new_object("adcp_up", None, mats, root, sensor_frame((hull.X(A["s"]), ra, 0.0), SE["adcp_up"]["look"]))
    S = G["side_scan"]
    a = math.radians(S["angle_deg"])
    rs = hull.r(S["s0"]) + S["proud"]
    xs = hull.X(S["s0"] + 0.5 * S["length"])
    for name, side in (("sss_port", -1.0), ("sss_starboard", 1.0)):
        look = (0.0, -math.sin(a), side * math.cos(a))
        new_object(name, None, mats, root, sensor_frame((xs, -rs * math.sin(a), side * rs * math.cos(a)), look))
    N = G["antenna"]
    new_object("gnss", None, mats, root, T(hull.X(N["s"]), hull.r(N["s"]) + N["height"] + 0.01, 0.0), props={"role": "antenna top"})
    return root


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="remus100_spec.json")
    ap.add_argument("--out", default="remus100.glb")
    ap.add_argument("--check", action="store_true", help="build the geometry and print the checks, no Blender")
    a = ap.parse_args(argv)

    def path(p):
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    hull, parts, fins = build_geometry(spec)
    prop = build_propeller(spec)
    used = set(m for p in list(parts.values()) + [f[0] for f in fins.values()] + [prop] for m in p.mats)
    missing = sorted(used - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)
    check(spec, hull, parts, fins)
    if a.check:
        return
    import bpy
    build_blender(spec, hull, parts, fins)
    bpy.context.view_layer.update()
    tris_n = 0
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            tris_n += sum(len(p.vertices) - 2 for p in ob.data.polygons)
    bpy.ops.export_scene.gltf(filepath=path(a.out), export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True)
    print(f"[remus] wrote {path(a.out)}  triangles={tris_n}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

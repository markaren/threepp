"""Build the Otter USV: otter.glb (via Blender) and otter_hydro.json, from otter_spec.json and the
Otter model NTNU publishes.

The geometry is not drawn here. It is the Otter Jan Henrik Lenes made for his USV simulator
(github.com/jhlenes/usv_simulator) from Maritime Robotics' data, as NTNU's ITK publishes it with
the Gemini data under the MIT licence: the spec's "source" block. It is not kept in the
repository. The first build fetches it into ~/.cache/threepp/models/otter_ntnu
($THREEPP_MODEL_CACHE) with its licence, and every build checks it against the sha256 the spec
pins. This script turns its parts into the vessel frame, groups them into the nodes below, gives
them the spec's materials, adds the sensor and light empties, and integrates the hydrostatics
from its pontoons. otter.glb carries the copyright, the attribution and the licence text in its
root's extras.

Blender is a build tool only; the threepp runtime loads the .glb with
tp.GLTFLoader().load(path) and never imports bpy.

Headless:
    blender --background --factory-startup \
        --python build_otter_blender.py -- --spec otter_spec.json --out otter.glb

Hydrostatics only (plain Python + numpy, no Blender; writes otter_hydro.json):
    python build_otter_blender.py --hydro-only

--source PATH builds from a copy of the model you already have; it is checked against the same hash.

Node hierarchy (all names exact):
    otter                        root, identity, origin = centreline x keel line x mid-LOA
      pontoons                   both demihulls
      body                       deck plate, orange hatches, the pontoons' fittings, lettering
      frame                      the black frames fore and aft, the thruster guards, the centre mount
      deck_fittings              GNSS antennas, whip, the pucks and glands on the deck
      gantry                     A-frame, box, top plate, posts, the camera arch, sidelights
      thruster_port, thruster_stbd          fixed pods and their struts
        thruster_<side>_rotor               propeller, spins about local +X
        thrust_<side>                       empty at the propeller, thrust along local +X
      camera_main, gnss_fore, gnss_aft, ais_vhf_antenna, lte_antenna, imu, transducer   sensor empties
      nav_light_port, nav_light_stbd

Sensor empties look down local -Z with local +Y up (threepp camera convention).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/ (build_common)
from build_common import (Part, T, capped, fetch_source, frame_from_forward_up, hydrostatics,  # noqa: E402
                          make_materials, mass_check, mass_condition, new_object, read_glb, sensor_frame, weld)

NODES = ("pontoons", "body", "frame", "deck_fittings", "gantry")
SIDES = (("port", -1.0), ("stbd", 1.0))


# ---------------------------------------------------------------- the source, in the vessel frame
def to_vessel(src, P):
    """Source positions -> vessel axes, by the spec's axis map (a rotation)."""
    cols = []
    for ax in src["to_vessel"]["axes"]:
        cols.append((-1.0 if ax[0] == "-" else 1.0) * P[:, "xyz".index(ax[-1])])
    return np.stack(cols, axis=1)


def node_of(src, name, tex, P):
    """The node a source part goes in: the spec's lists first, then by where it sits."""
    S = src["parts"]
    for node, names in S["nodes"].items():
        if name in names:
            return node
    c = 0.5 * (P.min(axis=0) + P.max(axis=0))
    g = S["gantry_box"]
    if g["x"][0] <= c[0] <= g["x"][1] and c[1] >= g["y_min"] and abs(c[2]) <= g["half_z"]:
        return "gantry"
    if c[1] >= S["deck_top_y"]:
        return "deck_fittings"
    return "frame" if tex == "black4" else "body"


def load_source(spec, path):
    """The source's parts in the vessel frame: [(node, material, V, F)], each part's vertices
    welded (so the smoothing finds its edges), and how far the source was lifted."""
    src = spec["source"]
    M = src["material_map"]
    raw = [(p["name"], p["texture"], to_vessel(src, p["positions"]), p["faces"]) for p in read_glb(path)]
    names = {r[0] for r in raw}
    listed = {n for ns in src["parts"]["nodes"].values() for n in ns}
    assert listed <= names, ("parts the spec lists are not in the source", sorted(listed - names))
    lift = -min(P[:, 1].min() for (name, _, P, _) in raw if name in src["parts"]["nodes"]["pontoons"])
    parts = []
    for name, tex, P, F in raw:
        P = P + (0.0, lift, 0.0)
        node = node_of(src, name, tex, P)
        if node == "drop":
            continue
        key = tex or "none"
        mat = M["by_node"].get(node, {}).get(key) or M["by_texture"][key]
        V, F = weld(P, F)
        parts.append((node, mat, V, F))
    return parts, lift


def verts(parts, node):
    return np.concatenate([V for (n, _, V, _) in parts if n == node])


# ---------------------------------------------------------------- the spec against the source
def check(spec, parts):
    """The spec's numbers against the source's geometry: the build stops where they part.
    Returns the dry CoM."""
    com = mass_check(spec)
    pr, P, TH = spec["principal"], spec["pontoons"], spec["propulsion"]["thrusters"]

    def near(got, want, tol, what):
        assert abs(got - want) <= tol, (what, round(float(got), 4), want)
    A = np.concatenate([V for (_, _, V, _) in parts])
    near(np.ptp(A[:, 0]), pr["loa"], 0.005, "length overall")
    near(np.ptp(A[:, 2]), pr["beam"], 0.005, "beam")
    near(A[:, 1].min(), pr["lowest_point_y"], 0.002, "lowest point")
    near(verts(parts, "gantry")[:, 1].max() - A[:, 1].min(), pr["height"], 0.003, "height to the camera arch")
    pv = verts(parts, "pontoons")
    near(pv[:, 1].max(), P["deck_y"], 0.002, "pontoon deck")
    for side, s in SIDES:
        Rv = verts(parts, f"thruster_{side}_rotor")
        yc, zc = 0.5 * (Rv[:, 1].min() + Rv[:, 1].max()), 0.5 * (Rv[:, 2].min() + Rv[:, 2].max())
        near(yc, TH["pod_y"], 0.002, f"{side} rotor axis y")
        near(zc, s * TH["pod_z"], 0.002, f"{side} rotor axis z")
        near(0.5 * (Rv[:, 0].min() + Rv[:, 0].max()), TH["strut_x"] + TH["rotor"]["x"], 0.003, f"{side} rotor x")
        near(np.hypot(Rv[:, 1] - yc, Rv[:, 2] - zc).max(), TH["rotor"]["blade_tip_r"], 0.003, f"{side} blade tips")
        Sv = verts(parts, f"thruster_{side}")
        near(Sv[Sv[:, 1] > TH["pod_y"] + 0.06, 0].mean(), TH["strut_x"], 0.003, f"{side} strut x")
    g = TH["guard"]
    low = min(y for (_, y) in g["path_xy"]) - g["radius"]
    near(low, pr["lowest_point_y"], 1e-6, "guard feet")
    return com


def buoyant(parts):
    """The pontoons' triangles with their wells and holes capped, for the hydrostatics."""
    tris, n_caps = [], 0
    for (node, _, V, F) in parts:
        if node == "pontoons":
            t, c = capped(V, F)
            tris += [t, c]
            n_caps += len(c)
    return np.concatenate(tris), n_caps


# ---------------------------------------------------------------- Blender side
def build_blender(spec, parts, hydro, licence, lift):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats = make_materials(spec)
    for m in mats.values():
        m.use_backface_culling = False          # the source's parts are open shells, decals and cards
    src, I = spec["source"], np.eye(4)
    d = hydro["conditions"][hydro["design_condition"]]
    m_dry, com = mass_condition(spec, "lightship")
    root = new_object("otter", None, mats, None, I, props={
        "frame": "X forward, Y up, Z starboard; origin = centreline x keel line x mid-LOA",
        "spec": "otter_spec.json", "hydro": "otter_hydro.json",
        "mass_dry": m_dry, "com_dry": [round(c, 4) for c in com],
        "design_waterline_y": d["waterline_y_at_x0"], "design_trim_deg": d["trim_deg"],
        "source": src["url"], "source_sha256": src["sha256"], "source_lift": round(lift, 5),
        "copyright": src["copyright"], "licence": src["licence"], "attribution": src["attribution"],
        "licence_text": licence})

    def part(node, at=(0.0, 0.0, 0.0)):
        p = Part()
        for (n, mat, V, F) in parts:
            if n == node:
                p.add(V - np.asarray(at), F, mat, closed=False)
        return p
    for name in NODES:
        new_object(name, part(name), mats, root, I)

    TH = spec["propulsion"]["thrusters"]
    rx = TH["rotor"]["x"]
    for side, s in SIDES:
        at = np.array([TH["strut_x"], TH["pod_y"], s * TH["pod_z"]])
        node = new_object(f"thruster_{side}", part(f"thruster_{side}", at), mats, root, T(*at), props={"fixed": True})
        new_object(f"thruster_{side}_rotor", part(f"thruster_{side}_rotor", at + (rx, 0.0, 0.0)), mats, node, T(rx, 0.0, 0.0),
                   props={"axis": "+X"})
        new_object(f"thrust_{side}", None, mats, node, T(rx, 0.0, 0.0),
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
    ap.add_argument("--source", help="a copy of the source model to build from instead of the cache (checked against the spec's hash)")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    def path(p):
        return p if os.path.isabs(p) else os.path.join(here, p)
    with open(path(a.spec), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    src_path, licence = fetch_source(spec["source"], a.source and os.path.abspath(a.source), log=lambda s: print(f"[otter] {s}"))
    parts, lift = load_source(spec, src_path)
    n_tris = sum(len(F) for (_, _, _, F) in parts)
    print(f"[otter] source {src_path}: {len(parts)} parts, {n_tris} triangles, lifted {lift:.4f} m")
    com = check(spec, parts)
    print(f"[otter] spec ok against the source; dry CoM from budget = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
    missing = sorted({m for (_, m, _, _) in parts} - set(spec["materials"]))
    assert not missing, ("materials used but not in the spec", missing)

    tris, n_caps = buoyant(parts)
    print(f"[otter] buoyant: the pontoons, {len(tris) - n_caps} triangles and {n_caps} capping their wells and holes")
    hydro = hydrostatics(spec, tris, os.path.basename(a.spec), "keel line")
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
    build_blender(spec, parts, hydro, licence, lift)
    bpy.context.view_layer.update()
    tris_n, ymax, ymin = 0, -1e9, 1e9
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            tris_n += sum(len(p.vertices) - 2 for p in ob.data.polygons)
            mw = ob.matrix_world
            zs = [(mw @ v.co).z for v in ob.data.vertices]
            ymax, ymin = max(ymax, max(zs)), min(ymin, min(zs))
    out = path(a.out)
    tmp = out + ".tmp.glb"
    bpy.ops.export_scene.gltf(filepath=tmp, export_format="GLB", export_yup=True,
                              export_apply=False, export_extras=True, export_cameras=False,
                              export_lights=False, use_selection=False, export_animations=False,
                              export_texcoords=True, export_normals=True,
                              export_copyright=f"{spec['source']['copyright']} ({spec['source']['licence']}); {spec['source']['attribution']}")
    os.replace(tmp, out)
    print(f"[otter] wrote {out}  triangles={tris_n}  y range [{ymin:.3f}, {ymax:.3f}]")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

"""Give Spot's URDF the PHYSICS it is missing, so a URDF-driven simulator can build the trained plant.

Boston Dynamics' spot_base_urdf carries visuals and kinematics only — no <inertial>, no <collision>
(`spot_deploy.build_spot` therefore hand-builds the colliders and masses in code, which is why the
trained plant exists nowhere as a file). A URDF-driven host like the threepp editor cannot import
that: links with no collision and no mass are not a robot it can stand up.

This writes a NEW urdf beside the original (never overwriting it) with <inertial> and <collision>
injected into every link, taken from spot_deploy's own constants — so the file describes exactly the
plant the policy was trained against, and everything else (visual meshes, joint origins, axes,
per-leg limits) is the original's, untouched.

    python make_spot_urdf.py                       # -> ~/.cache/threepp/spot/spot_physics.urdf
    python make_spot_urdf.py --collision-axis urdf # spec-correct cylinders (see below)

COLLISION AXIS. A URDF <cylinder> is Z-aligned by spec. threepp's loader honours that for VISUAL
cylinders (URDFLoader.cpp rotates them +PI/2 about X) but NOT for COLLISION ones: a collision
cylinder becomes a threepp CapsuleGeometry, which is Y-aligned, with no correction. So today's
threepp needs the leg capsules authored against Y (`--collision-axis threepp`, the default here) and
any spec-conforming tool needs Z (`--collision-axis urdf`). When the loader is fixed, `urdf` becomes
the correct default and this flag can go. The masses, radii and lengths are identical either way.
"""
import argparse
import math
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOT = os.path.dirname(_HERE)
sys.path.insert(0, _SPOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(_SPOT)))

# spot_deploy imports threepp, whose PhysX DLLs sit beside the built .pyd in the main checkout.
_dll = os.environ.get("THREEPP_DLL_DIR", r"C:\dev\threepp\python\threepp")
if os.path.isdir(_dll):
    os.add_dll_directory(_dll)

from spot_deploy import HIP_X, HIP_Y, HY_Y, KN, FOOT, MASS, LEGS, SIGN   # noqa: E402

HALF_PI = math.pi / 2
# The upper leg runs from the hy joint to the knee: mostly -z, tipped slightly +x.
ULEG_PITCH = -math.atan2(float(KN[0]), float(-KN[2]))


def _inertial(mass, com, ixx, iyy, izz):
    e = ET.Element("inertial")
    ET.SubElement(e, "origin", xyz=" ".join(f"{v:.6f}" for v in com), rpy="0 0 0")
    ET.SubElement(e, "mass", value=f"{mass:.6f}")
    ET.SubElement(e, "inertia", ixx=f"{ixx:.6f}", ixy="0", ixz="0",
                  iyy=f"{iyy:.6f}", iyz="0", izz=f"{izz:.6f}")
    return e


def _collision(xyz, rpy, geom):
    e = ET.Element("collision")
    ET.SubElement(e, "origin", xyz=" ".join(f"{v:.6f}" for v in xyz),
                  rpy=" ".join(f"{v:.9f}" for v in rpy))
    g = ET.SubElement(e, "geometry")
    if geom[0] == "box":
        ET.SubElement(g, "box", size=" ".join(f"{v:.6f}" for v in geom[1]))
    else:
        ET.SubElement(g, "cylinder", radius=f"{geom[1]:.6f}", length=f"{geom[2]:.6f}")
    return e


def _box_inertia(mass, sx, sy, sz):
    k = mass / 12.0
    return k * (sy * sy + sz * sz), k * (sx * sx + sz * sz), k * (sx * sx + sy * sy)


def _capsule_inertia(mass, radius, length):
    """Solid-cylinder approximation, about the capsule's own axis (good enough: threepp's loader
    converts <mass> to a density and lets PhysX compute the real inertia from the shape)."""
    along = 0.5 * mass * radius * radius
    across = mass * (3.0 * radius * radius + length * length) / 12.0
    return across, along, across            # (perpendicular, along-axis, perpendicular)


def data_dir():
    """threepp_data's checkout. THREEPP_DATA_DIR wins; otherwise the usual places."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    here = pathlib.Path(__file__).resolve()
    repo = here.parents[4]                       # <repo>/python/examples/spot/editor_deploy
    for candidate in [repo.parent / "threepp_data",
                      *sorted(repo.glob("cmake-build-*/_deps/threepp_data-src"))]:
        if candidate.is_dir():
            return str(candidate)
    return ""


def main():
    ap = argparse.ArgumentParser()
    spot = os.path.join(data_dir(), "urdf", "spot") if data_dir() else ""
    ap.add_argument("--urdf", default=os.path.join(spot, "model.urdf") if spot else "")
    ap.add_argument("--out", default=os.path.join(spot, "spot_physics.urdf") if spot else "")
    ap.add_argument("--collision-axis", choices=("threepp", "urdf"), default="threepp")
    args = ap.parse_args()
    if not args.urdf or not os.path.exists(args.urdf):
        print(f"no Spot URDF at {args.urdf or '<threepp_data not found>'} - point "
              f"THREEPP_DATA_DIR at your threepp_data checkout (it ships urdf/spot/)")
        return 1

    # A capsule's local axis before <origin rpy> is applied: Y in threepp, Z per URDF spec.
    if args.collision_axis == "threepp":
        along_y, along_z = (0.0, 0.0, 0.0), (HALF_PI, 0.0, 0.0)
        uleg_rpy = (HALF_PI, ULEG_PITCH, 0.0)
    else:
        along_y, along_z = (-HALF_PI, 0.0, 0.0), (0.0, 0.0, 0.0)
        uleg_rpy = (0.0, ULEG_PITCH, 0.0)

    tree = ET.parse(args.urdf)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    touched = []

    # base: the body box, centred on the link origin (as build_spot places it)
    BASE = (0.70, 0.18, 0.19)
    b = links["base"]
    b.append(_inertial(MASS["base"], (0, 0, 0), *_box_inertia(MASS["base"], *BASE)))
    b.append(_collision((0, 0, 0), (0, 0, 0), ("box", BASE)))
    touched.append("base")

    for L in LEGS:
        sy = SIGN[L][1]
        # Each leg link's frame sits at its inbound joint, so a segment's collider is centred
        # halfway along the offset to the NEXT joint — the same midpoints build_spot uses.
        segs = [
            (f"{L}.hip", MASS["hip"], 0.045, 0.06, (0.0, sy * HY_Y / 2.0, 0.0), along_y),
            (f"{L}.uleg", MASS["uleg"], 0.045, 0.30, tuple(KN / 2.0), uleg_rpy),
            (f"{L}.lleg", MASS["lleg"], 0.028, 0.30, tuple(FOOT / 2.0), along_z),
        ]
        for name, mass, radius, length, xyz, rpy in segs:
            link = links[name]
            ia, ib, ic = _capsule_inertia(mass, radius, length)
            # rotate the inertia's "along" axis with the capsule; diagonal either way
            i3 = (ia, ib, ic) if rpy is along_y else (ib, ia, ic)
            link.append(_inertial(mass, xyz, *i3))
            link.append(_collision(xyz, rpy, ("cylinder", radius, length)))
            touched.append(name)

    ET.indent(tree, space="  ")
    # "Modified versions of the SDK Software must be conspicuously marked as such"
    # (Boston Dynamics SDK License 20191101-BDSDK-SL, section 2(e)). This file is a
    # derivative of BD's model.urdf, so it says so in its first lines - not only in a
    # NOTICE next to it, because a URDF gets copied around on its own.
    banner = (
        "<!--\n"
        "  MODIFIED VERSION of the Boston Dynamics Spot URDF.\n"
        "\n"
        "  Original: model.urdf from the Boston Dynamics Spot SDK\n"
        "  (files/spot_base_urdf.zip), Copyright 2021 Boston Dynamics, Inc.\n"
        "  Licensed under the BD Software Development Kit License\n"
        "  (20191101-BDSDK-SL) - see the LICENSE file beside this one. NOT MIT.\n"
        "\n"
        "  Modifications by the threepp project, generated by\n"
        "  python/examples/spot/editor_deploy/make_spot_urdf.py:\n"
        "    * <inertial> added to all 13 links (per-link masses, 28.0 kg total)\n"
        "    * <collision> added to all 13 links (box body, capsule leg segments)\n"
        "  Nothing else is touched: visuals, joint origins, axes and limits are\n"
        "  the original's. The added values come from spot_deploy.build_spot, so\n"
        "  this file describes the plant threepp's Spot policies were trained on.\n"
        "\n"
        "  Used here for software-based simulation only, per section 2(c) of that\n"
        "  license. Not for use with hardware.\n"
        "-->\n")
    body = ET.tostring(root, encoding="unicode")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(banner)
        f.write(body)
        f.write("\n")
    total = MASS["base"] + 4 * (MASS["hip"] + MASS["uleg"] + MASS["lleg"])
    print(f"[urdf] {args.out}")
    print(f"  {len(touched)} links given <inertial> + <collision>   total mass {total:.1f} kg")
    print(f"  collision axis: {args.collision_axis}"
          f"{'  (threepp capsules are Y-aligned; see the module docstring)' if args.collision_axis == 'threepp' else '  (spec-correct Z cylinders)'}")
    print(f"  uleg pitch {ULEG_PITCH:+.6f} rad  (knee offset {tuple(KN)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

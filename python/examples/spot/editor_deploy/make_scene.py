"""Write an editor scene that runs the policy bundle on Spot: ground, robot, articulation, script.

    python make_scene.py                    # -> spot_policy.json beside this file

Emitted rather than hand-authored so every number in it is derived from the training contract:
the robot's spawn pose, its default joint values, and the PD gains all come from spot_deploy, and
the URDF reference points at the physics-augmented file make_spot_urdf.py writes.

Two frame details worth naming, because they are what a hand-authored scene gets wrong:

  * The editor's PhysX world is Y-UP; the URDF is Z-up. The robot node therefore carries a -90°
    rotation about X, which sends body +z (the robot's up) to world +y and leaves body +x (forward)
    pointing along world +x. The policy is indifferent to this — the runner derives up from gravity.
  * The base spawns at the default pose's natural stand height plus a few centimetres, so the feet
    settle onto the ground instead of starting inside it.
"""
import argparse
import json
import math
import os
import pathlib
import sys
import uuid as uuidlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOT = os.path.dirname(_HERE)
sys.path.insert(0, _SPOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(_SPOT)))

_dll = os.environ.get("THREEPP_DLL_DIR", r"C:\dev\threepp\python\threepp")
if os.path.isdir(_dll):
    os.add_dll_directory(_dll)

from spot_deploy import DEFAULT, LEGS, STIFF_GAINS   # noqa: E402
from spot_terrain_env import SPAWN_Z                 # noqa: E402

# The URDF's own joint document order (fl.hx, fl.hy, fl.kn, fr.hx, ...). `jointValues` is written
# in the robot's joint-table order; the SCRIPT maps by name, so this only sets the authored pose.
URDF_ORDER = [f"{L}.{j}" for L in LEGS for j in ("hx", "hy", "kn")]


def uid(tag):
    return str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, "threepp-spot-policy/" + tag))


def main():
    ap = argparse.ArgumentParser()
    cache = pathlib.Path.home() / ".cache" / "threepp" / "spot"
    ap.add_argument("--urdf", default=str(cache / "spot_physics.urdf"))
    ap.add_argument("--bundle", default=os.path.join(_HERE, "bundle_spot_steps"))
    ap.add_argument("--script", default=os.path.join(_HERE, "spot_editor_script.py"))
    ap.add_argument("--log", default=os.path.join(_HERE, "spot_editor_trace.csv"))
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_policy.json"))
    ap.add_argument("--vx", type=float, default=0.8)
    ap.add_argument("--settle", type=int, default=100)
    # A play session cannot be steered by hand (the inspector is read-only while playing and
    # scripts have no input access), so the command sequence is authored instead.
    ap.add_argument("--keys", type=int, default=1,
                    help="1 = drive it with the keyboard (arrows/WASD), 0 = follow --route")
    # '|'-separated, not ';': scriptFields is itself a flat key=value; string.
    ap.add_argument("--route", default="5:0.8,0,0|4:0.5,0,0.5|4:0.8,0,0|3:0,0.5,0|4:1.2,0,0",
                    help="seconds:vx,vy,wz segments, '|'-separated. Empty = hold vx forever.")
    args = ap.parse_args()
    if not os.path.exists(args.urdf):
        print(f"no physics URDF at {args.urdf} - run make_spot_urdf.py first")
        return 1

    fwd = lambda p: str(p).replace("\\", "/")
    spawn = SPAWN_Z + 0.03                      # natural stand height + a small drop

    # -90 deg about X, column-major: body +z -> world +y, body +x stays world +x
    robot_matrix = [1, 0, 0, 0,
                    0, 0, -1, 0,
                    0, 1, 0, 0,
                    0, spawn, 0, 1]
    joint_values = ",".join(f"{DEFAULT[n.replace('.', '_')]:g}" for n in URDF_ORDER)
    articulation = (f"fixedbase=0;stiffness={STIFF_GAINS['hx'][0]:g};damping={STIFF_GAINS['hx'][1]:g};"
                    f"maxforce={STIFF_GAINS['kn'][2]:g};selfcollision=0;iterations=12;density=1000")
    script_fields = (f"bundle={fwd(args.bundle)};vx={args.vx:g};vy=0;wz=0;"
                     f"settle={args.settle};log={fwd(args.log)};"
                     f"chase=Chase;keys={args.keys};route={args.route}")

    doc = {
        "metadata": {"version": 4.5, "type": "Object", "generator": "make_scene.py"},
        "geometries": [
            # 120 m of ground, not 400: a directional light's shadow camera is an ORTHO box, so
            # the ground has to stay within a shadow frustum you can actually afford. A huge
            # receiveShadow floor outside that box samples the shadow map's clamped edge and
            # reads as shadowed almost everywhere — which looks exactly like "the scene is dark".
            {"uuid": uid("geo-ground"), "type": "BoxGeometry",
             "width": 120, "height": 1, "depth": 120},
        ],
        "materials": [
            {"uuid": uid("mat-ground"), "type": "MeshStandardMaterial",
             "color": 0x8a8f96, "roughness": 0.95, "metalness": 0.0},
        ],
        "object": {
            "uuid": uid("scene"), "type": "Scene", "name": "Scene",
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "children": [
                {"uuid": uid("ground"), "type": "Mesh", "name": "Ground",
                 "geometry": uid("geo-ground"), "material": uid("mat-ground"),
                 "receiveShadow": True,
                 "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -0.5, 0, 1],
                 # Grippy, non-bouncy ground: the trained plant used restitution-0 feet.
                 "userData": {"physics": "body=static;shape=box;friction=1;restitution=0"}},
                {"uuid": uid("sun"), "type": "DirectionalLight", "name": "Sun",
                 "color": 0xffffff, "intensity": 2.6, "castShadow": True,
                 "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 6, 11, 5, 1],
                 # The default shadow camera is +-5 m, which would leave everything the robot
                 # walks to unshadowed-or-worse. Cover the whole route explicitly.
                 "shadow": {"mapSize": [2048, 2048], "bias": -0.0005,
                            "camera": {"left": -18, "right": 18, "top": 18, "bottom": -18,
                                       "near": 0.5, "far": 60}}},
                # A HemisphereLight's POSITION is its sky direction, not a location: the renderer
                # takes the normalized world position as the axis to blend sky->ground along
                # (Lights.cpp, setFromMatrixPosition + normalize). At the origin that axis is
                # zero, every surface gets the flat 50/50 average instead of full sky, and the
                # scene reads as dark. threepp's constructor defaults it to (0,1,0) for exactly
                # this reason - authoring an identity matrix throws that default away.
                {"uuid": uid("hemi"), "type": "HemisphereLight", "name": "Sky",
                 "color": 0xd0e4f7, "groundColor": 0x4a5a6a, "intensity": 1.15,
                 "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]},
                # Driven by the script (its `chase` parameter): select it and the camera dock
                # shows the trot from behind. Nothing follows it otherwise — a camera in a
                # document is just an object.
                {"uuid": uid("cam"), "type": "PerspectiveCamera", "name": "Chase",
                 "fov": 46, "aspect": 1.0, "near": 0.05, "far": 200,
                 "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -3.0, 1.5, 0, 1]},
                {"uuid": uid("spot"), "type": "Object3D", "name": "Spot",
                 "matrix": robot_matrix,
                 "userData": {
                     "urdf": fwd(args.urdf),
                     "jointValues": joint_values,
                     "articulation": articulation,
                     "script": fwd(args.script),
                     "scriptFields": script_fields,
                 }},
            ],
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[scene] {args.out}")
    print(f"  urdf    {args.urdf}")
    print(f"  spawn   base at y={spawn:.2f} (Z-up robot rotated -90 deg about X into the Y-up world)")
    print(f"  joints  {joint_values}")
    print(f"  articulation  {articulation}")
    print(f"  script  {os.path.basename(args.script)}  [{script_fields}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

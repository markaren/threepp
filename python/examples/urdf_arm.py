"""Import any URDF as a PhysX articulation and wave its joints.

    python urdf_arm.py --urdf <robot.urdf>            # interactive (orbit)
    python urdf_arm.py --urdf <robot.urdf> --shot out.png

Uses threepp.urdf.load_articulation to turn a URDF into a simulatable
reduced-coordinate articulation (collision via primitives / bounding boxes,
revolute+prismatic joints with PD drives), then sweeps every joint with a
phase-shifted sine so you can see it move. Best with a fixed-base arm. Needs a
PhysX-enabled threepp build (`tp.HAS_PHYSX`).
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threepp as tp
from threepp.urdf import load_articulation

PALETTE = [0xd94f3a, 0xe9e9ee, 0xf2a93b, 0x3f74e0, 0x29b06a, 0xb05cc8, 0x33c4c0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="path to a .urdf file")
    ap.add_argument("--shot", metavar="PNG", help="render headless and save")
    ap.add_argument("--free", action="store_true", help="free-floating base (default: fixed)")
    args = ap.parse_args()
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"

    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    robot = load_articulation(world, args.urdf, fixed_base=not args.free, base_position=(0, 0, 0.05),
                              default_density=1200, stiffness=400, damping=40, max_force=400)
    print(f"{robot.num_dof} DOF: {robot.joint_names}")
    for _ in range(40):                     # settle at the zero pose
        robot.set_targets(np.zeros(robot.num_dof, np.float32)); world.step(0.02)

    headless = bool(args.shot)
    canvas = tp.Canvas("threepp - URDF arm", width=820, height=820, antialiasing=4, headless=headless)
    rend = tp.GLRenderer(canvas); rend.set_clear_color(0x0e1118)
    rend.shadow_map_enabled = True; rend.tone_mapping = tp.ToneMapping.ACESFilmic
    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xbcd0ff, 0x20242c, 0.95))
    key = tp.DirectionalLight(0xffffff, 2.6); key.position.set(3, -3, 7); key.cast_shadow = True; scene.add(key)
    floor = tp.Mesh(tp.PlaneGeometry(20, 20), tp.MeshStandardMaterial())
    floor.material.color = 0x171b24; floor.material.roughness = 0.95; floor.receive_shadow = True
    scene.add(floor)
    for i, m in enumerate(robot.meshes):
        m.material.color = PALETTE[i % len(PALETTE)]; m.material.roughness = 0.5; m.cast_shadow = True
        scene.add(m)

    camera = tp.PerspectiveCamera(45, 1.0, 0.01, 100); camera.up.set(0, 0, 1)
    camera.position.set(2.4, -2.4, 1.5); camera.look_at(0, 0, 0.6)
    amp = np.array([0.9 if "prismatic" not in n else 0.1 for n in robot.joint_names], np.float32)
    phase = np.linspace(0, math.pi, robot.num_dof, dtype=np.float32)
    clock = tp.Clock()

    def step_sweep():
        t = clock.get_elapsed_time()
        robot.set_targets(amp * np.sin(0.8 * t + phase))
        world.step(0.02)

    if headless:
        for _ in range(70):
            step_sweep()
        rend.render(scene, camera); rend.save_frame(args.shot); print(f"saved {args.shot}")
        return

    controls = tp.OrbitControls(camera, canvas); controls.enable_damping = True
    controls.target = tp.Vector3(0, 0, 0.6)

    def on_resize(w, h):
        camera.aspect = w / max(h, 1); camera.update_projection_matrix(); rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    def frame():
        step_sweep(); controls.update(); rend.render(scene, camera)

    canvas.animate(frame)


if __name__ == "__main__":
    main()

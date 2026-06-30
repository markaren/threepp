"""play_kuka_grasp.py — watch a trained KUKA grasp policy reach, grip, and lift cubes.

    python play_kuka_grasp.py                              # latest checkpoint, interactive
    python play_kuka_grasp.py --model kuka_grasp_best.pt   # a specific checkpoint
    python play_kuka_grasp.py --shot out.png               # headless: render a few grasps -> PNG

Runs the SAME KukaGraspEnv used for training (K=1) so the observation is identical to what the
policy was trained on (the #1 deploy gotcha) — the only extra work is syncing the render meshes
from the sim's GPU link/cube poses each frame. The grasp-lock assist is OFF here: you see the real
friction grasp. The cube auto-respawns each episode.
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import threepp as tp  # noqa: E402

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_env import KukaGraspEnv  # noqa: E402


def _sync_meshes(env):
    """Copy the sim's GPU-read link/cube world poses onto the render meshes (env 0)."""
    lp = env.sim.link_pose[0].cpu().numpy()          # [L, 7] = [qx,qy,qz,qw,px,py,pz]
    for i, m in enumerate(env.sim.arm_meshes[0]):
        q = lp[i]
        m.position.set(float(q[4]), float(q[5]), float(q[6]))
        m.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    cp = env.sim.cube_pose[0].cpu().numpy()
    cm = env.sim.cube_meshes[0]
    cm.position.set(float(cp[4]), float(cp[5]), float(cp[6]))
    cm.quaternion.set(float(cp[0]), float(cp[1]), float(cp[2]), float(cp[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "kuka_grasp_best.pt"))
    ap.add_argument("--shot", metavar="PNG", help="render headless and save a frame after a few grasps")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from threepp.rl import load_policy

    if not os.path.exists(args.model):
        # fall back to any checkpoint that exists
        for alt in ("kuka_grasp_latest.pt", "kuka_grasp.pt"):
            p = os.path.join(_HERE, alt)
            if os.path.exists(p):
                args.model = p
                break
    print(f"[play] policy: {args.model}")
    ac, norm, meta = load_policy(args.model, device=args.device)

    env = KukaGraspEnv(num_envs=1, device=args.device, render_visuals=True)
    env.set_iter(10_000)            # assist off, full spawn region — the real task
    obs = env.reset()

    # ---- scene ----
    scene = tp.Scene()
    scene.background = tp.Color(0x10151c) if hasattr(tp, "Color") else 0x10151c
    scene.add(tp.HemisphereLight(0xbfd8ff, 0x223044, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(2.5, -2.0, 4.0)
    scene.add(sun)

    floor = tp.Mesh(tp.BoxGeometry(6.0, 6.0, 0.02), tp.MeshStandardMaterial())
    floor.material.color = 0x2b2f37
    floor.position.set(0.0, 0.0, -0.01)
    scene.add(floor)
    table = tp.Mesh(tp.BoxGeometry(C.TABLE_HALF[0] * 2, C.TABLE_HALF[1] * 2, C.TABLE_THICK),
                    tp.MeshStandardMaterial())
    table.material.color = 0x3b4252
    table.position.set(C.TABLE_CX, C.TABLE_CY, C.TABLE_TOP_Z - C.TABLE_THICK / 2)
    scene.add(table)

    for m in env.sim.arm_meshes[0]:
        scene.add(m)
    scene.add(env.sim.cube_meshes[0])

    @torch.no_grad()
    def policy_step():
        nonlocal obs
        a = ac.act_mean(norm.norm(obs) if norm is not None else obs)
        obs, _, _, _, _ = env.step(a)
        _sync_meshes(env)

    headless = bool(args.shot)
    canvas = tp.Canvas("threepp - KUKA grasp", width=1100, height=720, antialiasing=4, headless=headless)
    rend = tp.GLRenderer(canvas)
    rend.shadow_map_enabled = True

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.02, 50.0)
    camera.up.set(0, 0, 1)                       # Z-up world
    camera.position.set(1.25, -1.05, 0.95)
    camera.look_at(0.45, 0.0, 0.45)

    _sync_meshes(env)

    if headless:
        for _ in range(C.MAX_STEPS):             # one full episode
            policy_step()
        rend.render(scene, camera)
        rend.save_frame(args.shot)
        print(f"[play] saved {args.shot}  (cube z={float(env.sim.cube_position()[0,2]):.3f})")
        return

    controls = tp.OrbitControls(camera, canvas) if hasattr(tp, "OrbitControls") else None
    if controls is not None:
        controls.target.set(0.45, 0.0, 0.45)

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    def do_reset():
        nonlocal obs
        obs = env.reset()
        _sync_meshes(env)

    print("[play] keys:  R = reset episode (respawn cube),  drag = orbit camera")
    reset_held = [False]

    def frame():
        # manual reset on R (debounced); the env also auto-resets on timeout / lost cube
        if hasattr(canvas, "is_key_down") and canvas.is_key_down("R"):
            if not reset_held[0]:
                do_reset()
                reset_held[0] = True
        else:
            reset_held[0] = False
        policy_step()
        if controls is not None:
            controls.update()
        rend.render(scene, camera)
    canvas.animate(frame)


if __name__ == "__main__":
    main()

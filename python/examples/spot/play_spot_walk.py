"""Watch Spot walk with the FROM-SCRATCH policy (single Spot, CPU deploy + render).

    python play_spot_walk.py                  # live window (chase cam); hot-reloads the latest checkpoint
    python play_spot_walk.py --shot 8         # headless montage -> spot_walk_shot.png

The policy directly drives the 12 joints (no Isaac walker). CPU world so the visual meshes follow
the sim; reuses spot_walk_env's ACTION_SCALE + obs layout so train and deploy can't drift. R resets.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets, grid_texture, _write_png,
                         _quat_to_R, default_q, add_to_isaac, Z0)
import spot_walk_env as W

DISP = 760
STAND_Q = default_q[add_to_isaac].astype(np.float32)      # add-order default stance
GRAV = np.array([0.0, 0.0, -1.0])


def walk_obs(art, last_a, cmd, phase, with_phase):
    rs, rv = art.root_state(), art.root_velocity()
    Rt = _quat_to_R(rs[3:7]).T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV
    qpos = art.joint_positions() - STAND_Q
    qvel = art.joint_velocities()
    obs = np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, qvel, last_a])
    if with_phase:                                       # 50-d gait-phase policy: append the trot clock
        ph = phase * 2.0 * math.pi
        obs = np.concatenate([obs, [math.sin(ph), math.cos(ph)]])
    return obs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "spot_gait_best.pt"))
    ap.add_argument("--vx", type=float, default=0.5, help="forward speed command (m/s)")
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    if not os.path.exists(args.model):
        print("No policy — run train_spot_walk.py first."); sys.exit(0)
    live = args.shot == 0 and not args.frames
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ac, _, meta = load_policy(args.model, device=dev)
    with_phase = meta.get("obs_dim", 48) == 50              # 50-d = gait-phase policy -> append the trot clock
    pol = {"ac": ac, "mt": os.path.getmtime(args.model)}

    # match the GPU training sim: SUBSTEPS=6 AND the TGS+PCM contact model (a plain CPU world
    # defaults to PGS/no-PCM, a different effective friction + penetration recovery than the
    # GpuSim the policy trained on — tgs_pcm=True closes that sim-to-sim solver gap).
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.02 / 6, max_substeps=10, tgs_pcm=True)
    ground = tp.Mesh(tp.BoxGeometry(400, 400, 1.0), tp.MeshStandardMaterial()); ground.position.set(0, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, fetch_assets(), gains=W.WALK_GAINS)   # real Spot visuals; stiff gains match training

    canvas = tp.Canvas("spot-walk (from scratch)", width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(400, 400), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(400 / 1.5); floor.material.color = 0xffffff
    floor.material.roughness = 0.92; floor.receive_shadow = True; scene.add(floor)
    for m in meshes:
        scene.add(m)
    cam = tp.PerspectiveCamera(48, 1.0, 0.05, 200); cam.up.set(0, 0, 1)

    cmd = np.array([args.vx, 0.0, 0.0], np.float32)
    last_a = np.zeros(12, np.float32)
    phase = 0.0                                             # gait-phase clock (advanced each control tick)

    def init_stance():
        # Drop from the straight-leg build height and settle to the natural stance, THEN wait until the
        # base is at rest so the policy starts from a clean, still, upright stand (the old fixed 150-tick
        # settle left it drifting at ~0.06 m/s). On the CPU solver this drop is the robust init:
        # teleporting straight to the stance pose at SPAWN_Z instead COLLAPSES (feet penetrate and splay) —
        # a sign of how different the CPU contact model is from the GPU sim the policy trained on.
        art.reset(tp.Vector3(float(art.root_state()[0]), 0.0, Z0))
        for t in range(300):
            art.set_drive_targets(STAND_Q); world.step(0.02)
            if t >= 130 and np.linalg.norm(art.root_velocity()[0:3]) < 0.04:   # past the bounce + at rest
                break

    init_stance()

    def control_tick():
        nonlocal last_a, phase
        R = _quat_to_R(art.root_state()[3:7])                   # heading-hold: steer back toward straight (+x),
        cmd[2] = float(np.clip(-1.5 * math.atan2(R[1, 0], R[0, 0]), -1.0, 1.0))   # matching the training command
        with torch.no_grad():
            ra = pol["ac"].act_mean(torch.from_numpy(walk_obs(art, last_a, cmd, phase, with_phase))[None].to(dev)).clamp(-1, 1)[0].cpu().numpy()
        last_a = ra
        art.set_drive_targets((STAND_Q + W.ACTION_SCALE * ra).astype(np.float32))
        world.step(0.02)
        phase = (phase + W.GAIT_FREQ * 0.02) % 1.0          # advance the trot clock (no-op for 48-d policies)

    def render_chase():
        rs = art.root_state(); p = rs[0:3]; R = _quat_to_R(rs[3:7]); fwd = R[:, 0]
        eye = p - fwd * 2.6 + np.array([0, 0, 1.2])
        cam.position.set(float(eye[0]), float(eye[1]), float(eye[2]))
        cam.look_at(float(p[0] + fwd[0]), float(p[1] + fwd[1]), float(p[2] + 0.05))
        rend.render(scene, cam)

    def reset():
        nonlocal last_a, phase
        init_stance(); last_a = np.zeros(12, np.float32); phase = 0.0
        print("⟳ reset")

    if live:
        def on_resize(w, h):
            cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
        canvas.on_window_resize(on_resize)
        nf = [0]; r_down = [False]

        def frame():
            nonlocal last_a
            if canvas.is_key_down("R"):
                if not r_down[0]:
                   reset()
                r_down[0] = True
            else:
                r_down[0] = False
            nf[0] += 1
            if nf[0] % 40 == 0:
                try:
                    mt = os.path.getmtime(args.model)
                    if mt != pol["mt"]:
                        pol["ac"], _, _ = load_policy(args.model, device=dev); pol["mt"] = mt
                        print("↻ reloaded latest policy")
                        reset()
                except Exception:
                    pass
            control_tick(); render_chase()
        print(__doc__ + "\n(R = reset; hot-reloads the policy as the trainer checkpoints it)")
        canvas.animate(frame)
        return

    if args.frames:
        os.makedirs(args.frames, exist_ok=True)
    shots = []
    spacing = max(1, args.steps // max(args.shot, 1))
    for i in range(args.steps):
        control_tick()
        if args.frames:
            render_chase(); _write_png(os.path.join(args.frames, f"f{i:04d}.png"), rend.read_pixels(True))
        elif i % spacing == 0 and len(shots) < args.shot:
            render_chase(); shots.append(rend.read_pixels(True))
    print(f"traveled to x={art.root_state()[0]:.2f} m, base_z={art.root_state()[2]:.2f}")
    if not args.frames and shots:
        cols = 4
        rows = [np.concatenate(shots[r:r + cols], axis=1) for r in range(0, len(shots), cols)]
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=20) for r in rows]
        out = os.path.join(_HERE, "spot_walk_shot.png")
        _write_png(out, np.ascontiguousarray(np.concatenate(rows, axis=0))); print("saved ->", out)


if __name__ == "__main__":
    main()

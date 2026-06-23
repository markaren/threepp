"""Watch the trained residual policy climb stairs (single Spot, CPU deploy + render).

    python play_spot_stairs.py --rise 0.10           # live window
    python play_spot_stairs.py --rise 0.12 --shot 8  # headless montage -> spot_stairs_shot.png

One Spot on a normal CPU PhysX world (so the bound visual meshes follow the sim and render) climbs
a staircase, driven by the FROZEN Isaac walker + the trained per-joint residual — the same control
stack trained on GpuSim. Reuses spot_stairs_env's constants + obs so train and deploy can't drift.
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
from spot_deploy import (build_spot, fetch_assets, grid_texture, _write_png, _quat_to_R,
                         default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, Z0, SpotController)
import spot_stairs_env as S

DISP = 760
DQ_ADD = default_q[add_to_isaac]                   # standing joints, add-order
GRAV = np.array([0.0, 0.0, -1.0])


def terr_h(x, rise, run):
    return float(np.clip(math.floor((x - S.STAIR_X0) / run) + 1.0, 0.0, S.STAIR_N) * rise)


def residual_obs(art, last_res, rise, run):
    rs, rv = art.root_state(), art.root_velocity()
    Rt = _quat_to_R(rs[3:7]).T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV
    qpos = art.joint_positions() - DQ_ADD
    qvel = art.joint_velocities()
    x, z = float(rs[0]), float(rs[2])
    h = terr_h(x, rise, run)
    ahead = np.array([terr_h(x + dx, rise, run) - h for dx in S.PROBE_DX], np.float32)
    return np.concatenate([lin_b, ang_b, proj_g, qpos, qvel, last_res,
                           [z - h], ahead]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "spot_stairs_policy.pt"))
    ap.add_argument("--rise", type=float, default=0.10)
    ap.add_argument("--run", type=float, default=0.32)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    if not os.path.exists(args.model):
        print("No policy — run train_spot_stairs.py first."); sys.exit(0)
    live = args.shot == 0 and not args.frames
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ac, _, _ = load_policy(args.model, device=dev)
    pol = {"ac": ac, "mt": os.path.getmtime(args.model)}   # holder so --live can hot-reload the latest checkpoint

    assets = fetch_assets()
    walker_policy = torch.jit.load(os.path.join(assets, "spot_policy.pt"), map_location="cpu").eval()
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.005, max_substeps=8)  # match GPU training substep
    ground = tp.Mesh(tp.BoxGeometry(60, 6, 1.0), tp.MeshStandardMaterial()); ground.position.set(20, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, assets)
    ctrl = SpotController(art, walker_policy)

    # scene
    canvas = tp.Canvas("spot-stairs", width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(60, 6), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(20); floor.material.color = 0xffffff
    floor.material.roughness = 0.95; floor.receive_shadow = True; scene.add(floor)
    for s in range(S.STAIR_N):                                   # the staircase (matches terr_h)
        h = (s + 1) * args.rise
        step = tp.Mesh(tp.BoxGeometry(args.run, 2.4, h), tp.MeshStandardMaterial())
        step.material.color = 0x8a96a6; step.material.roughness = 0.9
        step.position.set(S.STAIR_X0 + s * args.run + args.run * 0.5, 0.0, h * 0.5)
        step.receive_shadow = True
        world.add_static(step)                                  # physics collider (was missing!)
        scene.add(step)
    for m in meshes:
        scene.add(m)

    cam = tp.PerspectiveCamera(46, 1.0, 0.05, 120); cam.up.set(0, 0, 1)
    last_res = np.zeros(12, np.float32)

    ctrl.hold(world, 120)                                        # stand up before climbing

    def control_tick():
        nonlocal last_res
        rs0 = art.root_state(); qq = rs0[3:7]                    # heading + lane hold (drive straight at the stairs)
        yaw = math.atan2(2 * (qq[0] * qq[1] + qq[2] * qq[3]), 1 - 2 * (qq[1] ** 2 + qq[2] ** 2))
        cmd = np.array([S.FWD_CMD, float(np.clip(-rs0[1], -0.4, 0.4)),
                        float(np.clip(-1.5 * yaw, -1.0, 1.0))], np.float32)
        wobs = ctrl.obs(cmd)
        with torch.no_grad():
            wa = ctrl.policy(torch.from_numpy(wobs)[None]).numpy()[0]
        ctrl.last_action = wa
        w_targets = (default_q + ACTION_SCALE * wa)[add_to_isaac]
        with torch.no_grad():
            ra = pol["ac"].act_mean(torch.from_numpy(residual_obs(art, last_res, args.rise, args.run))[None].to(dev))
        ra = ra.clamp(-1, 1)[0].cpu().numpy()
        last_res = ra
        art.set_drive_targets((w_targets + S.RESIDUAL_SCALE * ra).astype(np.float32))
        world.step(0.02)

    def render_chase():
        rs = art.root_state(); p = rs[0:3]
        cam.position.set(float(p[0] - 2.6), -2.7, float(p[2] + 1.2))
        cam.look_at(float(p[0] + 0.4), 0.0, float(p[2] + 0.05))
        rend.render(scene, cam)

    if live:
        def on_resize(w, h):
            cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
        canvas.on_window_resize(on_resize)
        nf = [0]

        def maybe_reload():
            nf[0] += 1
            if nf[0] % 40:                              # check the checkpoint ~once a second
                return
            try:
                mt = os.path.getmtime(args.model)
                if mt != pol["mt"]:
                    pol["ac"], _, _ = load_policy(args.model, device=dev)
                    pol["mt"] = mt
                    print("↻ reloaded latest policy")
            except Exception:                           # mid-write checkpoint -> try again next tick
                pass
        r_down = [False]

        def frame():
            nonlocal last_res
            if canvas.is_key_down("R"):
                if not r_down[0]:                       # edge-triggered: reset once per press
                    art.reset(tp.Vector3(0, 0, Z0)); ctrl.last_action[:] = 0.0
                    last_res = np.zeros(12, np.float32); ctrl.hold(world, 40)
                    print("⟳ reset Spot to the bottom")
                r_down[0] = True
            else:
                r_down[0] = False
            maybe_reload(); control_tick(); render_chase()

        print(__doc__ + "\n(R = reset Spot to the bottom; hot-reloads the policy as the trainer checkpoints it)")
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
    bx = float(art.root_state()[0]); step_h = terr_h(bx, args.rise, args.run)
    print(f"base over step height {step_h:.2f} m (~{step_h / args.rise:.0f} steps of {args.rise} m); base_z={art.root_state()[2]:.2f}")
    if not args.frames and shots:
        cols = 4
        rows = [np.concatenate(shots[r:r + cols], axis=1) for r in range(0, len(shots), cols)]
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=20) for r in rows]
        out = os.path.join(_HERE, "spot_stairs_shot.png")
        _write_png(out, np.ascontiguousarray(np.concatenate(rows, axis=0))); print("saved montage ->", out)


if __name__ == "__main__":
    main()

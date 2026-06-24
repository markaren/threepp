"""Watch the FINE-TUNED Isaac walker climb stairs (single Spot, CPU deploy + render), hot-reloading
the checkpoint as train_spot_stairs_ft.py writes it.

    python play_spot_stairs_ft.py --rise 0.10           # live window (chase cam) — hot-reloads
    python play_spot_stairs_ft.py --rise 0.12 --shot 8  # headless montage -> spot_stairs_ft_shot.png

Unlike play_spot_stairs.py (frozen walker + residual), this drives the joints with the fine-tuned
policy's OWN action (targets = default_q + ACTION_SCALE*a, isaac order, UNclamped ~+-8) and feeds the
58-d obs (48 Isaac proprio + base height + 9-probe terrain scan) on RAW obs (no normalizer) — matching
SpotStairFTEnv exactly so train and deploy can't drift. Policy runs on CPU to leave the GPU to training.
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
                         default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, Z0)
import spot_stairs_env as S

DISP = 760
GRAV = np.array([0.0, 0.0, -1.0])


def terr_h(x, rise, run, n):
    return float(np.clip(math.floor((x - S.STAIR_X0) / run) + 1.0, 0.0, n) * rise)


def ft_obs(art, last_act, rise, run, n, cmd):
    """The 58-d FT observation, single-robot — byte-for-byte the layout SpotStairFTEnv._obs builds.
    `n` is the live step count (the slider) -> the height scan reflects the staircase you're watching."""
    rs, rv = art.root_state(), art.root_velocity()
    Rt = _quat_to_R(rs[3:7]).T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV
    jp_isaac = art.joint_positions()[isaac_to_add]                 # add-order sim -> isaac order
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q                                    # default_q is isaac order
    x, z = float(rs[0]), float(rs[2])
    h = terr_h(x, rise, run, n)
    ahead = np.array([terr_h(x + dx, rise, run, n) - h for dx in S.PROBE_DX], np.float32)
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, last_act,
                           [z - h], ahead]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "spot_stairs_ft.pt"))
    ap.add_argument("--rise", type=float, default=0.10)
    ap.add_argument("--run", type=float, default=0.32)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--steps", type=int, default=700)
    args = ap.parse_args()
    if not os.path.exists(args.model):
        print("No policy yet — run train_spot_stairs_ft.py first."); sys.exit(0)
    live = args.shot == 0 and not args.frames
    dev = "cpu"                                                    # leave the GPU to the trainer
    ac, _, _ = load_policy(args.model, device=dev)
    pol = {"ac": ac, "mt": os.path.getmtime(args.model)}

    # tgs_pcm=True: the policy trained on the GPU TGS+PCM solver -> match its contact model on CPU.
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.005, max_substeps=8, tgs_pcm=True)
    ground = tp.Mesh(tp.BoxGeometry(60, 6, 1.0), tp.MeshStandardMaterial()); ground.position.set(20, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, fetch_assets())               # default Isaac gains (no override)

    canvas = tp.Canvas("spot-stairs (fine-tuned)", width=DISP, height=DISP, headless=not live)
    rend = tp.VulkanRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(60, 6), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(20); floor.material.color = 0xffffff
    floor.material.roughness = 0.95; floor.receive_shadow = True; scene.add(floor)
    # CONFIGURABLE STAIRCASE — rebuilt on demand via world.remove() (the rise / # steps sliders). cfg holds
    # the slider targets; `built` is what's physically in the world, and the obs reads `built` so it always
    # matches the colliders the robot is on. Rebuilt DEBOUNCED (after the slider settles) to avoid churning
    # Vulkan geometry every drag-frame.
    N_MAX = 20
    cfg = {"rise": args.rise, "run": args.run, "n": min(S.STAIR_N, N_MAX)}
    built = {"rise": None, "run": None, "n": None}
    stairs = {"bodies": [], "meshes": []}

    def _stair_box(w, h, x):
        m = tp.Mesh(tp.BoxGeometry(w, 2.4, h), tp.MeshStandardMaterial())
        m.material.color = 0x8a96a6; m.material.roughness = 0.9; m.receive_shadow = True
        m.position.set(x, 0.0, h * 0.5)
        scene.add(m); stairs["meshes"].append(m)
        stairs["bodies"].append(world.add_static(m))

    def rebuild_stairs():
        for b in stairs["bodies"]:
            world.remove(b)                                       # release the old colliders (now possible)
        for m in stairs["meshes"]:
            scene.remove(m)
        stairs["bodies"].clear(); stairs["meshes"].clear()
        rise, run, n = cfg["rise"], cfg["run"], int(cfg["n"])
        for s in range(n):
            _stair_box(run, (s + 1) * rise, S.STAIR_X0 + s * run + run * 0.5)
        _stair_box(4.0, n * rise, S.STAIR_X0 + n * run + 2.0)     # flat landing at the top
        built.update(rise=rise, run=run, n=n)

    rebuild_stairs()
    for m in meshes:
        scene.add(m)

    cam = tp.PerspectiveCamera(46, 1.0, 0.05, 120); cam.up.set(0, 0, 1)
    last_act = np.zeros(12, np.float32)

    def settle(n):                                                # stand up before climbing
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32)); world.step(0.02)
    settle(120)

    def control_tick():
        nonlocal last_act
        rs0 = art.root_state(); qq = rs0[3:7]                     # heading + lane hold -> drive straight at the stairs
        yaw = math.atan2(2 * (qq[0] * qq[1] + qq[2] * qq[3]), 1 - 2 * (qq[1] ** 2 + qq[2] ** 2))
        cmd = np.array([S.FWD_CMD, float(np.clip(-rs0[1], -0.4, 0.4)),
                        float(np.clip(-1.5 * yaw, -1.0, 1.0))], np.float32)
        with torch.no_grad():
            a = pol["ac"].act_mean(torch.from_numpy(ft_obs(art, last_act, built["rise"], built["run"], built["n"], cmd))[None])[0].numpy()
        last_act = a                                              # NO clamp: isaac action range is ~+-8
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)

    def render_chase():
        rs = art.root_state(); p = rs[0:3]
        cam.position.set(float(p[0] - 2.6), -2.7, float(p[2] + 1.2))
        cam.look_at(float(p[0] + 0.4), 0.0, float(p[2] + 0.05))
        rend.render(scene, cam)

    if live:
        ui = tp.ImguiContext(canvas, rend)                       # create AFTER the renderer
        def on_resize(w, h):
            cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
        canvas.on_window_resize(on_resize)
        nf = [0]; r_down = [False]
        prev = {"rise": cfg["rise"], "n": int(cfg["n"])}; rb_wait = [0]   # debounce the staircase rebuild

        def draw_ui():
            tp.imgui.set_next_window_pos(12, 12)
            tp.imgui.set_next_window_size(260, 0)
            tp.imgui.begin("staircase")
            _, cfg["rise"] = tp.imgui.slider_float("rise (m)", cfg["rise"], 0.04, 0.30)
            _, cfg["n"] = tp.imgui.slider_int("# steps", int(cfg["n"]), 1, N_MAX)
            tp.imgui.text(f"top of flight: {int(cfg['n']) * cfg['rise']:.2f} m")
            if cfg["rise"] != built["rise"] or int(cfg["n"]) != built["n"]:
                tp.imgui.text("rebuilding on release...")
            tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
            tp.imgui.end()

        def reset_spot():
            art.reset(tp.Vector3(0, 0, Z0)); last_act = np.zeros(12, np.float32); settle(40)
            print("⟳ reset Spot to the bottom")
        def frame():
            nonlocal last_act
            if canvas.is_key_down("R"):
                if not r_down[0]:
                    reset_spot()
                r_down[0] = True
            else:
                r_down[0] = False
            nf[0] += 1
            if nf[0] % 40 == 0:                                   # hot-reload the latest checkpoint ~1/s
                try:
                    mt = os.path.getmtime(args.model)
                    if mt != pol["mt"]:
                        pol["ac"], _, _ = load_policy(args.model, device=dev); pol["mt"] = mt
                        reset_spot()
                        print("↻ reloaded latest policy")
                except Exception:
                    pass
            # debounced staircase rebuild: rebuild once the rise / # steps sliders settle (no per-frame churn)
            cn = int(cfg["n"])
            if cfg["rise"] != prev["rise"] or cn != prev["n"]:
                prev["rise"], prev["n"] = cfg["rise"], cn; rb_wait[0] = 0
            elif cfg["rise"] != built["rise"] or cn != built["n"]:
                rb_wait[0] += 1
                if rb_wait[0] > 6:
                    rebuild_stairs()
                    reset_spot()

            control_tick(); render_chase(); ui.render(draw_ui)

        print(__doc__ + "\n(rise / # steps sliders rebuild the staircase; R = reset; hot-reloads the checkpoint)")
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
    bx = float(art.root_state()[0]); step_h = terr_h(bx, built["rise"], built["run"], built["n"])
    print(f"base over step height {step_h:.2f} m (~{step_h / args.rise:.0f} steps of {args.rise} m); base_z={art.root_state()[2]:.2f}")
    if not args.frames and shots:
        cols = 4
        rows = [np.concatenate(shots[r:r + cols], axis=1) for r in range(0, len(shots), cols)]
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=20) for r in rows]
        out = os.path.join(_HERE, "spot_stairs_ft_shot.png")
        _write_png(out, np.ascontiguousarray(np.concatenate(rows, axis=0))); print("saved montage ->", out)


if __name__ == "__main__":
    main()

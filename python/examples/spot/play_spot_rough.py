"""Watch the spotv2 policy DRIVE on UNEVEN (rough) terrain — single Spot, CPU deploy + render,
hot-reloading the checkpoint as train_spot_rough.py writes it. Steer it and watch it track.

    python play_spot_rough.py                       # live window; forward DepthSensor scan
    python play_spot_rough.py --model spot_rough.pt
    python play_spot_rough.py --analytic            # privileged exact-height oracle (the old fake scan)
    python play_spot_rough.py --check 400           # headless smoke (no window)

DRIVE (body frame, +x fwd / +y left):
    forward  UP / KP8        back     DOWN / KP2
    strafe L LEFT / KP4      strafe R RIGHT / KP6
    turn  L  N / KP7         turn  R  M / KP9
    R = reset    (ImGui: bump amplitude slider, auto-forward, heading-hold)

Same 58-d SpotRoughEnv obs (heading-relative scan + box-top terrain height), raw obs, tgs_pcm/0.005
CPU contact to match GpuSim training. The terrain is gentle smooth-noise humps (no discrete steps).
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
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets, grid_texture, _quat_to_R,
                         default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, Z0)
from spot_terrain_env import scan_xy_np, HALF_W, VX_HI, VY_HI, WZ_HI
from spot_rough_env import rough_profile, ROUGH_X0, ROUGH_RUN, N_BOXES
from spot_depth_scan import ForwardDepthScanner

DISP = 820
GRAV = np.array([0.0, 0.0, -1.0])
BOX_W = 2.0 * HALF_W
CRUISE = 0.8


def terr_h(x, y, heights):
    """numpy mirror of SpotRoughEnv._terrain_h for ONE lane at y=0: box-top height in-lane, 0 outside."""
    if abs(y) >= HALF_W:
        return 0.0
    j = int(np.clip(np.floor((x - ROUGH_X0) / ROUGH_RUN), 0, N_BOXES - 1))
    return float(heights[j])


def analytic_scan(art, heights):
    """Privileged scan: exact box-top height at the 45 heading-relative grid points (oracle baseline)."""
    rs = art.root_state(); R = _quat_to_R(rs[3:7])
    x, y = float(rs[0]), float(rs[1])
    hx, hy = float(R[0, 0]), float(R[1, 0]); nrm = math.hypot(hx, hy) or 1.0
    cyaw, syaw = hx / nrm, hy / nrm
    h_here = terr_h(x, y, heights)
    px, py = scan_xy_np(x, y, cyaw, syaw)
    ahead = np.clip(np.array([terr_h(float(pxi), float(pyi), heights) - h_here
                              for pxi, pyi in zip(px, py)], np.float32), -1.0, 1.0)
    return ahead, h_here


def v2_obs(art, last_act, cmd, ahead, h_here):
    """94-d SpotRoughEnv obs. `ahead` (45) + `h_here` come from the depth sensor OR the oracle."""
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7]); Rt = R.T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV
    jp_isaac = art.joint_positions()[isaac_to_add]
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q
    z = float(rs[2])
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, last_act,
                           [z - h_here], ahead]).astype(np.float32)


def _resolve_model(path):
    if os.path.exists(path):
        return path
    latest = os.path.splitext(path)[0] + "_latest.pt"
    return latest if os.path.exists(latest) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "spot_steps.pt"))
    ap.add_argument("--amp", type=float, default=0.1, help="bump amplitude of the display terrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--analytic", action="store_true",
                    help="use the privileged exact-height scan oracle instead of the depth sensor")
    ap.add_argument("--noise", type=float, default=0.0, help="depth-sensor range noise std-dev (m)")
    ap.add_argument("--pgs", action="store_true",
                    help="use PhysX's default PGS solver at 0.002 substep (spot_deploy.py's contact model) "
                         "instead of tgs_pcm/0.005. Both transfer; tgs_pcm just matches the GPU training "
                         "contact model (TGS+PCM, 0.005 substep) most closely.")
    args = ap.parse_args()
    model = _resolve_model(args.model)
    if not os.path.exists(model):
        print(f"No policy at {model} — run train_spot_rough.py first."); sys.exit(0)
    live = args.check == 0
    dev = "cpu"
    ac, _, _ = load_policy(model, device=dev)
    pol = {"ac": ac, "mt": os.path.getmtime(model), "reloads": 0}
    print(f"[policy] {os.path.basename(model)}")

    if args.pgs:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    else:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.005, max_substeps=8, tgs_pcm=True)
    print(f"[solver] {'PGS/0.002 (PhysX default CPU solver)' if args.pgs else 'tgs_pcm/0.005 (matches GpuSim training)'}")
    ground = tp.Mesh(tp.BoxGeometry(60, 30, 1.0), tp.MeshStandardMaterial()); ground.position.set(20, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, fetch_assets())

    canvas = tp.Canvas("spotv2 - drive on rough terrain", width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(60, 30), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(40); floor.material.color = 0xffffff
    floor.material.roughness = 0.95; floor.receive_shadow = True; scene.add(floor)

    cfg = {"amp": args.amp}
    bumps = {"bodies": [], "meshes": [], "heights": None}

    def rebuild_rough():
        for b in bumps["bodies"]:
            world.remove(b)
        for m in bumps["meshes"]:
            scene.remove(m)
        bumps["bodies"].clear(); bumps["meshes"].clear()
        heights = rough_profile(amp=cfg["amp"], seed=args.seed)
        for j in range(N_BOXES):
            h = float(heights[j])
            if h < 0.006:
                continue
            m = tp.Mesh(tp.BoxGeometry(ROUGH_RUN, BOX_W, h), tp.MeshStandardMaterial())
            m.material.color = 0x8a7d6a; m.material.roughness = 0.95; m.receive_shadow = True
            m.position.set(ROUGH_X0 + j * ROUGH_RUN + ROUGH_RUN * 0.5, 0.0, h * 0.5)
            scene.add(m); bumps["meshes"].append(m)
            bumps["bodies"].append(world.add_static(m))
        bumps["heights"] = heights

    rebuild_rough()
    for m in meshes:
        scene.add(m)

    # PERCEPTION: forward depth camera + accumulating elevation map -> the 45-cell scan (vs --analytic oracle).
    scanner = None if args.analytic else ForwardDepthScanner(
        rend, scene, meshes,
        bounds=(ROUGH_X0 - 3.0, ROUGH_X0 + N_BOXES * ROUGH_RUN + 3.0, -4.0, 4.0), noise=args.noise)
    print(f"[scan] {'analytic oracle (privileged)' if args.analytic else 'forward DepthSensor + elevation map'}")

    cam = tp.PerspectiveCamera(46, 1.0, 0.05, 120); cam.up.set(0, 0, 1)
    cam.position.set(-2.6, -2.7, 1.4)
    state = {"last_act": np.zeros(12, np.float32), "hdg_lock": None,
             "auto_fwd": True, "hdg_hold": True, "cmd": (0.0, 0.0, 0.0)}
    BACK, HEIGHT, LAG = 2.8, 1.4, 0.10

    def settle(n):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32)); world.step(0.02)

    def reset_spot(nsettle=40):
        # lift the spawn by the highest box under the footprint so the feet clear the bumps (Z0 = feet
        # just above z=0 on flat ground); else they spawn INSIDE a hump -> invalid spawn / jolt.
        foot = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))
        h = max(terr_h(dx, dy, bumps["heights"]) for dx, dy in foot)
        art.reset(tp.Vector3(0.0, 0.0, Z0 + h + 0.02)); state["last_act"] = np.zeros(12, np.float32)
        state["hdg_lock"] = None; settle(nsettle)
        if scanner is not None:                                       # forget stale terrain, then pre-fill from here
            scanner.clear_map(); scanner.prewarm(art.root_state())

    def key_cmd():
        d = lambda *ks: any(canvas.is_key_down(k) for k in ks)
        vx = (VX_HI if d("UP", "KP8") else 0.0) - (1.0 if d("DOWN", "KP2") else 0.0)
        vy = (VY_HI if d("LEFT", "KP4") else 0.0) - (VY_HI if d("RIGHT", "KP6") else 0.0)
        turn = (WZ_HI if d("N", "KP7") else 0.0) - (WZ_HI if d("M", "KP9") else 0.0)
        return vx, vy, turn

    def control_tick(use_keys=True):
        vx, vy, turn = key_cmd() if use_keys else (0.0, 0.0, 0.0)
        if state["auto_fwd"] and vx == 0.0 and vy == 0.0 and turn == 0.0:
            vx = CRUISE
        rs0 = art.root_state(); R0 = _quat_to_R(rs0[3:7])
        yaw = math.atan2(float(R0[1, 0]), float(R0[0, 0]))
        if turn != 0.0 or not state["hdg_hold"]:
            wz = turn; state["hdg_lock"] = yaw
        else:
            if state["hdg_lock"] is None:
                state["hdg_lock"] = yaw
            err = (yaw - state["hdg_lock"] + math.pi) % (2 * math.pi) - math.pi
            wz = float(np.clip(-2.0 * err, -1.0, 1.0))
        cmd = np.array([vx, vy, wz], np.float32); state["cmd"] = (vx, vy, wz)
        ahead, h_here = scanner.scan(art.root_state()) if scanner is not None else analytic_scan(art, bumps["heights"])
        with torch.no_grad():
            obs = v2_obs(art, state["last_act"], cmd, ahead, h_here)
            a = pol["ac"].act_mean(torch.from_numpy(obs)[None])[0].numpy()
        state["last_act"] = a
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)

    def render_chase():
        rs = art.root_state(); p = np.array(rs[0:3], float)
        fwd = _quat_to_R(rs[3:7])[:, 0]; fwd = np.array([fwd[0], fwd[1], 0.0])
        nrm = np.linalg.norm(fwd); fwd = fwd / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
        desired = p - fwd * BACK + np.array([0.0, 0.0, HEIGHT])
        cam.position.lerp(tp.Vector3(float(desired[0]), float(desired[1]), float(desired[2])), LAG)
        cam.look_at(float(p[0] + fwd[0] * 0.4), float(p[1] + fwd[1] * 0.4), float(p[2] + 0.1))
        rend.render(scene, cam)

    reset_spot(120)                                                  # initial drop-settle (lifted off the bumps)

    if not live:
        for _ in range(args.check):
            control_tick(use_keys=False)
        rs = art.root_state(); bx, by, bz = float(rs[0]), float(rs[1]), float(rs[2])
        h = terr_h(bx, by, bumps["heights"])
        print(f"after {args.check} fwd steps: base=({bx:+.2f},{by:+.2f},{bz:.2f})  "
              f"terrain_under_base={h:.2f}  clearance={bz - h:.2f}  up_ok={bool(_quat_to_R(rs[3:7])[2,2] > 0.5)}")
        print("SPOTV2 ROUGH PLAY HEADLESS CHECK: OK")
        return

    ui = tp.ImguiContext(canvas, rend)
    def on_resize(w, h):
        cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
    canvas.on_window_resize(on_resize)
    nf = [0]; r_down = [False]; prev_amp = [cfg["amp"]]; built_amp = [cfg["amp"]]; rb_wait = [0]

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12); tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("spotv2 drive (rough)")
        tp.imgui.text(f"policy: {os.path.basename(model)}  (reloads {pol['reloads']})")
        vx, vy, wz = state["cmd"]
        tp.imgui.text(f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        rs = art.root_state(); h = terr_h(float(rs[0]), float(rs[1]), bumps["heights"])
        tp.imgui.text(f"clearance over terrain: {float(rs[2]) - h:.2f} m")
        _, state["auto_fwd"] = tp.imgui.checkbox("auto-forward (idle)", state["auto_fwd"])
        _, state["hdg_hold"] = tp.imgui.checkbox("heading-hold (no-turn)", state["hdg_hold"])
        if scanner is not None:
            tp.imgui.text("scan: forward DepthSensor + elevation map")
            _, scanner.show_grid = tp.imgui.checkbox("show scan grid (policy input)", scanner.show_grid)
            _, scanner.show_cloud = tp.imgui.checkbox("show depth point cloud", scanner.show_cloud)
            _, scanner.sensor.range_noise = tp.imgui.slider_float("range noise (m)", scanner.sensor.range_noise, 0.0, 0.15)
        else:
            tp.imgui.text("scan: analytic oracle (privileged)")
        _, cfg["amp"] = tp.imgui.slider_float("bump amplitude (m)", cfg["amp"], 0.0, 0.12)
        if abs(cfg["amp"] - built_amp[0]) > 1e-4:
            tp.imgui.text("rebuilding on release...")
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   |   R = reset")
        tp.imgui.end()

    def frame():
        if canvas.is_key_down("R"):
            if not r_down[0]:
                reset_spot(); print("[reset]")
            r_down[0] = True
        else:
            r_down[0] = False
        nf[0] += 1
        if nf[0] % 40 == 0:
            try:
                mt = os.path.getmtime(model)
                if mt != pol["mt"]:
                    pol["ac"], _, _ = load_policy(model, device=dev); pol["mt"] = mt
                    pol["reloads"] += 1; reset_spot(); print(f"[reload] {os.path.basename(model)} (#{pol['reloads']})")
            except Exception:
                pass
        if cfg["amp"] != prev_amp[0]:                            # slider still moving -> reset the debounce
            prev_amp[0] = cfg["amp"]; rb_wait[0] = 0
        elif abs(cfg["amp"] - built_amp[0]) > 1e-4:              # settled and differs from what's built -> rebuild
            rb_wait[0] += 1
            if rb_wait[0] > 6:
                rebuild_rough(); reset_spot(); built_amp[0] = cfg["amp"]
        control_tick(use_keys=True); render_chase(); ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()

"""Watch the spotv2 policy CLIMB DISCRETE STAIRS — single Spot, CPU deploy + render, hot-reloading the
checkpoint as train_spot_steps.py writes it. The terrain is the curriculum ladder: tents of increasing
riser along +x. Drive it through, or use the `level` slider to spawn at a chosen riser band.

    python play_spot_steps.py                   # live; hot-reloads spot_steps_latest.pt
    python play_spot_steps.py --model spot_steps.pt
    python play_spot_steps.py --check 400 --level 2

DRIVE (body frame, +x fwd / +y left):
    forward  UP / KP8        back     DOWN / KP2
    strafe L LEFT / KP4      strafe R RIGHT / KP6
    turn  L  N / KP7         turn  R  M / KP9
    R = reset    (ImGui: level slider = which riser band to spawn at, auto-forward, heading-hold)
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
from spot_terrain_env import scan_xy_np, N_SCAN, HALF_W, VX_HI, VY_HI, WZ_HI
from spot_steps_env import (RISERS, N_LEVELS, N_UP, STEP_RUN, FLAT_APPROACH, LAND, BAND_LEN,
                            SPAWN_OFF, STRIP_LEN, HALF_W_BOX, HALF_W_STEPS)

DISP = 820
GRAV = np.array([0.0, 0.0, -1.0])
CRUISE = 0.8


def terr_h(x, y):
    """numpy mirror of SpotStepsEnv._terrain_h for ONE lane at y=0 (the full band ladder)."""
    if abs(y) >= HALF_W_STEPS or x < 0.0 or x >= STRIP_LEN:
        return 0.0
    band = min(int(x // BAND_LEN), N_LEVELS - 1)
    r = RISERS[band]
    t = (x - band * BAND_LEN) - FLAT_APPROACH
    up_end = N_UP * STEP_RUN; land_end = up_end + LAND; tent_end = land_end + N_UP * STEP_RUN
    if t < 0.0 or t >= tent_end:
        return 0.0
    if t < up_end:
        steps = min(int(math.floor(t / STEP_RUN)) + 1, N_UP)
    elif t < land_end:
        steps = N_UP
    else:
        steps = N_UP - min(int(math.floor((t - land_end) / STEP_RUN)) + 1, N_UP)
    return float(r * steps)


def _scan_color(d):
    """Height-delta d in [-1,1] -> 0xRRGGBB for a probe marker: blue (drop) -> green (level) -> red (rise)."""
    if d >= 0.0:
        r, g, b = 0.2 + 0.8 * d, 0.9 - 0.7 * d, 0.2
    else:
        r, g, b = 0.2, 0.9 + 0.6 * d, 0.3 - 0.6 * d
    clamp = lambda v: int(max(0.0, min(1.0, v)) * 255)
    return (clamp(r) << 16) | (clamp(g) << 8) | clamp(b)


def v2_obs(art, last_act, cmd):
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7]); Rt = R.T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV
    jp_isaac = art.joint_positions()[isaac_to_add]
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q
    x, y, z = float(rs[0]), float(rs[1]), float(rs[2])
    hx, hy = float(R[0, 0]), float(R[1, 0])
    nrm = math.hypot(hx, hy) or 1.0
    cyaw, syaw = hx / nrm, hy / nrm
    h_here = terr_h(x, y)
    px, py = scan_xy_np(x, y, cyaw, syaw)                          # 2-D heading-relative grid (world)
    ahead = np.clip(np.array([terr_h(float(pxi), float(pyi)) - h_here
                              for pxi, pyi in zip(px, py)], np.float32), -1.0, 1.0)
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
    ap.add_argument("--level", type=int, default=0, help="riser band to spawn at (0..%d)" % (N_LEVELS - 1))
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--pgs", action="store_true",
                    help="use PhysX's default PGS/0.002 solver instead of tgs_pcm/0.005 (both transfer)")
    args = ap.parse_args()
    model = _resolve_model(args.model)
    if not os.path.exists(model):
        print(f"No policy at {model} — run train_spot_steps.py first."); sys.exit(0)
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
    ground = tp.Mesh(tp.BoxGeometry(STRIP_LEN + 20, 2.0 * HALF_W_STEPS, 1.0), tp.MeshStandardMaterial())
    ground.position.set(STRIP_LEN * 0.5, 0, -0.5); world.add_static(ground)
    art, meshes = build_spot(world, fetch_assets())

    # build the band ladder (one lane): a tent at each riser level along +x (added to the scene below)
    step_meshes = []
    def _box(w, h, cx, col):
        m = tp.Mesh(tp.BoxGeometry(w, HALF_W_BOX, h), tp.MeshStandardMaterial())
        m.material.color = col; m.material.roughness = 0.9; m.receive_shadow = True
        m.position.set(cx, 0.0, h * 0.5); world.add_static(m); step_meshes.append(m)
    for j in range(N_LEVELS):
        r = RISERS[j]; x0 = j * BAND_LEN + FLAT_APPROACH; col = 0x6f7d8c if j % 2 == 0 else 0x808e9c
        for s in range(N_UP):
            _box(STEP_RUN, (s + 1) * r, x0 + s * STEP_RUN + STEP_RUN * 0.5, col)
        up_end = x0 + N_UP * STEP_RUN; top = N_UP * r
        _box(LAND, top, up_end + LAND * 0.5, col)
        land_end = up_end + LAND
        for s in range(N_UP - 1):
            _box(STEP_RUN, (N_UP - 1 - s) * r, land_end + s * STEP_RUN + STEP_RUN * 0.5, col)

    canvas = tp.Canvas("spotv2 - climb stairs", width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(STRIP_LEN + 20, 6), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(40); floor.material.color = 0xffffff
    floor.material.roughness = 0.95; floor.receive_shadow = True
    floor.position.set(STRIP_LEN * 0.5, 0, 0); scene.add(floor)
    for m in step_meshes:
        scene.add(m)
    for m in meshes:
        scene.add(m)

    # 2-D scan-grid overlay: one small sphere per probe, repositioned each frame to the world grid
    # and colored by measured height-delta -> shows exactly what the policy senses on the terrain.
    scan_group = tp.Group(); scan_markers = []
    for _ in range(N_SCAN):
        mk = tp.Mesh(tp.SphereGeometry(0.022, 8, 6), tp.MeshStandardMaterial())
        mk.material.color = 0x22ff88
        scan_markers.append(mk); scan_group.add(mk)
    scene.add(scan_group)

    cam = tp.PerspectiveCamera(46, 1.0, 0.05, 200); cam.up.set(0, 0, 1)
    cam.position.set(-2.6, -2.7, 1.4)
    state = {"last_act": np.zeros(12, np.float32), "hdg_lock": None,
             "auto_fwd": True, "hdg_hold": True, "cmd": (0.0, 0.0, 0.0), "level": int(args.level),
             "show_scan": True}
    BACK, HEIGHT, LAG = 3.0, 1.5, 0.10

    def settle(n):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32)); world.step(0.02)

    def reset_spot(nsettle=40):
        sx = state["level"] * BAND_LEN + SPAWN_OFF                    # spawn at the chosen band's flat approach
        foot = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))
        h = max(terr_h(sx + dx, dy) for dx, dy in foot)
        art.reset(tp.Vector3(sx, 0.0, Z0 + h + 0.02)); state["last_act"] = np.zeros(12, np.float32)
        state["hdg_lock"] = None; settle(nsettle)

    def key_cmd():
        d = lambda *ks: any(canvas.is_key_down(k) for k in ks)
        vx = (VX_HI if d("UP", "KP8") else 0.0) - (1.0 if d("DOWN", "KP2") else 0.0)
        vy = (VY_HI if d("LEFT", "KP4") else 0.0) - (VY_HI if d("RIGHT", "KP6") else 0.0)
        turn = (WZ_HI if d("N", "KP7") else 0.0) - (WZ_HI if d("M", "KP9") else 0.0)
        return vx, vy, turn

    def control_tick(use_keys=True):
        vx, vy, turn = key_cmd() if use_keys else (CRUISE, 0.0, 0.0)
        if use_keys and state["auto_fwd"] and vx == 0.0 and vy == 0.0 and turn == 0.0:
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
        with torch.no_grad():
            obs = v2_obs(art, state["last_act"], cmd)
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

    def update_scan_viz():
        scan_group.visible = state["show_scan"]
        if not state["show_scan"]:
            return
        rs = art.root_state(); R = _quat_to_R(rs[3:7])
        x, y = float(rs[0]), float(rs[1])
        hx, hy = float(R[0, 0]), float(R[1, 0]); nrm = math.hypot(hx, hy) or 1.0
        cyaw, syaw = hx / nrm, hy / nrm
        px, py = scan_xy_np(x, y, cyaw, syaw)
        h_here = terr_h(x, y)
        for i in range(N_SCAN):
            h = terr_h(float(px[i]), float(py[i]))
            scan_markers[i].position.set(float(px[i]), float(py[i]), h + 0.03)
            scan_markers[i].material.color = _scan_color(float(np.clip((h - h_here) / 0.2, -1.0, 1.0)))

    reset_spot(120)

    if not live:
        for _ in range(args.check):
            control_tick(use_keys=False)
        rs = art.root_state(); bx, by, bz = float(rs[0]), float(rs[1]), float(rs[2])
        h = terr_h(bx, by)
        print(f"after {args.check} fwd steps (spawn level {args.level}, riser {RISERS[args.level]:.02f}): "
              f"base=({bx:+.2f},{by:+.2f},{bz:.2f})  terrain_under_base={h:.2f}  "
              f"clearance={bz - h:.2f}  up_ok={bool(_quat_to_R(rs[3:7])[2,2] > 0.5)}")
        print("SPOTV2 STEPS PLAY HEADLESS CHECK: OK")
        return

    ui = tp.ImguiContext(canvas, rend)
    def on_resize(w, h):
        cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
    canvas.on_window_resize(on_resize)
    nf = [0]; r_down = [False]; prev_lvl = [state["level"]]

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12); tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("spotv2 climb stairs")
        tp.imgui.text(f"policy: {os.path.basename(model)}  (reloads {pol['reloads']})")
        vx, vy, wz = state["cmd"]
        tp.imgui.text(f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        rs = art.root_state(); h = terr_h(float(rs[0]), float(rs[1]))
        tp.imgui.text(f"climbed {h:.2f} m   clearance {float(rs[2]) - h:.2f} m")
        _, state["auto_fwd"] = tp.imgui.checkbox("auto-forward (idle)", state["auto_fwd"])
        _, state["hdg_hold"] = tp.imgui.checkbox("heading-hold (no-turn)", state["hdg_hold"])
        _, state["show_scan"] = tp.imgui.checkbox("show 2-D scan grid", state["show_scan"])
        _, state["level"] = tp.imgui.slider_int("spawn riser band", int(state["level"]), 0, N_LEVELS - 1)
        tp.imgui.text(f"band {int(state['level'])} riser = {RISERS[int(state['level'])]:.02f} m")
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
        if int(state["level"]) != prev_lvl[0]:                       # level slider changed -> respawn there
            prev_lvl[0] = int(state["level"]); reset_spot()
        if nf[0] % 40 == 0:
            try:
                mt = os.path.getmtime(model)
                if mt != pol["mt"]:
                    pol["ac"], _, _ = load_policy(model, device=dev); pol["mt"] = mt
                    pol["reloads"] += 1; reset_spot(); print(f"[reload] {os.path.basename(model)} (#{pol['reloads']})")
            except Exception:
                pass
        control_tick(use_keys=True); update_scan_viz(); render_chase(); ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()

"""Watch the spotv2 velocity-tracking policy DRIVE on tent terrain — single Spot, CPU deploy + render,
hot-reloading the checkpoint as train_spot_terrain.py writes it. STEER it yourself and watch it track.

    python play_spot_terrain.py                     # live window; forward DepthSensor scan
    python play_spot_terrain.py --model spot_terrain_latest.pt
    python play_spot_terrain.py --analytic          # privileged exact-height oracle (the old fake scan)
    python play_spot_terrain.py --check 400         # headless smoke (no window): drive forward, print clearance

DRIVE (body frame, +x fwd / +y left):
    forward  UP / KP8        back     DOWN / KP2
    strafe L LEFT / KP4      strafe R RIGHT / KP6
    turn  L  N / KP7         turn  R  M / KP9
    R = reset to start    (ImGui: tent rise / #steps, auto-forward, heading-hold)

This is the v2 (velocity-command) policy: it STEERS from the command in obs[9:12] — there is no lane-hold
crutch. The obs is the 58-d SpotTerrainEnv layout, byte-for-byte (heading-relative 9-probe scan, 2-D tent
height: in-lane vs flat off-lane), on RAW obs (no normalizer). CPU world uses tgs_pcm + 0.005 substeps to
match the GpuSim training contact model. Policy runs on CPU to leave the GPU to the trainer.
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

try:                                                  # the Windows cp1252 console chokes on unicode glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets, grid_texture, _quat_to_R,
                         default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, Z0)
from spot_terrain_env import scan_xy_np, STAIR_X0, LAND_LEN, HALF_W, VX_HI, VY_HI, WZ_HI
from spot_depth_scan import ForwardDepthScanner

DISP = 820
GRAV = np.array([0.0, 0.0, -1.0])
BOX_W = 2.0 * HALF_W            # physical tent width == the obs on-lane gate (|y| < HALF_W)
CRUISE = 0.8                    # auto-forward speed (drives into the tent)


def terr_h(x, y, rise, run, n, x0=STAIR_X0, land=LAND_LEN):
    """numpy mirror of SpotTerrainEnv._terrain_h for ONE lane centered at y=0: tent height in-lane,
    0 (flat) beyond HALF_W. Up-then-down: flat -> ascend n -> landing -> descend n -> flat."""
    if abs(y) >= HALF_W:
        return 0.0
    up_end = x0 + n * run
    land_end = up_end + land
    asc = min(max(math.floor((x - x0) / run) + 1.0, 0.0), n)
    desc = min(max(math.floor((x - land_end) / run) + 1.0, 0.0), n)
    steps = asc if x < up_end else (n if x < land_end else n - desc)
    return float(steps * rise)


def analytic_scan(art, rise, run, n):
    """Privileged scan: exact tent height at the 45 heading-relative grid points (oracle baseline)."""
    rs = art.root_state(); R = _quat_to_R(rs[3:7])
    x, y = float(rs[0]), float(rs[1])
    hx, hy = float(R[0, 0]), float(R[1, 0]); nrm = math.hypot(hx, hy) or 1.0
    cyaw, syaw = hx / nrm, hy / nrm
    h_here = terr_h(x, y, rise, run, n)
    px, py = scan_xy_np(x, y, cyaw, syaw)
    ahead = np.clip(np.array([terr_h(float(pxi), float(pyi), rise, run, n) - h_here
                              for pxi, pyi in zip(px, py)], np.float32), -1.0, 1.0)
    return ahead, h_here


def v2_obs(art, last_act, cmd, ahead, h_here):
    """94-d SpotTerrainEnv obs. `ahead` (45) + `h_here` come from the depth sensor OR the oracle."""
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7]); Rt = R.T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV
    jp_isaac = art.joint_positions()[isaac_to_add]                 # add-order sim -> isaac order
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q                                    # default_q is isaac order
    z = float(rs[2])
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, last_act,
                           [z - h_here], ahead]).astype(np.float32)


def _resolve_model(path):
    """Default to the BEST checkpoint, fall back to the always-current _latest one."""
    if os.path.exists(path):
        return path
    latest = os.path.splitext(path)[0] + "_latest.pt"
    return latest if os.path.exists(latest) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "spot_steps.pt"))
    ap.add_argument("--rise", type=float, default=0.10)
    ap.add_argument("--run", type=float, default=0.32)
    ap.add_argument("--steps", type=int, default=6, help="tent steps up (= steps down)")
    ap.add_argument("--check", type=int, default=0, help="headless smoke: run N steps forward, print, exit")
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
        print(f"No policy at {model} — run train_spot_terrain.py first."); sys.exit(0)
    live = args.check == 0
    dev = "cpu"                                                    # leave the GPU to the trainer
    ac, _, _ = load_policy(model, device=dev)
    pol = {"ac": ac, "mt": os.path.getmtime(model), "reloads": 0}
    print(f"[policy] {os.path.basename(model)}")

    # tgs_pcm + 0.005 substep -> matches the GpuSim TGS+PCM training contact model on CPU. --pgs uses
    # PhysX's default PGS solver at 0.002 (spot_deploy.py's model); both transfer for this gait.
    if args.pgs:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    else:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.005, max_substeps=8, tgs_pcm=True)
    print(f"[solver] {'PGS/0.002 (PhysX default CPU solver)' if args.pgs else 'tgs_pcm/0.005 (matches GpuSim training)'}")
    ground = tp.Mesh(tp.BoxGeometry(60, 30, 1.0), tp.MeshStandardMaterial()); ground.position.set(20, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, fetch_assets())               # default Isaac gains (the warm-start regime)

    canvas = tp.Canvas("spotv2 - drive on terrain", width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(60, 30), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(40); floor.material.color = 0xffffff
    floor.material.roughness = 0.95; floor.receive_shadow = True; scene.add(floor)

    cfg = {"rise": args.rise, "run": args.run, "n": int(args.steps)}   # slider targets
    built = {"rise": None, "run": None, "n": None}                     # what's physically in the world (obs reads this)
    stairs = {"bodies": [], "meshes": []}

    def _stair_box(w, h, x):
        m = tp.Mesh(tp.BoxGeometry(w, BOX_W, h), tp.MeshStandardMaterial())
        m.material.color = 0x8a96a6; m.material.roughness = 0.9; m.receive_shadow = True
        m.position.set(x, 0.0, h * 0.5)
        scene.add(m); stairs["meshes"].append(m)
        stairs["bodies"].append(world.add_static(m))

    def rebuild_tent():
        for b in stairs["bodies"]:
            world.remove(b)
        for m in stairs["meshes"]:
            scene.remove(m)
        stairs["bodies"].clear(); stairs["meshes"].clear()
        rise, run, n = cfg["rise"], cfg["run"], int(cfg["n"])
        for s in range(n):                                        # ascend
            _stair_box(run, (s + 1) * rise, STAIR_X0 + s * run + run * 0.5)
        up_end = STAIR_X0 + n * run
        _stair_box(LAND_LEN, n * rise, up_end + LAND_LEN * 0.5)   # landing
        land_end = up_end + LAND_LEN
        for s in range(n - 1):                                    # descend
            _stair_box(run, (n - 1 - s) * rise, land_end + s * run + run * 0.5)
        built.update(rise=rise, run=run, n=n)

    rebuild_tent()
    for m in meshes:
        scene.add(m)

    # PERCEPTION: forward depth camera + accumulating elevation map -> the 45-cell scan (vs --analytic oracle).
    scanner = None if args.analytic else ForwardDepthScanner(
        rend, scene, meshes, bounds=(-3.0, 24.0, -5.0, 5.0), noise=args.noise)
    print(f"[scan] {'analytic oracle (privileged)' if args.analytic else 'forward DepthSensor + elevation map'}")

    cam = tp.PerspectiveCamera(46, 1.0, 0.05, 120); cam.up.set(0, 0, 1)
    cam.position.set(-2.6, -2.7, 1.4)
    state = {"last_act": np.zeros(12, np.float32), "hdg_lock": None,
             "auto_fwd": True, "hdg_hold": True, "cmd": (0.0, 0.0, 0.0)}
    BACK, HEIGHT, LAG = 2.8, 1.4, 0.10

    def settle(n):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32)); world.step(0.02)

    def reset_spot():
        art.reset(tp.Vector3(0, 0, Z0)); state["last_act"] = np.zeros(12, np.float32)
        state["hdg_lock"] = None; settle(40)
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
            vx = CRUISE                                           # idle -> cruise into the tent
        rs0 = art.root_state(); R0 = _quat_to_R(rs0[3:7])
        yaw = math.atan2(float(R0[1, 0]), float(R0[0, 0]))
        if turn != 0.0 or not state["hdg_hold"]:                 # heading-hold = deploy convenience (small wz)
            wz = turn; state["hdg_lock"] = yaw
        else:
            if state["hdg_lock"] is None:
                state["hdg_lock"] = yaw
            err = (yaw - state["hdg_lock"] + math.pi) % (2 * math.pi) - math.pi
            wz = float(np.clip(-2.0 * err, -1.0, 1.0))
        cmd = np.array([vx, vy, wz], np.float32); state["cmd"] = (vx, vy, wz)
        ahead, h_here = (scanner.scan(art.root_state()) if scanner is not None
                         else analytic_scan(art, built["rise"], built["run"], built["n"]))
        with torch.no_grad():
            obs = v2_obs(art, state["last_act"], cmd, ahead, h_here)
            a = pol["ac"].act_mean(torch.from_numpy(obs)[None])[0].numpy()
        state["last_act"] = a                                    # NO clamp: isaac action range ~+-8
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

    settle(120)
    if scanner is not None:                                           # pre-fill the elevation map before driving
        scanner.clear_map(); scanner.prewarm(art.root_state())

    # ----- headless smoke: drive forward N steps, report, exit (no window) -----
    if not live:
        for _ in range(args.check):
            control_tick(use_keys=False)
        rs = art.root_state(); bx, by, bz = float(rs[0]), float(rs[1]), float(rs[2])
        h = terr_h(bx, by, built["rise"], built["run"], built["n"])
        print(f"after {args.check} fwd steps: base=({bx:+.2f},{by:+.2f},{bz:.2f})  "
              f"terrain_under_base={h:.2f}  clearance={bz - h:.2f}  up_ok={bool(_quat_to_R(rs[3:7])[2,2] > 0.5)}")
        print("SPOTV2 PLAY HEADLESS CHECK: OK")
        return

    # ----- live window -----
    ui = tp.ImguiContext(canvas, rend)
    def on_resize(w, h):
        cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
    canvas.on_window_resize(on_resize)
    nf = [0]; r_down = [False]
    prev = {"rise": cfg["rise"], "run": cfg["run"], "n": int(cfg["n"])}; rb_wait = [0]

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12); tp.imgui.set_next_window_size(290, 0)
        tp.imgui.begin("spotv2 drive")
        tp.imgui.text(f"policy: {os.path.basename(model)}  (reloads {pol['reloads']})")
        vx, vy, wz = state["cmd"]
        tp.imgui.text(f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        rs = art.root_state(); h = terr_h(float(rs[0]), float(rs[1]), built["rise"], built["run"], built["n"])
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
        _, cfg["rise"] = tp.imgui.slider_float("rise (m)", cfg["rise"], 0.02, 0.30)
        _, cfg["n"] = tp.imgui.slider_int("# steps up=down", int(cfg["n"]), 1, 12)
        tp.imgui.text(f"peak height: {int(cfg['n']) * cfg['rise']:.2f} m")
        if cfg["rise"] != built["rise"] or int(cfg["n"]) != built["n"]:
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
        if nf[0] % 40 == 0:                                       # hot-reload the checkpoint ~1/s
            try:
                mt = os.path.getmtime(model)
                if mt != pol["mt"]:
                    pol["ac"], _, _ = load_policy(model, device=dev); pol["mt"] = mt
                    pol["reloads"] += 1; reset_spot(); print(f"[reload] {os.path.basename(model)} (#{pol['reloads']})")
            except Exception:
                pass
        cn = int(cfg["n"])                                        # debounced tent rebuild on slider release
        if cfg["rise"] != prev["rise"] or cn != prev["n"]:
            prev["rise"], prev["n"] = cfg["rise"], cn; rb_wait[0] = 0
        elif cfg["rise"] != built["rise"] or cn != built["n"]:
            rb_wait[0] += 1
            if rb_wait[0] > 6:
                rebuild_tent(); reset_spot()
        control_tick(use_keys=True); render_chase(); ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()

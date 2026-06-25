"""Watch the spotv2 policy roam a BIG, ROUGH heightfield patch — single Spot, CPU deploy + render,
hot-reloading the checkpoint as it trains. Steer anywhere across the patch and watch it stay upright.

    python play_spot_heightfield.py                  # live; forward DepthSensor scan; hot-reloads spot_steps.pt
    python play_spot_heightfield.py --model spot_hf.pt --amp 0.30
    python play_spot_heightfield.py --analytic       # privileged exact-height oracle (the old fake scan)
    python play_spot_heightfield.py --check 400      # headless smoke

DRIVE (body frame, +x fwd / +y left):
    forward  UP / KP8        back     DOWN / KP2
    strafe L LEFT / KP4      strafe R RIGHT / KP6
    turn  L  N / KP7         turn  R  M / KP9
    R = reset    (ImGui: amplitude + shape-seed sliders, auto-forward, heading-hold)

The terrain is a big continuous triangle-mesh heightfield (20 m x 16 m, Python soup -> add_static_trimesh),
smooth in x AND y but harder than the training tile (shorter rotated octaves, amplitude to 0.45 m). Obs is
the 94-d SpotHeightfieldEnv layout (heading-relative 2-D scan grid + bilinear height). tgs_pcm/0.005.
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
from spot_terrain_env import scan_xy_np, VX_HI, VY_HI, WZ_HI
from spot_heightfield_env import build_hf_geom, _taper
from spot_depth_scan import ForwardDepthScanner

DISP = 820
GRAV = np.array([0.0, 0.0, -1.0])
CRUISE = 0.8

# ---- BIG 2-D demo patch (independent of the narrow per-lane training tile): roam + turn anywhere ----
PATCH_X0, PATCH_X1 = -4.0, 16.0      # 20 m forward extent
PATCH_HALF_Y = 8.0                    # +/-8 m -> 16 m wide (drive AND steer across rough terrain)
PATCH_RES = 0.20                      # grid spacing (m); fine enough for the scan to resolve the features
PATCH_NX = int(round((PATCH_X1 - PATCH_X0) / PATCH_RES)) + 1     # 101
PATCH_NY = int(round((2.0 * PATCH_HALF_Y) / PATCH_RES)) + 1      # 81
# HARDER than the training tile: shorter dominant wavelengths + an extra octave -> steeper, busier terrain.
# (wavelength_m, weight); each octave is randomly ROTATED so ridges don't line up with the grid.
PATCH_OCTAVES = ((4.0, 1.0), (2.0, 0.6), (1.1, 0.32))
SPAWN_X = PATCH_X0 + 0.6              # spawn on the tapered patch edge (terrain~0) -> stable fold; walk in


def make_patch(seed):
    """A big square smooth-noise height grid H[nx,ny] in [0,1], edge-tapered to 0 so it meets the flat
    ground. Same separable-sine recipe as the training tiles but with shorter wavelengths + rotation."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(PATCH_X0, PATCH_X1, PATCH_NX).astype(np.float32)
    ys = np.linspace(-PATCH_HALF_Y, PATCH_HALF_Y, PATCH_NY).astype(np.float32)
    h = np.zeros((PATCH_NX, PATCH_NY), np.float32)
    for wl, w in PATCH_OCTAVES:
        ang = float(rng.uniform(0.0, 2.0 * np.pi)); ca, sa = np.cos(ang), np.sin(ang)
        xr = ca * xs[:, None] - sa * ys[None, :]
        yr = sa * xs[:, None] + ca * ys[None, :]
        phx, phy = rng.uniform(0.0, 2.0 * np.pi, 2)
        h += w * np.sin(2.0 * np.pi * xr / wl + phx) * np.sin(2.0 * np.pi * yr / wl + phy)
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    win = _taper(PATCH_NX, 0.12)[:, None] * _taper(PATCH_NY, 0.12)[None, :]
    return (h * win).astype(np.float32), xs, ys


def terr_h(x, y, hf):
    """numpy bilinear height of the big 2-D demo patch at world (x,y); 0 outside the patch (flat ground)."""
    if x < PATCH_X0 or x > PATCH_X1 or y < -PATCH_HALF_Y or y > PATCH_HALF_Y:
        return 0.0
    H, amp = hf["H"], hf["amp"]
    fx = (x - PATCH_X0) / (PATCH_X1 - PATCH_X0) * (PATCH_NX - 1)
    fy = (y + PATCH_HALF_Y) / (2.0 * PATCH_HALF_Y) * (PATCH_NY - 1)
    ix = int(min(max(fx, 0), PATCH_NX - 2)); iy = int(min(max(fy, 0), PATCH_NY - 2))
    tx = fx - ix; ty = fy - iy
    h = ((1 - tx) * (1 - ty) * H[ix, iy] + tx * (1 - ty) * H[ix + 1, iy]
         + (1 - tx) * ty * H[ix, iy + 1] + tx * ty * H[ix + 1, iy + 1])
    return float(amp * h)


def analytic_scan(art, hf):
    """Privileged scan: exact bilinear patch height at the 45 heading-relative grid points (oracle)."""
    rs = art.root_state(); R = _quat_to_R(rs[3:7])
    x, y = float(rs[0]), float(rs[1])
    hx, hy = float(R[0, 0]), float(R[1, 0]); nrm = math.hypot(hx, hy) or 1.0
    cyaw, syaw = hx / nrm, hy / nrm
    h_here = terr_h(x, y, hf)
    px, py = scan_xy_np(x, y, cyaw, syaw)
    ahead = np.clip(np.array([terr_h(float(pxi), float(pyi), hf) - h_here
                              for pxi, pyi in zip(px, py)], np.float32), -1.0, 1.0)
    return ahead, h_here


def v2_obs(art, last_act, cmd, ahead, h_here):
    """94-d SpotHeightfieldEnv obs. `ahead` (45) + `h_here` come from the depth sensor OR the oracle."""
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
    ap.add_argument("--amp", type=float, default=0.22, help="display heightfield amplitude (harder default)")
    ap.add_argument("--seed", type=int, default=0, help="terrain shape seed")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--analytic", action="store_true",
                    help="use the privileged exact-height scan oracle instead of the depth sensor")
    ap.add_argument("--noise", type=float, default=0.0, help="depth-sensor range noise std-dev (m)")
    ap.add_argument("--pgs", action="store_true",
                    help="use PhysX's default PGS/0.002 solver instead of tgs_pcm/0.005 (both transfer)")
    args = ap.parse_args()
    model = _resolve_model(args.model)
    if not os.path.exists(model):
        print(f"No policy at {model} — run train_spot_heightfield.py first."); sys.exit(0)
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
    ground = tp.Mesh(tp.BoxGeometry(80, 80, 1.0), tp.MeshStandardMaterial()); ground.position.set(6, 0, -0.5)
    world.add_static(ground)
    art, meshes = build_spot(world, fetch_assets())

    canvas = tp.Canvas("spotv2 - drive on heightfield", width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas); rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic; rend.tone_mapping_exposure = 1.1
    scene = tp.Scene(); scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7); sun.position.set(4, -6, 12); sun.cast_shadow = True; scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(80, 80), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(53); floor.material.color = 0xffffff
    floor.material.roughness = 0.95; floor.receive_shadow = True; floor.position.set(6, 0, 0); scene.add(floor)

    cfg = {"amp": args.amp, "seed": args.seed}
    hf = {"body": None, "mesh": None, "H": None, "amp": cfg["amp"]}
    geom_cache = {}                                                  # seed -> (geom, H); built once per shape

    def rebuild_hf():
        seed = int(cfg["seed"])
        if seed not in geom_cache:                                  # the slow triangle-soup build: once per seed
            H, xs, ys = make_patch(seed)
            geom_cache[seed] = (build_hf_geom(H, xs, ys), H)
        geom, H = geom_cache[seed]
        if hf["body"] is not None:
            world.remove(hf["body"])
        if hf["mesh"] is not None:
            scene.remove(hf["mesh"])
        m = tp.Mesh(geom, tp.MeshStandardMaterial())               # amplitude is just a z-scale of the [0,1] tile
        m.material.color = 0x8a7d6a; m.material.roughness = 0.97; m.receive_shadow = True
        m.position.set(0.0, 0.0, 0.0); m.scale.set(1.0, 1.0, cfg["amp"])
        scene.add(m)
        hf["body"] = world.add_static_trimesh(m); hf["mesh"] = m
        hf["H"] = H; hf["amp"] = cfg["amp"]

    rebuild_hf()
    for m in meshes:
        scene.add(m)

    # PERCEPTION: forward depth camera + accumulating elevation map -> the 45-cell scan (vs --analytic oracle).
    scanner = None if args.analytic else ForwardDepthScanner(
        rend, scene, meshes, bounds=(PATCH_X0 - 2.0, PATCH_X1 + 2.0, -PATCH_HALF_Y - 1.0, PATCH_HALF_Y + 1.0),
        noise=args.noise)
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
        # Spawn on the flat tapered edge (terrain~0) and walk in: art.reset() gives the zero-config
        # STRAIGHT-legs pose that folds during settle, which tips/penetrates if dropped onto a steep
        # 2-D patch. The tiny lift covers any residual taper height.
        foot = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))
        h = max(terr_h(SPAWN_X + dx, dy, hf) for dx, dy in foot)
        art.reset(tp.Vector3(SPAWN_X, 0.0, Z0 + h + 0.02)); state["last_act"] = np.zeros(12, np.float32)
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
        ahead, h_here = scanner.scan(art.root_state()) if scanner is not None else analytic_scan(art, hf)
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

    reset_spot(120)                                                  # initial drop-settle (lifted off the terrain)

    if not live:
        for _ in range(args.check):
            control_tick(use_keys=False)
        rs = art.root_state(); bx, by, bz = float(rs[0]), float(rs[1]), float(rs[2])
        h = terr_h(bx, by, hf)
        print(f"after {args.check} fwd steps: base=({bx:+.2f},{by:+.2f},{bz:.2f})  "
              f"terrain_under_base={h:.2f}  clearance={bz - h:.2f}  up_ok={bool(_quat_to_R(rs[3:7])[2,2] > 0.5)}")
        print("SPOTV2 HEIGHTFIELD PLAY HEADLESS CHECK: OK")
        return

    ui = tp.ImguiContext(canvas, rend)
    def on_resize(w, h):
        cam.aspect = w / max(h, 1); cam.update_projection_matrix(); rend.set_size(w, h)
    canvas.on_window_resize(on_resize)
    nf = [0]; r_down = [False]
    prev = {"amp": cfg["amp"], "seed": int(cfg["seed"])}; built = {"amp": cfg["amp"], "seed": int(cfg["seed"])}; rb_wait = [0]

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12); tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("spotv2 drive (heightfield)")
        tp.imgui.text(f"policy: {os.path.basename(model)}  (reloads {pol['reloads']})")
        vx, vy, wz = state["cmd"]
        tp.imgui.text(f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        rs = art.root_state(); h = terr_h(float(rs[0]), float(rs[1]), hf)
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
        _, cfg["amp"] = tp.imgui.slider_float("amplitude (m)", cfg["amp"], 0.0, 0.65)
        _, cfg["seed"] = tp.imgui.slider_int("shape seed", int(cfg["seed"]), 0, 15)
        if abs(cfg["amp"] - built["amp"]) > 1e-4 or int(cfg["seed"]) != built["seed"]:
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
        cs = int(cfg["seed"])                                        # debounced rebuild on amp/seed release
        if cfg["amp"] != prev["amp"] or cs != prev["seed"]:
            prev["amp"], prev["seed"] = cfg["amp"], cs; rb_wait[0] = 0
        elif abs(cfg["amp"] - built["amp"]) > 1e-4 or cs != built["seed"]:
            rb_wait[0] += 1
            if rb_wait[0] > 6:
                rebuild_hf(); reset_spot(); built["amp"], built["seed"] = cfg["amp"], cs
        control_tick(use_keys=True); render_chase(); ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()

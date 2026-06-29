"""Deploy/visualization viewer for the from-scratch flat-terrain Spot policy.

Loads scratch_flat.pt (or scratch_flat_latest.pt), runs the policy on CPU in a live
threepp GL window with a chase cam, keyboard steering, ImGui HUD, and hot-reload.

Flat ground only — no tent geometry, no depth scanner, no scan obs.

DRIVE (body frame, +x fwd / +y left):
    forward   UP / KP8        back      DOWN / KP2
    strafe L  LEFT / KP4      strafe R  RIGHT / KP6
    turn  L   N / KP7         turn  R   M / KP9
    R = reset to start

OBS contract (50-d scratch contract, normalized obs, CPU tgs_pcm/0.005):
  [0:3]   lin_b        body-frame linear velocity
  [3:6]   ang_b        body-frame angular velocity
  [6:9]   proj_g       gravity direction in body frame
  [9:12]  cmd          [vx, vy, wz]
  [12:24] qpos_rel     joint positions minus default_q (Isaac order)
  [24:36] qvel         joint velocities (Isaac order)
  [36:48] last_action  (Isaac order)
  [48:50] clock        [sin(2π·phi), cos(2π·phi)]

Obs is ALWAYS passed through norm.norm() before inference — the policy was trained
with normalize_obs=True; feeding raw obs gives garbage (wrong scale + units).

    python play_spot_flat.py                       # live window (default model)
    python play_spot_flat.py --model scratch_flat.pt
    python play_spot_flat.py --check 200           # headless smoke, 200 control ticks
    python play_spot_flat.py --pgs                 # use PGS/0.002 instead of tgs_pcm/0.005
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))   # scratch_distillation/
_SPOT_DIR  = os.path.dirname(_HERE)                        # examples/spot/
_PYROOT    = os.path.dirname(os.path.dirname(_SPOT_DIR))   # python/
sys.path.insert(0, _PYROOT)    # threepp / threepp.rl
sys.path.insert(0, _SPOT_DIR)  # spot_deploy / spot_terrain_env

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets, grid_texture, _quat_to_R,
                         default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, Z0)
from scratch_env import STIFF_GAINS
from scratch_clock import GAIT_PERIOD, DT

# --------------------------------------------------------------------------- #
#  Command envelope (matches the full-envelope training settings in scratch_env)
# --------------------------------------------------------------------------- #
VX_HI = 1.5
VY_HI = 0.8
WZ_HI = 1.2
VX_BACK = 1.0        # back speed limit
CRUISE = 0.8         # auto-forward speed when idle

GRAV = np.array([0.0, 0.0, -1.0])
DISP = 820


def _resolve_model(path):
    """Prefer the best checkpoint; fall back to the always-current _latest one."""
    if os.path.exists(path):
        return path
    latest = os.path.splitext(path)[0] + "_latest.pt"
    return latest if os.path.exists(latest) else path


def flat_obs(art, last_act, cmd, phi):
    """Build the 50-d scratch obs: 48 Isaac proprio + 2 clock dims.

    Layout (byte-identical to SpotScratchEnv._obs for the 48 proprio block):
      lin_b(3)  ang_b(3)  proj_g(3)  cmd(3)  qpos(12)  qvel(12)  last_act(12)
      sin(2pi*phi)(1)  cos(2pi*phi)(1)
    """
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7])
    Rt = R.T
    lin_b  = Rt @ rv[0:3]                           # body-frame linear vel
    ang_b  = Rt @ rv[3:6]                           # body-frame angular vel
    proj_g = Rt @ GRAV                              # gravity in body frame
    qpos   = art.joint_positions()[isaac_to_add] - default_q  # isaac order, relative
    qvel   = art.joint_velocities()[isaac_to_add]             # isaac order
    clk    = np.array([math.sin(2.0 * math.pi * phi),
                       math.cos(2.0 * math.pi * phi)], np.float32)
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, qvel, last_act, clk]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "scratch_flat.pt"),
                    help="checkpoint path (default: scratch_flat.pt, falls back to scratch_flat_latest.pt)")
    ap.add_argument("--check", type=int, default=0,
                    help="headless smoke: run N control ticks driving forward, print base pos + up_ok, exit")
    ap.add_argument("--pgs", action="store_true",
                    help="use PhysX PGS solver at 0.002 substep instead of tgs_pcm/0.005. "
                         "Both transfer; tgs_pcm matches the GpuSim TGS+PCM training contact model.")
    args = ap.parse_args()

    model = _resolve_model(args.model)
    if not os.path.exists(model):
        print(f"No policy at {model} — run train_scratch.py first.")
        sys.exit(0)

    live = args.check == 0
    dev = "cpu"    # leave the GPU to the trainer

    # Norm-aware policy load: the scratch policy was trained with normalize_obs=True,
    # so norm is ALWAYS present. Feeding raw obs to this net gives garbage.
    ac, norm, meta = load_policy(model, device=dev)
    assert norm is not None, (
        "Policy has no RunningNorm — was it trained with normalize_obs=True? "
        "Feeding raw obs to a normalized net gives garbage outputs."
    )
    pol = {"ac": ac, "norm": norm, "mt": os.path.getmtime(model), "reloads": 0}
    print(f"[policy] {os.path.basename(model)}  obs_dim={meta.get('obs_dim')}  norm=True")

    # tgs_pcm + 0.005 substep matches the GpuSim TGS+PCM training contact model on CPU.
    # --pgs uses PhysX's default PGS solver at 0.002; both transfer for this gait.
    if args.pgs:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    else:
        world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.005, max_substeps=8, tgs_pcm=True)
    print(f"[solver] {'PGS/0.002 (PhysX default CPU solver)' if args.pgs else 'tgs_pcm/0.005 (matches GpuSim training)'}")

    # Flat ground (no tent geometry).
    ground = tp.Mesh(tp.BoxGeometry(80, 80, 1.0), tp.MeshStandardMaterial())
    ground.position.set(0, 0, -0.5)
    world.add_static(ground)

    # Build Spot with STIFF_GAINS and mid-range friction:
    #   - STIFF_GAINS (stiffness 90 vs Isaac default 60): the from-scratch policy was TRAINED
    #     with these gains and needs a stiff skeleton to stand upright; deploy must match or
    #     the rigid rest pose expected by the network won't materialize (legs will sag).
    #   - friction 0.9: mid-range within the [0.6, 1.2] training DR band; gives clean push-off
    #     without being at either extreme. Matches the _SpotScratch build in scratch_env.py.
    foot_mat = world.create_material(0.9, 0.9, 0.0)
    art, meshes = build_spot(world, fetch_assets(), gains=STIFF_GAINS, foot_material=foot_mat)

    canvas = tp.Canvas("spot-scratch - flat terrain drive",
                       width=DISP, height=DISP, headless=not live)
    rend = tp.GLRenderer(canvas)
    rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic
    rend.tone_mapping_exposure = 1.1

    scene = tp.Scene()
    scene.background = tp.Background(0x9fb6cf)
    scene.add(tp.HemisphereLight(0xdce8f6, 0x55606c, 1.15))
    sun = tp.DirectionalLight(0xffffff, 2.7)
    sun.position.set(4, -6, 12)
    sun.cast_shadow = True
    scene.add(sun)

    floor = tp.Mesh(tp.PlaneGeometry(80, 80), tp.MeshStandardMaterial())
    floor.material.map = grid_texture(60)
    floor.material.color = 0xffffff
    floor.material.roughness = 0.95
    floor.receive_shadow = True
    scene.add(floor)

    for m in meshes:
        scene.add(m)

    cam = tp.PerspectiveCamera(46, 1.0, 0.05, 120)
    cam.up.set(0, 0, 1)
    cam.position.set(-2.6, -2.7, 1.4)

    # State: phi is the global phase clock ∈ [0, 1), advanced by DT/GAIT_PERIOD each tick.
    state = {
        "last_act": np.zeros(12, np.float32),
        "phi": 0.0,         # global phase clock ∈ [0, 1)
        "hdg_lock": None,
        "hdg_hold": True,
        "auto_fwd": True,
        "cmd": (0.0, 0.0, 0.0),
    }
    BACK, HEIGHT, LAG = 2.8, 1.4, 0.10

    def settle(n):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
            world.step(0.02)

    def reset_spot():
        art.reset(tp.Vector3(0, 0, Z0))
        state["last_act"] = np.zeros(12, np.float32)
        state["phi"] = 0.0          # reset phase to 0 on every episode start
        state["hdg_lock"] = None
        settle(40)

    def key_cmd():
        d = lambda *ks: any(canvas.is_key_down(k) for k in ks)
        vx   = (VX_HI   if d("UP",    "KP8") else 0.0) - (VX_BACK if d("DOWN",  "KP2") else 0.0)
        vy   = (VY_HI   if d("LEFT",  "KP4") else 0.0) - (VY_HI   if d("RIGHT", "KP6") else 0.0)
        turn = (WZ_HI   if d("N",     "KP7") else 0.0) - (WZ_HI   if d("M",     "KP9") else 0.0)
        return vx, vy, turn

    def control_tick(use_keys=True):
        vx, vy, turn = key_cmd() if use_keys else (0.0, 0.0, 0.0)
        if state["auto_fwd"] and vx == 0.0 and vy == 0.0 and turn == 0.0:
            vx = CRUISE    # idle -> cruise forward

        # Heading-hold P-controller: keeps Spot pointing straight when not actively turning.
        rs0 = art.root_state()
        R0  = _quat_to_R(rs0[3:7])
        yaw = math.atan2(float(R0[1, 0]), float(R0[0, 0]))
        if turn != 0.0 or not state["hdg_hold"]:
            wz = turn
            state["hdg_lock"] = yaw
        else:
            if state["hdg_lock"] is None:
                state["hdg_lock"] = yaw
            err = (yaw - state["hdg_lock"] + math.pi) % (2 * math.pi) - math.pi
            wz  = float(np.clip(-2.0 * err, -1.0, 1.0))

        cmd = np.array([vx, vy, wz], np.float32)
        state["cmd"] = (float(vx), float(vy), float(wz))

        # Build 50-d obs, normalize it, infer action.
        # The scratch policy was trained with normalize_obs=True — MUST call norm.norm().
        obs = flat_obs(art, state["last_act"], cmd, state["phi"])
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)[None]
            a = pol["ac"].act_mean(pol["norm"].norm(obs_t))[0].numpy()

        state["last_act"] = a
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)

        # Advance phase clock AFTER stepping the world, consistent with scratch_env.py
        # (phi aligns with the NEXT obs the policy will see).
        state["phi"] = (state["phi"] + DT / GAIT_PERIOD) % 1.0

    def render_chase():
        rs  = art.root_state()
        p   = np.array(rs[0:3], float)
        fwd = _quat_to_R(rs[3:7])[:, 0]
        fwd = np.array([fwd[0], fwd[1], 0.0])
        nrm = np.linalg.norm(fwd)
        fwd = fwd / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
        desired = p - fwd * BACK + np.array([0.0, 0.0, HEIGHT])
        cam.position.lerp(tp.Vector3(float(desired[0]), float(desired[1]), float(desired[2])), LAG)
        cam.look_at(float(p[0] + fwd[0] * 0.4), float(p[1] + fwd[1] * 0.4), float(p[2] + 0.1))
        rend.render(scene, cam)

    settle(120)

    # ----- headless smoke: drive forward N ticks, report base pos + up_ok, exit -----
    if not live:
        state["auto_fwd"] = True    # auto-forward during smoke
        for _ in range(args.check):
            control_tick(use_keys=False)
        rs = art.root_state()
        bx, by, bz = float(rs[0]), float(rs[1]), float(rs[2])
        R   = _quat_to_R(rs[3:7])
        up_ok = bool(R[2, 2] > 0.5)
        print(f"after {args.check} fwd steps: base=({bx:+.2f},{by:+.2f},{bz:.2f})  up_ok={up_ok}")
        print("SPOT-FLAT PLAY HEADLESS CHECK: OK")
        return

    # ----- live window -----
    ui = tp.ImguiContext(canvas, rend)

    def on_resize(w, h):
        cam.aspect = w / max(h, 1)
        cam.update_projection_matrix()
        rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    nf      = [0]
    r_down  = [False]

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("spot-scratch flat")
        tp.imgui.text(f"policy: {os.path.basename(model)}  (reloads {pol['reloads']})")
        vx, vy, wz = state["cmd"]
        tp.imgui.text(f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
        tp.imgui.text(f"phase phi: {state['phi']:.3f}")
        tp.imgui.text(f"obs: 50-d scratch (48 proprio + 2 clock), normalized")
        _, state["auto_fwd"]  = tp.imgui.checkbox("auto-forward (idle)", state["auto_fwd"])
        _, state["hdg_hold"]  = tp.imgui.checkbox("heading-hold (no-turn)", state["hdg_hold"])
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   |   R = reset")
        tp.imgui.end()

    def frame():
        if canvas.is_key_down("R"):
            if not r_down[0]:
                reset_spot()
                print("[reset]")
            r_down[0] = True
        else:
            r_down[0] = False

        nf[0] += 1
        if nf[0] % 40 == 0:    # hot-reload the checkpoint ~1/s
            try:
                mt = os.path.getmtime(model)
                if mt != pol["mt"]:
                    ac_new, norm_new, _ = load_policy(model, device=dev)
                    assert norm_new is not None, "reloaded checkpoint has no RunningNorm"
                    pol["ac"] = ac_new
                    pol["norm"] = norm_new
                    pol["mt"] = mt
                    pol["reloads"] += 1
                    state["phi"] = 0.0    # reset phase on reload
                    reset_spot()
                    print(f"[reload] {os.path.basename(model)} (#{pol['reloads']})")
            except Exception:
                pass

        control_tick(use_keys=True)
        render_chase()
        ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()

"""Drive the GPU-trained hexapod policy (hexapod_gpu.pt) in a rendered window.

    python play_gpu.py

Loads the policy trained by train_gpu.py and runs it on ONE hexapod in a normal CPU
PhysX world (for rendering). The observation is assembled exactly as gpu_env.py builds
it, so the policy sees what it trained on. Controls: W/S faster/slower, A/D turn,
R reset, Esc quit. Inference is on the CPU (one robot — instant).
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from arena import build_arena, reset_props
from gpu_ppo import load_policy
from hexapod import Hexapod
from hexapod_env import read_state

MODEL = os.path.join(_HERE, "hexapod_gpu.pt")

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend."); sys.exit(0)
if not os.path.exists(MODEL):
    print("No GPU policy yet — run train_gpu.py first (saves hexapod_gpu.pt)."); sys.exit(0)

ac, norm, meta = load_policy(MODEL, device="cpu")
JOINT_SCALE = np.array(meta["joint_scale"], np.float32)
CLOCK_HZ = float(meta.get("clock_hz", 1.6))

canvas = tp.Canvas("threepp — GPU-trained spider", width=1100, height=720)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True
scene = tp.Scene()
camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)

# Match the training physics cadence (the GPU env steps once at dt=1/30 per control
# step). A vanilla world (1/60 x 4 substeps) integrates differently and the GPU-trained
# gait degrades badly (sim-to-sim gap) — this single change recovers ~2 m/s vs ~0.6.
world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), fixed_timestep=1.0 / 30.0, max_substeps=1)
props = build_arena(scene, world)
spider = Hexapod(world, position=(0, 0.40, 0))
spider.add_to_scene(scene)
for m in spider.meshes:
    m.cast_shadow = True
# Policy was trained on GPU-DOF-order joints; CPU getters are add-order. Reorder both.
DOF = spider.art.dof_order()


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)
ui = tp.ImguiContext(canvas) if tp.HAS_IMGUI else None
CONTROL_DT = 1 / 30
clock = tp.Clock()
st = {"accum": 0.0, "clock": 0.0, "prev": (0.0, 0.0), "prev_yaw": 0.0,
      "vel": np.zeros(2, np.float32), "yawrate": 0.0, "key_r": False, "cmd": [0.6, 0.0]}
cam = [-3.5, 2.6, 0.0]


def observation():
    jp, jv, up, fx, fz, _, _, _ = read_state(spider)
    jp_gpu = np.empty(12, np.float32); jp_gpu[DOF] = jp   # add-order -> GPU order
    jv_gpu = np.empty(12, np.float32); jv_gpu[DOF] = jv
    v_fwd = st["vel"][0] * fx + st["vel"][1] * fz
    v_lat = st["vel"][0] * fz - st["vel"][1] * fx
    tail = np.array([up, fx, fz, v_fwd, v_lat, st["yawrate"],
                     math.cos(st["clock"]), math.sin(st["clock"]),
                     st["cmd"][0], st["cmd"][1]], np.float32)
    return np.concatenate([jp_gpu, jv_gpu * 0.1, tail]).astype(np.float32)


@torch.no_grad()
def control_step():
    st["clock"] += CLOCK_HZ * 2.0 * math.pi * CONTROL_DT
    o = torch.from_numpy(observation()).unsqueeze(0)
    a = ac.act_mean(norm.norm(o)).squeeze(0).numpy()
    targets_gpu = np.clip(a, -1.0, 1.0).astype(np.float32) * JOINT_SCALE  # GPU order
    spider.art.set_drive_targets(targets_gpu[DOF])                        # -> add-order
    world.step(CONTROL_DT)
    p = spider.position
    v = np.array([(p.x - st["prev"][0]) / CONTROL_DT, (p.z - st["prev"][1]) / CONTROL_DT], np.float32)
    st["vel"] = 0.8 * st["vel"] + 0.2 * v
    dy = (spider.yaw - st["prev_yaw"] + math.pi) % (2 * math.pi) - math.pi
    st["yawrate"] = 0.8 * st["yawrate"] + 0.2 * (dy / CONTROL_DT)
    st["prev"] = (p.x, p.z)
    st["prev_yaw"] = spider.yaw


def do_reset():
    spider.reset()
    reset_props(props)
    st["clock"] = 0.0
    p, f = spider.position, spider.forward
    st["prev"] = (p.x, p.z)
    st["prev_yaw"] = spider.yaw
    st["vel"][:] = 0.0
    st["yawrate"] = 0.0
    fl = math.hypot(f.x, f.z) or 1.0
    cam[:] = [p.x - 3.4 * f.x / fl, p.y + 2.3, p.z - 3.4 * f.z / fl]


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(250, 0)
    tp.imgui.begin("GPU-trained spider")
    tp.imgui.text("PhysX direct-GPU PPO (trained in minutes)")
    tp.imgui.text("speed: %.2f m/s" % float(math.hypot(st["vel"][0], st["vel"][1])))
    tp.imgui.text("cmd: fwd %.1f  turn %+.1f" % (st["cmd"][0], st["cmd"][1]))
    tp.imgui.text("W/S walk  A/D turn  R reset  Esc quit")
    tp.imgui.end()


def animate():
    if canvas.is_key_down("ESCAPE"):
        canvas.close()
        return
    r = canvas.is_key_down("R")
    if r and not st["key_r"]:
        do_reset()
    st["key_r"] = r

    fwd = 1.0 if canvas.is_key_down("W") else (0.3 if canvas.is_key_down("S") else 0.6)
    turn = (0.8 if canvas.is_key_down("A") else 0.0) - (0.8 if canvas.is_key_down("D") else 0.0)
    st["cmd"] = [fwd, turn]

    st["accum"] += min(clock.get_delta(), 0.1)
    while st["accum"] >= CONTROL_DT:
        st["accum"] -= CONTROL_DT
        control_step()

    p = spider.position
    f = spider.forward
    fl = math.hypot(f.x, f.z) or 1.0
    want = (p.x - 3.4 * f.x / fl, p.y + 2.3, p.z - 3.4 * f.z / fl)
    for i in range(3):
        cam[i] += (want[i] - cam[i]) * 0.07
    camera.position.set(*cam)
    camera.look_at(p.x, p.y + 0.1, p.z)
    renderer.render(scene, camera)
    if ui:
        ui.render(draw_ui)


print(__doc__)
canvas.animate(animate)

"""Watch the FROM-SCRATCH policy (no CPG gait) drive the hexapod.

    python play_scratch.py

A gait discovered from zero (no CPG), trained command-conditioned so you can drive
it: W/S walk fast/slow, A/D turn, R reset, Esc quit. Needs a PhysX build, a
display, and a trained policy (run train_scratch.py first).
"""
import math
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from arena import build_arena, reset_props
from hexapod import Hexapod
from hexapod_env import read_state
from scratch_env import CLOCK_HZ, JOINT_SCALE

MODEL = os.path.join(_HERE, "hexapod_scratch")

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend.")
    sys.exit(0)
if not os.path.exists(MODEL + ".zip") or not os.path.exists(MODEL + "_vecnorm.pkl"):
    print("No from-scratch policy — run train_scratch.py first.")
    sys.exit(0)

from stable_baselines3 import PPO

policy = PPO.load(MODEL, device="cpu")
with open(MODEL + "_vecnorm.pkl", "rb") as f:
    vn = pickle.load(f)
_mean = vn.obs_rms.mean.astype(np.float32)
_std = np.sqrt(vn.obs_rms.var.astype(np.float32) + vn.epsilon)
_clip = vn.clip_obs


def norm(o):
    return np.clip((o - _mean) / _std, -_clip, _clip).astype(np.float32)


canvas = tp.Canvas("threepp — from-scratch spider", width=1100, height=720)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True
scene = tp.Scene()
camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
camera.position.set(-3.5, 2.6, 0.0)

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
props = build_arena(scene, world)
spider = Hexapod(world, position=(0, 0.40, 0))
spider.add_to_scene(scene)
for m in spider.meshes:
    m.cast_shadow = True


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
    s = read_state(spider)
    jp, jv, up, fx, fz, _, _, _ = s
    tail = np.array([up, fx, fz, st["vel"][0], st["vel"][1], st["yawrate"],
                     math.cos(st["clock"]), math.sin(st["clock"]),
                     st["cmd"][0], st["cmd"][1]], np.float32)
    return np.concatenate([jp, jv * 0.1, tail]).astype(np.float32)


def control_step():
    st["clock"] += CLOCK_HZ * 2.0 * math.pi * CONTROL_DT
    a, _ = policy.predict(norm(observation()), deterministic=True)
    spider.art.set_drive_targets(np.clip(a, -1.0, 1.0).astype(np.float32) * JOINT_SCALE)
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
    tp.imgui.set_next_window_size(240, 0)
    tp.imgui.begin("From-scratch spider")
    tp.imgui.text("learned from zero, no CPG")
    tp.imgui.text("speed: %.2f m/s" % float(math.hypot(st["vel"][0], st["vel"][1])))
    tp.imgui.text("W/S walk  A/D turn")
    tp.imgui.text("R reset   Esc quit")
    tp.imgui.end()


def animate():
    if canvas.is_key_down("ESCAPE"):
        canvas.close()
        return
    r = canvas.is_key_down("R")
    if r and not st["key_r"]:
        do_reset()
    st["key_r"] = r

    # WASD -> command. Forward stays in the trained range [0.3, 1.0]; turn is A/D.
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

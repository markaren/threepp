"""Drive the hexapod with the trained residual policy (Stage 2).

    python play.py                       # loads hexapod_policy.zip
    python play.py --model other.zip

  W / S walk · A / D turn · Esc quit

Same WASD control as spider_demo.py, but the learned policy adds residual joint
corrections on top of the CPG gait (deterministic / mean action). Needs a PhysX
build, a display, and a trained policy (run train.py first).
"""
import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod import Hexapod
from hexapod_env import RESIDUAL_SCALE, make_observation

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(_HERE, "hexapod_policy"))
args = ap.parse_args()

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend.")
    sys.exit(0)
if not os.path.exists(args.model if args.model.endswith(".zip") else args.model + ".zip"):
    print("No policy at %s(.zip) — run train.py first." % args.model)
    sys.exit(0)

from stable_baselines3 import PPO

policy = PPO.load(args.model, device="cpu")

canvas = tp.Canvas("threepp — spider (trained policy)", width=1100, height=720)
renderer = tp.GLRenderer(canvas)
renderer.set_clear_color(0x222a32)
scene = tp.Scene()
scene.background = 0x222a32
camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
camera.position.set(-3.5, 2.6, 0.0)
scene.add(tp.HemisphereLight(0xffffff, 0x334433, 1.0))
sun = tp.DirectionalLight(0xffffff, 2.0)
sun.position.set(6, 12, 5)
scene.add(sun)

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
gm = tp.MeshStandardMaterial()
gm.color = 0x55624f
ground = tp.Mesh(tp.BoxGeometry(120, 1, 120), gm)
ground.position.y = -0.5
scene.add(ground)
world.add_static(ground)
spider = Hexapod(world, position=(0, 0.40, 0))
spider.add_to_scene(scene)


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

CONTROL_DT = 1 / 30
clock = tp.Clock()
state = {"accum": 0.0, "prev": (0.0, 0.0), "prev_yaw": 0.0,
         "vel": np.zeros(2, np.float32), "yawrate": 0.0}
cam = [-3.5, 2.6, 0.0]


def control_step(command):
    spider.set_command(command[0], command[1])
    obs = make_observation(spider, state["vel"], state["yawrate"], command)
    action, _ = policy.predict(obs, deterministic=True)
    residuals = (np.clip(action, -1.0, 1.0) * RESIDUAL_SCALE).tolist()
    spider.update(CONTROL_DT, residuals)
    world.step(CONTROL_DT)
    # finite-diff body velocity + yaw rate (matches the env's measurement)
    p = spider.position
    v = np.array([(p.x - state["prev"][0]) / CONTROL_DT, (p.z - state["prev"][1]) / CONTROL_DT], np.float32)
    state["vel"] = 0.8 * state["vel"] + 0.2 * v
    dy = spider.yaw - state["prev_yaw"]
    dy = (dy + math.pi) % (2 * math.pi) - math.pi
    state["yawrate"] = 0.8 * state["yawrate"] + 0.2 * (dy / CONTROL_DT)
    state["prev"] = (p.x, p.z)
    state["prev_yaw"] = spider.yaw


def animate():
    if canvas.is_key_down("ESCAPE"):
        canvas.close()
        return
    command = (
        (1.0 if canvas.is_key_down("W") else 0.0) - (1.0 if canvas.is_key_down("S") else 0.0),
        (1.0 if canvas.is_key_down("A") else 0.0) - (1.0 if canvas.is_key_down("D") else 0.0),
    )
    state["accum"] += min(clock.get_delta(), 0.1)
    while state["accum"] >= CONTROL_DT:   # fixed 30 Hz control, matching training
        state["accum"] -= CONTROL_DT
        control_step(command)

    p = spider.position
    f = spider.forward
    fl = math.hypot(f.x, f.z) or 1.0
    want = (p.x - 3.4 * f.x / fl, p.y + 2.3, p.z - 3.4 * f.z / fl)
    for i in range(3):
        cam[i] += (want[i] - cam[i]) * 0.07
    camera.position.set(*cam)
    camera.look_at(p.x, p.y + 0.1, p.z)
    renderer.render(scene, camera)


print(__doc__)
canvas.animate(animate)

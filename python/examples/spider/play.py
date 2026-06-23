"""Drive the hexapod with the trained residual-on-CPG policy (owned GPU stack).

    python play.py                       # loads hexapod_policy.pt
    python play.py --model other.pt

  W / S walk · A / D turn · R reset · Esc quit

Same WASD control as spider_demo.py, but the policy (trained on the GPU via train_hexapod.py)
adds residual joint corrections on top of the CPG gait. The deploy runs ONE hexapod on a CPU
world and feeds the policy the exact same `make_obs` the trainer used (single source of truth),
so train and deploy can't drift. Needs a PhysX build, a display, and hexapod_policy.pt.
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
from arena import build_arena, reset_props
from hexapod import Hexapod
from hexapod_gpu_env import CONFIG as ENV_CONFIG
from hexapod_gpu_env import make_obs, quat_to_frame
from threepp.rl import load_policy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(_HERE, "hexapod_policy.pt"))
args = ap.parse_args()

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend."); sys.exit(0)
if not os.path.exists(args.model):
    print("No policy at %s — run train_hexapod.py first." % args.model); sys.exit(0)

ac, norm, meta = load_policy(args.model, device="cpu")
# the deploy contract comes from the policy's meta; guard against this module having drifted
for k in ("dt", "residual_scale"):
    if k in meta and abs(float(meta[k]) - float(ENV_CONFIG[k])) > 1e-9:
        raise SystemExit(f"hexapod_gpu_env.{k}={ENV_CONFIG[k]} drifted from the trained policy "
                         f"(meta {k}={meta[k]}) — retrain or revert before deploying.")
CONTROL_DT = float(meta["dt"])
RESIDUAL_SCALE = float(meta["residual_scale"])


def observation(spider, vel, yawrate, command):
    """Build the 34-d obs for ONE robot with the SAME make_obs the trainer used."""
    jp, jv = spider.joint_states()
    q = spider.chassis.quaternion
    up, fx, fz, _ = quat_to_frame(torch.tensor([[q.x, q.y, q.z, q.w]]))
    return make_obs(torch.tensor([jp]), torch.tensor([jv]), up, fx, fz,
                    torch.tensor([float(vel[0])]), torch.tensor([float(vel[1])]),
                    torch.tensor([float(yawrate)]), torch.tensor([float(spider.psi)]),
                    torch.tensor([[float(command[0]), float(command[1])]]))


canvas = tp.Canvas("threepp — spider (trained policy)", width=1100, height=720)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True
scene = tp.Scene()
camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
camera.position.set(-3.5, 2.6, 0.0)

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
props = build_arena(scene, world)  # lit ground arena: grid, perimeter, rocks, knock-around props

spider = Hexapod(world, position=(0, 0.40, 0))
spider.add_to_scene(scene)
for m in spider.meshes:
    m.cast_shadow = True


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

clock = tp.Clock()
state = {"accum": 0.0, "prev": (0.0, 0.0), "prev_yaw": 0.0,
         "vel": np.zeros(2, np.float32), "yawrate": 0.0, "key_r": False}
cam = [-3.5, 2.6, 0.0]


def do_reset():
    spider.reset()
    reset_props(props)
    p, f = spider.position, spider.forward
    state["prev"] = (p.x, p.z)
    state["prev_yaw"] = spider.yaw
    state["vel"][:] = 0.0
    state["yawrate"] = 0.0
    fl = math.hypot(f.x, f.z) or 1.0
    cam[:] = [p.x - 3.4 * f.x / fl, p.y + 2.3, p.z - 3.4 * f.z / fl]  # snap cam behind


@torch.no_grad()
def control_step(command):
    spider.set_command(command[0], command[1])
    obs = observation(spider, state["vel"], state["yawrate"], command)
    action = ac.act_mean(norm.norm(obs)).squeeze(0).clamp(-1.0, 1.0)
    spider.update(CONTROL_DT, (action * RESIDUAL_SCALE).tolist())
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
    r = canvas.is_key_down("R")
    if r and not state["key_r"]:
        do_reset()
    state["key_r"] = r
    command = (
        (1.0 if canvas.is_key_down("W") else 0.0) - (1.0 if canvas.is_key_down("S") else 0.0),
        (1.0 if canvas.is_key_down("A") else 0.0) - (1.0 if canvas.is_key_down("D") else 0.0),
    )
    state["accum"] += min(clock.get_delta(), 0.1)
    while state["accum"] >= CONTROL_DT:   # fixed-rate control, matching training
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

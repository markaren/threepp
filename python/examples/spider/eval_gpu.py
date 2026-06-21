"""Headless command-following eval of the GPU-trained policy (no window).

Runs the policy on one CPU hexapod for a few commands and reports speed + heading
change — the same test used on the earlier CPU-trained policies, so numbers compare
directly. Confirms the policy walks and is controllable before you open play_gpu.py.

    python eval_gpu.py
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
from gpu_ppo import load_policy
from hexapod import Hexapod
from hexapod_env import read_state

MODEL = os.path.join(_HERE, "hexapod_gpu.pt")
if not tp.HAS_PHYSX:
    print("no PhysX"); sys.exit(0)
if not os.path.exists(MODEL):
    print("no hexapod_gpu.pt yet"); sys.exit(0)

ac, norm, meta = load_policy(MODEL, device="cpu")
JOINT_SCALE = np.array(meta["joint_scale"], np.float32)
CLOCK_HZ = float(meta.get("clock_hz", 1.6))
DT = 1.0 / 30.0

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
ground = tp.Mesh(tp.BoxGeometry(200, 1, 200), tp.MeshStandardMaterial())
ground.position.y = -0.5
world.add_static(ground)
spider = Hexapod(world, position=(0, 0.40, 0))
# The policy was trained on GPU-DOF-order joints (PhysX cache order); the CPU getters
# are in add-order. Reorder both obs joints and action targets through this map.
DOF = spider.art.dof_order()  # add-order i -> GPU slot


@torch.no_grad()
def run(cmd, secs=6.0, settle=0.5):
    spider.reset((0, 0.40, 0))
    s = {"clock": 0.0, "prev": (0.0, 0.0), "prev_yaw": 0.0,
         "vel": np.zeros(2, np.float32), "yawrate": 0.0}
    # settle
    for _ in range(int(settle / DT)):
        spider.art.set_drive_targets(np.zeros(12, np.float32))
        world.step(DT)
    p0 = spider.position
    yaw0 = spider.yaw
    s["prev"] = (p0.x, p0.z); s["prev_yaw"] = yaw0
    ups = []
    for _ in range(int(secs / DT)):
        jp, jv, up, fx, fz, _, _, _ = read_state(spider)
        jp_gpu = np.empty(12, np.float32); jp_gpu[DOF] = jp      # add-order -> GPU order
        jv_gpu = np.empty(12, np.float32); jv_gpu[DOF] = jv
        v_fwd = s["vel"][0] * fx + s["vel"][1] * fz
        v_lat = s["vel"][0] * fz - s["vel"][1] * fx
        tail = np.array([up, fx, fz, v_fwd, v_lat, s["yawrate"],
                         math.cos(s["clock"]), math.sin(s["clock"]), cmd[0], cmd[1]], np.float32)
        o = torch.from_numpy(np.concatenate([jp_gpu, jv_gpu * 0.1, tail]).astype(np.float32)).unsqueeze(0)
        a = ac.act_mean(norm.norm(o)).squeeze(0).numpy()
        targets_gpu = np.clip(a, -1, 1).astype(np.float32) * JOINT_SCALE  # GPU order
        spider.art.set_drive_targets(targets_gpu[DOF])                    # -> add-order
        world.step(DT)
        s["clock"] += CLOCK_HZ * 2 * math.pi * DT
        p = spider.position
        v = np.array([(p.x - s["prev"][0]) / DT, (p.z - s["prev"][1]) / DT], np.float32)
        s["vel"] = 0.8 * s["vel"] + 0.2 * v
        dy = (spider.yaw - s["prev_yaw"] + math.pi) % (2 * math.pi) - math.pi
        s["yawrate"] = 0.8 * s["yawrate"] + 0.2 * (dy / DT)
        s["prev"] = (p.x, p.z); s["prev_yaw"] = spider.yaw
        ups.append(up)
    p1 = spider.position
    dist = math.hypot(p1.x - p0.x, p1.z - p0.z)
    dyaw = math.degrees((spider.yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return dist / secs, dyaw, float(np.mean(ups))


for label, cmd in [("straight  ", [1.0, 0.0]), ("turn LEFT ", [1.0, 0.8]),
                   ("turn RIGHT", [1.0, -0.8]), ("slow      ", [0.3, 0.0])]:
    sp, dyaw, up = run(cmd)
    print(f"cmd {label} -> speed {sp:4.2f} m/s   heading {dyaw:+5.0f} deg   up {up:.2f}")

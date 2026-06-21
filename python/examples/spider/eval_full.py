"""Comprehensive command-following eval of the GPU-trained hexapod (CPU deploy path).

Runs the deterministic policy across a grid of (forward, turn) commands and reports
forward speed, heading-change rate, lateral drift, and uprightness — the controllability
report card. CPU PhysX + CPU inference (the deployment path play_gpu.py uses).

    python eval_full.py
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

ac, norm, meta = load_policy(os.path.join(_HERE, "hexapod_gpu.pt"), device="cpu")
JOINT_SCALE = np.array(meta["joint_scale"], np.float32)
CLOCK_HZ = float(meta.get("clock_hz", 1.6))
DT = 1.0 / 30.0

# Match the training physics cadence: the GPU env steps ONCE at dt=1/30 per control
# step. A vanilla world (1/60, 4 substeps) integrates differently -> sim-to-sim gap.
world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), fixed_timestep=1.0 / 30.0, max_substeps=1)
ground = tp.Mesh(tp.BoxGeometry(400, 1, 400), tp.MeshStandardMaterial())
ground.position.y = -0.5
world.add_static(ground)
spider = Hexapod(world, position=(0, 0.40, 0))
DOF = spider.art.dof_order()


@torch.no_grad()
def run(cmd, secs=8.0, settle=0.6):
    spider.reset((0, 0.40, 0))
    s = {"clock": 0.0, "prev": (0.0, 0.0), "prev_yaw": 0.0,
         "vel": np.zeros(2, np.float32), "yawrate": 0.0}
    for _ in range(int(settle / DT)):
        spider.art.set_drive_targets(np.zeros(12, np.float32))
        world.step(DT)
    p0, yaw0 = spider.position, spider.yaw
    s["prev"] = (p0.x, p0.z); s["prev_yaw"] = yaw0
    ups, lat = [], []
    for _ in range(int(secs / DT)):
        jp, jv, up, fx, fz, _, _, _ = read_state(spider)
        jp_g = np.empty(12, np.float32); jp_g[DOF] = jp
        jv_g = np.empty(12, np.float32); jv_g[DOF] = jv
        v_fwd = s["vel"][0] * fx + s["vel"][1] * fz
        v_lat = s["vel"][0] * fz - s["vel"][1] * fx
        tail = np.array([up, fx, fz, v_fwd, v_lat, s["yawrate"],
                         math.cos(s["clock"]), math.sin(s["clock"]), cmd[0], cmd[1]], np.float32)
        o = torch.from_numpy(np.concatenate([jp_g, jv_g * 0.1, tail]).astype(np.float32)).unsqueeze(0)
        a = ac.act_mean(norm.norm(o)).squeeze(0).numpy()
        tg = np.clip(a, -1, 1).astype(np.float32) * JOINT_SCALE
        spider.art.set_drive_targets(tg[DOF])
        world.step(DT)
        s["clock"] += CLOCK_HZ * 2 * math.pi * DT
        p = spider.position
        v = np.array([(p.x - s["prev"][0]) / DT, (p.z - s["prev"][1]) / DT], np.float32)
        s["vel"] = 0.8 * s["vel"] + 0.2 * v
        dy = (spider.yaw - s["prev_yaw"] + math.pi) % (2 * math.pi) - math.pi
        s["yawrate"] = 0.8 * s["yawrate"] + 0.2 * (dy / DT)
        s["prev"] = (p.x, p.z); s["prev_yaw"] = spider.yaw
        ups.append(up); lat.append(abs(v_lat))
    p1 = spider.position
    dist = math.hypot(p1.x - p0.x, p1.z - p0.z)
    dyaw = math.degrees((spider.yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return dict(speed=dist / secs, yaw_rate=dyaw / secs, lat=float(np.mean(lat)),
                up=float(np.mean(ups)), fell=float(np.min(ups)) < 0.3)


GRID = [
    ("straight       ", [1.0, 0.0]),
    ("medium         ", [0.6, 0.0]),
    ("slow           ", [0.3, 0.0]),
    ("turn L strong  ", [1.0, 0.8]),
    ("turn L mild    ", [1.0, 0.4]),
    ("turn R strong  ", [1.0, -0.8]),
    ("turn R mild    ", [1.0, -0.4]),
    ("pivot L (slow) ", [0.3, 0.8]),
    ("pivot R (slow) ", [0.3, -0.8]),
]
print(f"{'command':16s} {'speed':>7s} {'yaw/s':>8s} {'lat':>6s} {'up':>5s}  note")
print("-" * 62)
for label, cmd in GRID:
    r = run(cmd)
    # expected turn sign: +turn command -> left -> +yaw (our convention)
    want = "L" if cmd[1] > 0 else ("R" if cmd[1] < 0 else "straight")
    got = "L" if r["yaw_rate"] > 5 else ("R" if r["yaw_rate"] < -5 else "straight")
    ok = "OK" if want == got else "MISS"
    fell = " FELL" if r["fell"] else ""
    print(f"{label:16s} {r['speed']:6.2f}  {r['yaw_rate']:+7.1f}  {r['lat']:5.2f} {r['up']:4.2f}  "
          f"want {want:>8s}/{got:<8s} {ok}{fell}")

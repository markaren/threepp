"""Verify the trained hexapod policy: does it track commands and beat the open-loop CPG?

Holds a fixed command, runs the deterministic policy across a batch, and measures the achieved
forward speed / yaw rate / uprightness — compared against the open-loop gait (zero residual).
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from hexapod_gpu_env import ACT_DIM, MAX_SPEED, MAX_YAW, HexapodGpuEnv
from threepp.rl import load_policy

ac, norm, meta = load_policy(os.path.join(_HERE, "hexapod_policy.pt"), device="cuda")
env = HexapodGpuEnv(num_envs=256)
dev = env.sim.device


@torch.no_grad()
def run(command, policy):
    """Hold `command` fixed for 300 steps; return (mean forward speed, mean yaw rate, upright frac, fell %)."""
    obs = env.reset()
    cmd = torch.tensor(command, device=dev)
    fwd, yaw, up = [], [], []
    for _ in range(300):
        env.cmd[:] = cmd                       # re-force each step (defeat resampling/reset)
        a = ac.act_mean(norm.norm(obs)) if policy else torch.zeros(env.K, ACT_DIM, device=dev)
        obs, _, _, _, _ = env.step(a)
        fwd.append(env.ema_v[:, 0] * env.fx + env.ema_v[:, 1] * env.fz)
        yaw.append(env.ema_w)
        up.append(env.up)
    fwd, yaw, up = torch.stack(fwd), torch.stack(yaw), torch.stack(up)
    fell = (up[-150:].min(0).values < 0.0).float().mean().item() * 100
    return fwd[-150:].mean().item(), yaw[-150:].mean().item(), up[-150:].mean().item(), fell


print(f"target forward at cmd=1.0: {MAX_SPEED:.2f} m/s   target yaw at cmd=1.0: {MAX_YAW:.2f} rad/s\n")
for label, cmd in (("walk forward (1,0)", [1.0, 0.0]), ("turn left  (0,1)", [0.0, 1.0])):
    pf, py, pu, pflr = run(cmd, policy=True)
    gf, gy, gu, gflr = run(cmd, policy=False)
    print(f"{label}:")
    print(f"  policy    : fwd={pf:+.3f} m/s  yaw={py:+.3f} rad/s  upright={pu:.3f}  fell={pflr:.0f}%")
    print(f"  open CPG  : fwd={gf:+.3f} m/s  yaw={gy:+.3f} rad/s  upright={gu:.3f}  fell={gflr:.0f}%")

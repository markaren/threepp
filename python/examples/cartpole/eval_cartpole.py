"""Verify the cart-pole swing-up from the worst case: start HANGING STRAIGHT DOWN, run the
deterministic policy, and measure whether it swings up and holds. Unfakeable proof of control.
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from cartpole_env import CartPoleEnv, RAIL
from threepp.rl import load_policy

model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "cartpole_swingup.pt")
ac, norm, meta = load_policy(model, device="cuda")
env = CartPoleEnv(num_envs=256)
env.reset()
dev = env.sim.device

# Force every env to hang straight DOWN (pole at pi), zero velocity.
pos = torch.zeros(env.K, env.sim.dof, device=dev); pos[:, 1] = math.pi
env.sim.set_joint_state(torch.arange(env.K, device=dev), pos, torch.zeros_like(pos))
env.sim.step(env.dt)
obs = env._obs()

reached = torch.zeros(env.K, dtype=torch.bool, device=dev)
t_up = torch.full((env.K,), -1.0, device=dev)
ups, cart_x, cart_v = [], [], []
with torch.no_grad():
    for step in range(400):                       # ~6.7 s
        obs, _, _, _, _ = env.step(ac.act_mean(norm.norm(obs)))
        jp, jv = env.sim.joint_pos, env.sim.joint_vel
        up = torch.cos(jp[:, 1])
        newly = (up > 0.9) & ~reached
        t_up[newly] = step / 60.0
        reached |= (up > 0.9)
        ups.append(up)
        cart_x.append(jp[:, 0].abs())             # cart distance from centre
        cart_v.append(jv[:, 0].abs())             # cart speed (the movement to minimize)

up_hist = torch.stack(ups)
final_up = up_hist[-100:].mean(0)
# cart motion over the last ~1.7 s, restricted to the cart-poles that are actually balanced
held = final_up > 0.8
final_x = torch.stack(cart_x[-100:]).mean(0)[held]
final_v = torch.stack(cart_v[-100:]).mean(0)[held]
print("start: hanging straight down")
print(f"  swung up to vertical:   {reached.float().mean().item()*100:3.0f}% of cart-poles")
print(f"  still up at the end:    {held.float().mean().item()*100:3.0f}%")
print(f"  mean swing-up time:     {t_up[reached].mean().item():.2f} s")
print(f"  mean final uprightness: {final_up.mean().item():.3f}")
print(f"  balance cart |x|:       {final_x.mean().item():.3f} m       (drift off centre)")
print(f"  balance cart |v|:       {final_v.mean().item():.3f} m/s     <- MOVEMENT (lower = stiller)")
peak_x = torch.stack(cart_x).max(0).values.mean().item()
print(f"  peak cart |x| (swing-up): {peak_x:.3f} m   of {RAIL:.1f} m rail  <- TRAVEL (rail = wall)")

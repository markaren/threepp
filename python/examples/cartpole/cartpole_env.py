"""GPU-vectorized cart-pole SWING-UP (cart + single pole).

The GPU plumbing — CUDA context, the direct-GPU batch, DOF-order remap, device buffers,
sync — lives in GpuSim. This file is just the task: build the scene, define the observation,
the reward, and the reset. The pole starts at a random angle (often hanging down), there is no
fall termination, and the reward is the pole height — so the only way to score is to pump the
cart and swing the pole up, then balance it.

This module is the SINGLE SOURCE OF TRUTH for the cart-pole: the timestep, force scale, rail
and observation (CONFIG + make_obs) live here and are imported by both the trainer and the
deployment viewer, so the two can never silently drift.
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from cartpole import CartPole
from threepp.rl import GpuSim

# ---- single source of truth: config + observation ----------------------------
CONTROL_HZ = 60
DT = 1.0 / CONTROL_HZ
MAX_SUBSTEPS = 1          # deploy world MUST use this so world.step(DT) == one GPU step
RAIL = 2.2               # cart prismatic limit (m)
FORCE_SCALE = 45.0       # action [-1,1] -> cart force (N)
V_SCALE = 0.2            # cart-velocity obs scale
W_SCALE = 0.1            # pole-angular-velocity obs scale
OBS_DIM = 5
ACT_DIM = 1

# Persisted into the policy checkpoint so deploy reconstructs + asserts the contract.
CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "max_substeps": MAX_SUBSTEPS,
          "rail": RAIL, "force_scale": FORCE_SCALE, "v_scale": V_SCALE, "w_scale": W_SCALE}


def make_obs(cart_x, cart_v, theta, theta_dot):
    """The 5-d observation, defined ONCE. Works for batched tensors (training) and 1-element
    tensors (single-robot deploy): [cart_x/rail, cart_v*s, sin th, cos th, w*s]."""
    return torch.stack([cart_x / RAIL, cart_v * V_SCALE,
                        torch.sin(theta), torch.cos(theta), theta_dot * W_SCALE], dim=-1)


class CartPoleEnv:
    def __init__(self, num_envs=4096, episode_s=10.0, device="cuda", seed=0):
        self.sim = GpuSim(num_envs, lambda world, i: CartPole(world, x0=i * 3.0), device=device)
        self.K, self.dt = num_envs, DT
        self.max_steps = int(episode_s * CONTROL_HZ)
        self.g = torch.Generator(device=self.sim.device).manual_seed(seed)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.sim.device)

    def _start_state(self, n):
        # cart centred, pole at a random angle (often hanging down) -> learn to swing up
        pos = torch.zeros(n, self.sim.dof, device=self.sim.device)
        pos[:, 1] = (torch.rand(n, device=self.sim.device, generator=self.g) * 2 - 1) * math.pi
        return pos, torch.zeros_like(pos)

    def _obs(self):
        jp, jv = self.sim.joint_pos, self.sim.joint_vel        # [K, dof] add-order: 0=cart, 1=pole
        return make_obs(jp[:, 0], jv[:, 0], jp[:, 1], jv[:, 1])

    def reset(self):
        idx = torch.arange(self.K, device=self.sim.device)
        self.sim.set_joint_state(idx, *self._start_state(self.K))
        self.steps.zero_()
        self.sim.step(self.dt)
        return self._obs()

    @torch.no_grad()
    def step(self, actions):
        """Returns (next_obs, reward, done, terminal_obs, is_timeout). Every done here IS a
        timeout (this env has no failure terminal), so is_timeout = done — the trainer bootstraps
        V(terminal_obs) on all of them. terminal_obs is the real post-step obs for done envs
        (captured before reset overwrites it)."""
        a = actions.clamp(-1.0, 1.0)
        force = torch.zeros(self.K, self.sim.dof, device=self.sim.device)
        force[:, 0] = a[:, 0] * FORCE_SCALE                    # cart is joint 0
        self.sim.apply_force(force)
        self.sim.step(self.dt)
        self.steps += 1

        jp, jv = self.sim.joint_pos, self.sim.joint_vel
        up = torch.cos(jp[:, 1])                               # +1 up, -1 hanging down
        rew = (up + 1.5 * (up > 0.9).float()
               - 0.01 * (jp[:, 0] / RAIL) ** 2
               - 0.0005 * jv[:, 1] ** 2
               - 0.003 * a[:, 0] ** 2)
        done = self.steps >= self.max_steps
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self.sim.set_joint_state(d, *self._start_state(d.numel()))
            self.steps[d] = 0
            self.sim.read()
            obs = self._obs()
        else:
            obs = term_obs
        return obs, rew, done, term_obs, done   # is_timeout = done (no failure terminal)


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = CartPoleEnv(num_envs=256)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    for _ in range(300):
        obs, rew, done, term, to = env.step(torch.rand(env.K, 1, device=env.sim.device) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"300 steps ok; reward [{rew.min():.2f},{rew.max():.2f}]")
    print("CARTPOLE ENV SELFTEST: PASS")

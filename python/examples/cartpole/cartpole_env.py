"""GPU-vectorized cart-pole SWING-UP (cart + single pole).

The choreography — CUDA context, direct-GPU batch, steps/timeout bookkeeping, terminal-obs
capture, partial resets — lives in threepp.rl (GpuSim + VecTask). This file is ONLY the task:
build the robot, define the observation, the reward terms, and the reset. The pole starts at a
random angle (often hanging down), there is no fall termination, and the reward is the pole
height — so the only way to score is to pump the cart and swing the pole up, then balance it.

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
from threepp.rl import VecTask

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


class CartPoleEnv(VecTask):
    control_hz = CONTROL_HZ
    act_dim = ACT_DIM
    control = "force"

    def __init__(self, num_envs=4096, episode_s=10.0, device="cuda", seed=0):
        self.episode_s = episode_s
        super().__init__(num_envs, lambda world, i: CartPole(world, x0=i * 3.0),
                         device=device, seed=seed)

    def on_reset(self, idx):
        # cart centred, pole at a random angle (often hanging down) -> learn to swing up
        n = idx.numel()
        pos = torch.zeros(n, self.sim.dof, device=self.device)
        pos[:, 1] = (torch.rand(n, device=self.device, generator=self.g) * 2 - 1) * math.pi
        self.sim.set_joint_state(idx, pos, torch.zeros_like(pos))

    def act(self, a):
        force = torch.zeros(self.K, self.sim.dof, device=self.device)
        force[:, 0] = a[:, 0] * FORCE_SCALE                    # cart is joint 0
        return force

    def observe(self, s):
        # add-order: joint 0 = cart, joint 1 = pole
        return make_obs(s.joint_pos[:, 0], s.joint_vel[:, 0], s.joint_pos[:, 1], s.joint_vel[:, 1])

    def reward_terms(self, s, a):
        up = torch.cos(s.joint_pos[:, 1])                      # +1 up, -1 hanging down
        x_n = s.joint_pos[:, 0] / RAIL                         # normalized cart position (+-1 at the rail)
        v_cart, v_pole = s.joint_vel[:, 0], s.joint_vel[:, 1]
        a_eff = a[:, 0]                                        # action effort

        # Continuous "uprightness" weight: ~1 when perfectly up, decaying smoothly as the pole falls.
        # Smooth (vs a hard up>0.9 gate) so the agent ANTICIPATES the stabilization cost and eases into
        # the balance gracefully. SHARP (exp(-8) not exp(-3)): a broad gate makes the v_pole^2 stabilization
        # term live during the final swing, taxing the very velocity needed to carry the pole over the top
        # -> it either can't hold or never swings up (stay-down local optimum). exp(-8) keeps it ~off below
        # up~0.85 (the whole pump) and engages only for the fine balance.
        upright_weight = torch.exp(-8.0 * (1.0 - up))
        # Energy-matching: penalize deviation from the ideal energy state (up=1, pole still) -> an
        # efficient pump instead of thrashing.
        energy_error = torch.abs((up - 1.0) + 0.1 * v_pole ** 2)

        return {
            "up": up,                                          # point the pole up
            "up_bonus": 2.5 * upright_weight,                  # BONUS for being up (sharp-gated): makes the
                                                               #   long resonant pump clearly worth it -> reliably
                                                               #   escapes the "never move / stay down" optimum
            "energy": -0.4 * energy_error,                     # pump efficiently (stronger -> stronger pull off bottom)
            "rail": -5.0 * torch.relu(x_n.abs() - 0.5) ** 2,   # soft wall: stay off the rail
            "effort": -0.01 * a_eff ** 2,                      # mild action effort (glide, don't thrash)
            # LQR-style stabilization, scaled by uprightness so it engages smoothly near the top:
            "stabilize": -upright_weight * (1.0 * x_n ** 2     # lock the cart to centre
                                            + 0.5 * v_cart ** 2   # kill cart velocity (minimize movement)
                                            + 0.1 * v_pole ** 2),  # kill pole wobble
        }

    def config(self):
        return {**super().config(), **CONFIG}


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = CartPoleEnv(num_envs=256)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    for _ in range(300):
        obs, rew, done, term, to = env.step(torch.rand(env.K, 1, device=env.device) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    assert bool((to == done).all()), "timeout-only task: every done must be a truncation"
    print(f"300 steps ok; reward [{rew.min():.2f},{rew.max():.2f}]")
    print("per-term:", env.stats_line() or "(no episode finished yet)")
    print("CARTPOLE ENV SELFTEST: PASS")

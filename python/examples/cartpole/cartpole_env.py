"""GPU-resident vectorized cart-pole SWING-UP (cart + single pole).

K cart-poles in ONE PhysX direct-GPU scene. Each control step the policy outputs a single
cart force (applied via the direct-GPU joint-force API), physics steps once, and joint state
is read back into torch cuda tensors — obs / reward / reset all run in torch on the GPU.

The pole starts at a random angle (often hanging down), there is NO fall termination, and the
reward is the pole height (cos of its angle) plus a bonus near the top — so the only way to
score is to pump the cart and swing the pole up, then balance it.

This module is the SINGLE SOURCE OF TRUTH for the cart-pole: the timestep, force scale, rail,
observation layout and its scales live here (CONFIG + make_obs) and are imported by both the
trainer and the deployment viewer so the two can never silently drift.
"""
import ctypes
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from cartpole import CartPole

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

# Persisted into the policy checkpoint so deploy can reconstruct + assert the contract.
CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "max_substeps": MAX_SUBSTEPS,
          "rail": RAIL, "force_scale": FORCE_SCALE, "v_scale": V_SCALE, "w_scale": W_SCALE}


def make_obs(cart_x, cart_v, theta, theta_dot):
    """The 5-d observation, defined ONCE. Works for batched torch tensors (training) and for
    1-element tensors (single-robot deploy): [cart_x/rail, cart_v*s, sin th, cos th, w*s]."""
    return torch.stack([cart_x / RAIL, cart_v * V_SCALE,
                        torch.sin(theta), torch.cos(theta), theta_dot * W_SCALE], dim=-1)


def current_cuda_context():
    """torch's current CUDA primary context handle as an int (the context PhysX must adopt
    so PhysX + torch share one context). Must be called AFTER torch has touched CUDA."""
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError:
        return 0
    ctx = ctypes.c_void_p()
    if drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0:
        return 0
    return int(ctx.value or 0)


class CartPoleEnv:
    def __init__(self, num_envs=4096, episode_s=10.0, spacing=3.0, device="cuda", seed=0):
        self.K = num_envs
        self.dt = DT
        self.max_steps = int(episode_s * CONTROL_HZ)
        self.device = torch.device(device)
        self.g = torch.Generator(device=self.device).manual_seed(seed)

        # torch creates + makes-current its device primary context; PhysX adopts it so the
        # two share ONE CUDA context (else torch's cuBLAS dies on a separate PhysX context).
        torch.zeros(1, device=self.device)
        _w = torch.randn(64, 64, device=self.device); (_w @ _w).sum().item()
        torch.cuda.synchronize()
        ctx = current_cuda_context()

        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), direct_gpu=True, cuda_context=ctx)
        self.carts = [CartPole(self.world, x0=i * spacing) for i in range(self.K)]
        self.batch = tp.PhysxGpuBatch(self.world, [c.art for c in self.carts])
        self.dof = self.batch.max_dofs                  # cart + pole = 2

        K, dev = self.K, self.device
        # GPU-vs-add DOF order: the direct-GPU buffers are in PhysX cache order, the named CPU
        # accessors are in add-order. perm[i] = the GPU slot of canonical (add-order) joint i.
        # We keep ALL env tensors canonical and (de)permute only at the GPU boundary, so obs
        # indices (0=cart, 1=pole) are stable for ANY robot, not just this 2-link chain.
        self.perm = torch.from_numpy(self.carts[0].art.dof_order().astype(np.int64)).to(dev)
        self.gpu_idx = torch.from_numpy(self.batch.gpu_indices().astype(np.int32)).to(dev)

        self.jp_gpu = torch.zeros(K, self.dof, device=dev)   # read targets (GPU-slot order)
        self.jv_gpu = torch.zeros(K, self.dof, device=dev)
        self.force_gpu = torch.zeros(K, self.dof, device=dev)
        self.jp = torch.zeros(K, self.dof, device=dev)       # canonical (add order)
        self.jv = torch.zeros(K, self.dof, device=dev)
        self._actions = torch.zeros(K, 1, device=dev)
        self.steps = torch.zeros(K, dtype=torch.long, device=dev)

    # --- permutation between canonical (add) order and GPU-slot order ----------
    def _to_gpu(self, canon):
        out = torch.zeros_like(canon)
        out[:, self.perm] = canon
        return out

    def _read(self):
        self.batch.read_joint_pos(self.jp_gpu)
        self.batch.read_joint_vel(self.jv_gpu)
        self.jp = self.jp_gpu[:, self.perm].contiguous()     # -> canonical
        self.jv = self.jv_gpu[:, self.perm].contiguous()

    def _obs(self):
        return make_obs(self.jp[:, 0], self.jv[:, 0], self.jp[:, 1], self.jv[:, 1])

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        sub = self.gpu_idx[idx].contiguous()
        jp0 = torch.zeros(n, self.dof, device=self.device)   # canonical: cart=0
        jp0[:, 1] = (torch.rand(n, device=self.device, generator=self.g) * 2 - 1) * math.pi  # pole anywhere
        jv0 = torch.zeros(n, self.dof, device=self.device)
        torch.cuda.synchronize()
        self.batch.write_subset_joint_pos(self._to_gpu(jp0), sub)
        self.batch.write_subset_joint_vel(self._to_gpu(jv0), sub)
        self.steps[idx] = 0

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.device))
        torch.cuda.synchronize()
        self.batch.step(self.dt)
        self._read()
        return self._obs()

    @torch.no_grad()
    def step(self, actions):
        """Returns (next_obs, reward, done, terminal_obs). done is a TRUNCATION (timeout) — this
        env has no failure terminal — so terminal_obs is the real post-step obs for done envs
        (before reset overwrites it), which the trainer needs to bootstrap V(s_T) correctly."""
        self._actions = actions.clamp(-1.0, 1.0)
        force = torch.zeros(self.K, self.dof, device=self.device)
        force[:, 0] = self._actions[:, 0] * FORCE_SCALE      # cart is canonical joint 0
        self.force_gpu = self._to_gpu(force)
        torch.cuda.synchronize()
        self.batch.write_joint_force(self.force_gpu)
        self.batch.step(self.dt)
        self._read()
        self.steps += 1

        th = self.jp[:, 1]
        up = torch.cos(th)                                   # +1 up, -1 hanging down
        cart_x = self.jp[:, 0]
        a = self._actions[:, 0]
        rew = (up + 1.5 * (up > 0.9).float()
               - 0.01 * (cart_x / RAIL) ** 2
               - 0.0005 * self.jv[:, 1] ** 2
               - 0.003 * a ** 2)
        done = self.steps >= self.max_steps
        term_obs = self._obs()                               # obs at the (possibly terminal) state
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self._read()
            obs = self._obs()                                # done rows now post-reset
        else:
            obs = term_obs
        return obs, rew, done, term_obs


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = CartPoleEnv(num_envs=256)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()), "perm", env.perm.tolist())
    for _ in range(300):
        obs, rew, done, term = env.step(torch.rand(env.K, 1, device=env.device) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all() and torch.isfinite(term).all()
    print(f"300 steps ok; reward [{rew.min():.2f},{rew.max():.2f}]")
    print("CARTPOLE ENV SELFTEST: PASS")

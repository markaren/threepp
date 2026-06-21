"""GPU-resident vectorized cart-pole SWING-UP (cart + single pole).

K cart-poles in ONE PhysX direct-GPU scene. Each control step the policy outputs a single
cart force (applied via the direct-GPU joint-force API), physics steps once, and joint state
is read back into torch cuda tensors — obs / reward / reset all run in torch on the GPU.

The pole starts at a random angle (often hanging down), there is NO fall termination, and the
reward is the pole height (cos of its angle) plus a bonus near the top — so the only way to
score is to pump the cart and swing the pole up, then balance it. Unfakeable proof of control.
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

OBS_DIM = 5
ACT_DIM = 1
FORCE_SCALE = 45.0     # action [-1,1] -> cart force (N)
RAIL = 2.2


def current_cuda_context():
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError:
        return 0
    ctx = ctypes.c_void_p()
    if drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0:
        return 0
    return int(ctx.value or 0)


class CartPoleEnv:
    def __init__(self, num_envs=4096, control_hz=60, episode_s=10.0, spacing=3.0,
                 device="cuda", seed=0):
        self.K = num_envs
        self.dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.device = torch.device(device)
        self.g = torch.Generator(device=self.device).manual_seed(seed)

        # torch creates + makes-current the device primary context; PhysX adopts it so the
        # two share ONE CUDA context (else torch's cuBLAS dies on a separate PhysX context).
        torch.zeros(1, device=self.device)
        _w = torch.randn(64, 64, device=self.device); (_w @ _w).sum().item()
        torch.cuda.synchronize()
        ctx = current_cuda_context()

        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), direct_gpu=True, cuda_context=ctx)
        self.carts = [CartPole(self.world, x0=i * spacing) for i in range(self.K)]
        self.batch = tp.PhysxGpuBatch(self.world, [c.art for c in self.carts])

        K, dev = self.K, self.device
        self.gpu_idx = torch.from_numpy(self.batch.gpu_indices().astype(np.int32)).to(dev)
        self.jp = torch.zeros(K, 2, device=dev)     # cart_x, pole_angle
        self.jv = torch.zeros(K, 2, device=dev)
        self.force = torch.zeros(K, 2, device=dev)
        self._actions = torch.zeros(K, 1, device=dev)
        self.steps = torch.zeros(K, dtype=torch.long, device=dev)

    def _read(self):
        self.batch.read_joint_pos(self.jp.data_ptr())
        self.batch.read_joint_vel(self.jv.data_ptr())

    def _obs(self):
        th = self.jp[:, 1]
        return torch.stack([self.jp[:, 0] / RAIL, self.jv[:, 0] * 0.2,
                            torch.sin(th), torch.cos(th), self.jv[:, 1] * 0.1], dim=1)

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        sub = self.gpu_idx[idx].contiguous()
        jp0 = torch.zeros(n, 2, device=self.device)
        # random start anywhere (incl. hanging down) -> learn to swing up from any state
        jp0[:, 1] = (torch.rand(n, device=self.device, generator=self.g) * 2 - 1) * math.pi
        jv0 = torch.zeros(n, 2, device=self.device)
        torch.cuda.synchronize()
        self.batch.write_subset_joint_pos(jp0.data_ptr(), sub.data_ptr(), n)
        self.batch.write_subset_joint_vel(jv0.data_ptr(), sub.data_ptr(), n)
        self.steps[idx] = 0

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.device))
        torch.cuda.synchronize()
        self.batch.step(self.dt)
        self._read()
        return self._obs()

    @torch.no_grad()
    def step(self, actions):
        self._actions = actions.clamp(-1.0, 1.0)
        self.force[:, 0] = self._actions[:, 0] * FORCE_SCALE
        torch.cuda.synchronize()
        self.batch.write_joint_force(self.force.data_ptr())
        self.batch.step(self.dt)
        self._read()
        self.steps += 1

        th = self.jp[:, 1]
        up = torch.cos(th)                                   # +1 up, -1 hanging down
        cart_x = self.jp[:, 0]
        a = self._actions[:, 0]
        rew = (up                                            # height (dense)
               + 1.5 * (up > 0.9).float()                    # balance bonus near upright
               - 0.01 * (cart_x / RAIL) ** 2                 # don't camp at the rail
               - 0.0005 * self.jv[:, 1] ** 2                 # tiny: leave room to pump energy
               - 0.003 * a ** 2)
        done = self.steps >= self.max_steps                  # run the full episode (no fall)
        obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self._read()
            obs = self._obs()
        return obs, rew, done


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = CartPoleEnv(num_envs=256)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    for _ in range(300):
        obs, rew, done = env.step(torch.rand(env.K, 1, device=env.device) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"300 steps ok; reward [{rew.min():.2f},{rew.max():.2f}]")
    print("CARTPOLE ENV SELFTEST: PASS")

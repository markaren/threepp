"""Train the hexapod's residual-on-CPG policy on the GPU (owned stack, no stable_baselines3).

    python train_hexapod.py --iters 400

K hexapods walk in one PhysX direct-GPU scene (threepp.rl.GpuSim); the CPG gait + the policy's
residual corrections, the obs/reward/reset, and the PPO update all run in torch on the GPU. The
policy is rewarded for tracking a commanded (forward, turn) velocity while staying upright, so it
sharpens the open-loop gait. The training loop lives in threepp.rl.PPO. Saves hexapod_policy.pt.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod_gpu_env import ACT_DIM, CONFIG, HexapodGpuEnv
from threepp.rl import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(_HERE, "hexapod_policy.pt"))
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)

    env = HexapodGpuEnv(num_envs=args.envs, device="cuda")
    # small initial residuals (log_std_init=-1.5) so the policy doesn't wreck the gait at start
    ppo = PPO(env, ACT_DIM, hidden=(256, 256), lr=args.lr, horizon=args.horizon,
              log_std_init=-1.5, meta=CONFIG)
    ppo.learn(args.iters)
    ppo.save(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

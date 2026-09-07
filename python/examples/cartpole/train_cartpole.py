"""Train the cart-pole swing-up on the GPU.

    python train_cartpole.py --iters 300

K cart-poles swing up in one PhysX direct-GPU scene; obs/reward/reset and the PPO update all
run in torch on the GPU. Learns to swing the pole up from any start and balance it in ~1.5 min.
The training loop lives in threepp.rl.PPO; this script just builds the env and configures it.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from cartpole_env import ACT_DIM, CONFIG, CartPoleEnv
from threepp.rl import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(_HERE, "cartpole_swingup.pt"))
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)

    env = CartPoleEnv(num_envs=args.envs, device="cuda")
    ppo = PPO(env, ACT_DIM, hidden=(128, 128), lr=args.lr, horizon=args.horizon,
              log_std_init=-0.5, meta=CONFIG)
    ppo.learn(args.iters)
    ppo.save(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

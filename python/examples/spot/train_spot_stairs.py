"""Train a residual stair-climbing policy for Spot on GpuSim (threepp.rl.PPO).

    python train_spot_stairs.py --iters 600

K Spots run in one PhysX direct-GPU scene, each driven by the FROZEN Isaac walker (batched on
CUDA) plus a learned per-joint *residual*; rewarded for climbing the per-lane staircase (height
gain + forward progress + upright). The walker handles balance/gait; the residual learns the
foot-clearance / pitch corrections that get Spot over the risers. Obs/reward/PPO all on the GPU.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from threepp.rl import PPO
from spot_stairs_env import ACT_DIM, CONFIG, SpotStairEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_stairs_policy.pt"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("need CUDA"); sys.exit(0)

    env = SpotStairEnv(num_envs=args.envs, device="cuda", seed=args.seed)
    ppo = PPO(env, ACT_DIM, hidden=(256, 256), lr=args.lr, horizon=args.horizon,
              log_std_init=-1.5, meta=CONFIG)   # small initial residual so it doesn't wreck the walker

    def log(msg):
        print(f"{msg} | climb {env.last_climb:.3f} m | fell/step {env.last_fell:.3f}")
        ppo.save(args.out)   # checkpoint every log so a live viewer can hot-reload the latest policy

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

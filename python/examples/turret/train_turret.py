"""Train the vision point-defense turret from pixels (threepp.rl.PPO + CNN).

    python train_turret.py --iters 1500

K turrets each watch their own small GL camera; the policy maps the stacked frames to (yaw, pitch,
fire) and is rewarded for centering + shooting down incoming colliders. The whole loop — render,
obs, reward, PPO update — runs each step; the CNN trains in a coffee break (~3500 env-steps/s).
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from threepp.rl import PPO
from turret_env import ACT_DIM, CONFIG, TurretEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(_HERE, "turret_policy.pt"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("need CUDA"); sys.exit(0)

    env = TurretEnv(num_envs=args.envs, device="cuda")
    ppo = PPO(env, ACT_DIM, hidden=(256,), lr=args.lr, horizon=args.horizon,
              log_std_init=-0.5, minibatches=4, meta=CONFIG)

    def log(msg):
        print(f"{msg} | hit/step {float(env.last_hit):.3f} | defense-fail/step {float(env.last_fail):.3f}")

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

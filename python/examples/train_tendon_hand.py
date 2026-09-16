"""Train the tendon hand's palm-up hold on the GPU.

    python train_tendon_hand.py --envs 4096 --iters 3000 --seed 0

K hands and K objects in one PhysX direct-GPU scene; the 25 cables are re-resolved in torch
every physics substep (threepp.rl.cable), and obs/reward/reset and the PPO update all stay on
the GPU. The training loop lives in threepp.rl.PPO; this script builds the env and configures
it, exactly as train_cartpole.py does.

EAGER, not CUDA-graphed. The step's torch region calls into PhysX four times per control step
through the cable evaluation, and a graph captures torch kernels only -- the same reason a
Warp-backed env has to train eagerly.

--max-minutes stops cleanly at a wall-clock cap and still saves, which is what makes a local
smoke test a bounded thing and what lets an Idun job keep its scoring slot.
"""
import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import threepp as tp
from tendon_hand_env import ACT_DIM, CONFIG, TendonHandEnv
from threepp.rl import PPO


class _TimeUp(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly at this wall-clock cap and save what is trained")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(_HERE, "tendon_hand_hold.pt"))
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)

    torch.manual_seed(args.seed)
    env = TendonHandEnv(num_envs=args.envs, device="cuda", seed=args.seed)
    ppo = PPO(env, ACT_DIM, hidden=(256, 256), lr=args.lr, horizon=args.horizon,
              log_std_init=-0.5, meta={**CONFIG, "seed": args.seed})
    print(f"training: K={args.envs}  iters={args.iters}  horizon={args.horizon}  "
          f"seed={args.seed}  cap={args.max_minutes} min  -> {args.out}")

    t0 = time.perf_counter()
    done = [0]

    def on_iter(it):
        done[0] = it + 1
        if args.max_minutes and (time.perf_counter() - t0) / 60.0 >= args.max_minutes:
            raise _TimeUp

    try:
        ppo.learn(args.iters, log_every=args.log_every, on_iter=on_iter)
    except _TimeUp:
        print(f"time cap reached after {done[0]} iterations "
              f"({(time.perf_counter()-t0)/60:.2f} min)")
    ppo.save(args.out)
    mins = (time.perf_counter() - t0) / 60.0
    print(f"saved -> {args.out}   {done[0]} iterations in {mins:.2f} min "
          f"({done[0]*args.envs*args.horizon/max(mins*60,1e-9):,.0f} env-steps/s)")


if __name__ == "__main__":
    main()

"""Train Spot to walk FROM SCRATCH on GpuSim (threepp.rl.PPO) — no frozen walker.

    python train_spot_walk.py --iters 1500            # flat forward walking
    python train_spot_walk.py --iters 2000 --turn     # + lateral/turn commands

K Spots learn the 12-joint gait directly (legged-gym style velocity tracking). Checkpoints every
log so a live viewer can hot-reload. Watch `fwd` climb toward the commanded speed.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from threepp.rl import PPO
from spot_walk_env import ACT_DIM, CONFIG, SpotWalkEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=48)   # >= one gait cycle (0.96s @ 50Hz) so PPO links lift-off to landing
    ap.add_argument("--lr", type=float, default=2e-4)   # gentler -> stable convergence (3e-4 let the trot drift off)
    ap.add_argument("--turn", action="store_true", help="also command lateral + yaw (default: forward only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmstart", default="", help="policy .pt to initialize from (e.g. the forward trot)")
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_walk_policy.pt"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("need CUDA"); sys.exit(0)

    env = SpotWalkEnv(num_envs=args.envs, device="cuda", seed=args.seed, forward_only=not args.turn)
    # entropy 0.002 + tighter target_kl (0.012) -> stable convergence: the gait-phase reward has a sharp
    # trot optimum that bigger updates kept overshooting (peak-then-crash). Gentler exploration holds it.
    ppo = PPO(env, ACT_DIM, hidden=(256, 256), lr=args.lr, horizon=args.horizon,
              log_std_init=-0.5, entropy=0.002, target_kl=0.012, meta=CONFIG)
    if args.warmstart and os.path.exists(args.warmstart):
        ck = torch.load(args.warmstart, map_location="cuda", weights_only=True)   # same 50-d obs + net
        ppo.ac.load_state_dict(ck["model"])
        if ck.get("norm") is not None and ppo.norm is not None:
            ppo.norm.load(ck["norm"])
        print("warm-started from", os.path.basename(args.warmstart))

    def log(msg):
        print(f"{msg} | fwd {env.last_speed:+.2f} | up {env.last_up:.2f} | fell {env.last_fell:.3f} "
              f"| drag {env.last_drag:.2f} | clr {env.last_clear:.3f} | gait {env.last_gait:.2f} "
              f"| feet {env.last_nc:.1f}")  # gait -> W_GAIT(1.5) as the trot locks in; feet ~2 = trot
        ppo.save(args.out)                                     # checkpoint every log -> live viewer hot-reload

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

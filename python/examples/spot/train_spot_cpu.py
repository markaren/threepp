"""Train Spot to walk DIRECTLY on the CPU articulation (no GpuSim->CPU sim-to-sim gap).

    python train_spot_cpu.py --iters 4000           # flat forward walking, train == deploy

Physics steps on CPU (~5k env-steps/s, ~12x slower than GpuSim) but whatever walks here IS the
deploy gait. Checkpoints every log so play_spot_walk can hot-reload. The policy net runs on GPU.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from threepp.rl import PPO
from spot_cpu_env import SpotCpuEnv, CONFIG, ACT_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--turn", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmstart", default="", help="policy .pt to initialize from (e.g. the GpuSim v6 walker)")
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_cpu_policy.pt"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    env = SpotCpuEnv(num_envs=args.envs, device=dev, seed=args.seed, forward_only=not args.turn)
    ppo = PPO(env, ACT_DIM, hidden=(256, 256), lr=args.lr, horizon=args.horizon,
              log_std_init=-0.5, entropy=0.005, meta=CONFIG)
    if args.warmstart and os.path.exists(args.warmstart):
        ck = torch.load(args.warmstart, map_location=dev, weights_only=True)   # same 48->12 net + obs layout
        ppo.ac.load_state_dict(ck["model"])
        if ck.get("norm") is not None and ppo.norm is not None:
            ppo.norm.load(ck["norm"])
        print("warm-started from", os.path.basename(args.warmstart), "-> fine-tuning on the CPU solver")

    def log(msg):
        print(f"{msg} | fwd {env.last_speed:+.2f} | up {env.last_up:.2f} | fell {env.last_fell:.3f} "
              f"| base_z {env.last_height:.3f}")
        ppo.save(args.out)

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

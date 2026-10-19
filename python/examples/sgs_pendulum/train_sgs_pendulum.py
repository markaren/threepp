"""Train the pendulum pool with one reset sampler and log the fixed evaluation as it goes.

    python train_sgs_pendulum.py --sampler sgs --envs 1024 --iters 300 --seed 0 --log run.jsonl

The sampler is the only thing that differs between arms: "uniform" draws cells uniformly,
"sgs" draws them with the success-guided kernel, "sfl" with p(1-p). PPO, the task, the reward
and the evaluation are identical. The evaluation runs every --eval-every iterations on every
cell with the deterministic policy and is independent of the training draws.
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from pendulum_env import ACT_DIM, N_CELLS, TAUS, PendulumPoolEnv, evaluate, summarize  # noqa: E402
from threepp.rl import PPO                                                            # noqa: E402
from threepp.rl.sgs import SuccessGuidedSampler                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=SuccessGuidedSampler.MODES, default="sgs")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--dense", type=float, default=0.0, help="reward shaping weight (0 = sparse)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatches", type=int, default=1)
    ap.add_argument("--graph", action="store_true", help="CUDA-graph the per-step torch region")
    ap.add_argument("--log", default=None, help="append one JSON line per evaluation here")
    ap.add_argument("--out", default=None, help="save the final policy here")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed + 1000)
    sampler = SuccessGuidedSampler(N_CELLS, dev, mode=args.sampler, target=args.target,
                                   kappa=args.kappa, temperature=args.temperature, generator=g)
    env = PendulumPoolEnv(args.envs, device=dev, seed=args.seed, sampler=sampler, dense=args.dense,
                          graph=args.graph)
    ppo = PPO(env, ACT_DIM, hidden=(128, 128), lr=args.lr, horizon=32, entropy=0.0,
              log_std_init=-1.0, minibatches=args.minibatches)
    t0 = time.time()
    log = open(args.log, "a") if args.log else None

    def on_iter(it):
        if it % args.eval_every and it != args.iters:
            return
        per_cell = evaluate(ppo.ac, ppo.norm, dev)
        overall, by_tau = summarize(per_cell)
        p = sampler.success_rate()
        rec = {"it": it, "sampler": args.sampler, "envs": args.envs, "seed": args.seed,
               "eval": overall, "eval_by_tau": by_tau, "train_success": env.success_since_last(),
               "cells_seen": int((sampler.count > 0).sum()),
               "frontier": int(((p > 0.1) & (p < 0.9) & (sampler.count > 0)).sum()),
               "t": time.time() - t0}
        print(f"it {it:4d}  eval {overall:.3f}  by tau {' '.join(f'{v:.2f}' for v in by_tau)}  "
              f"frontier {rec['frontier']:3d}/{N_CELLS}  {rec['t']:.0f}s")
        if log:
            log.write(json.dumps(rec) + "\n"); log.flush()

    ppo.learn(args.iters, log_every=50, on_iter=on_iter)
    if args.out:
        ppo.save(args.out)
    if log:
        log.close()


if __name__ == "__main__":
    main()

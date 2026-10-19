"""train_kuka_grasp.py — train the KUKA to reach, grasp by friction and lift a cube (threepp.rl PPO).

    python train_kuka_grasp.py --sampler scaffold --envs 512 --iters 50     # the June env, short run

--sampler scaffold is the June env: a shaped reward and a reset-height curriculum advanced by
set_iter(it) from the iteration hook. Every log line goes to stdout and, as JSON, to
<run>/train.jsonl; a policy snapshot is written to <run>/ckpt/ every --save-every iterations.
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_env import KukaGraspEnv  # noqa: E402
from threepp.rl import PPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=("scaffold",), default="scaffold")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--log-std-init", type=float, default=-0.5)
    ap.add_argument("--no-graph", action="store_true", help="run the per-step torch region eagerly")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--run", default=None, help="run directory (default runs/<sampler>_s<seed>)")
    args = ap.parse_args()

    run = args.run or os.path.join(_HERE, "runs", f"{args.sampler}_s{args.seed}")
    os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed)
    t_build = time.perf_counter()
    env = KukaGraspEnv(num_envs=args.envs, seed=args.seed, mode=args.sampler, graph=not args.no_graph)
    build_s = time.perf_counter() - t_build
    ppo = PPO(env, C.ACT_DIM, hidden=(512, 256, 128), lr=args.lr, horizon=args.horizon,
              minibatches=args.minibatches, normalize_obs=True, normalize_returns=True,
              target_kl=0.02, log_std_init=args.log_std_init)
    ppo.meta.update({"sampler": args.sampler, "seed": args.seed, "envs": args.envs})
    log = open(os.path.join(run, "train.jsonl"), "a")
    log.write(json.dumps({"event": "start", "args": vars(args), "build_s": build_s}) + "\n")
    print(f"[train] {args.sampler}: {args.envs} envs x {args.iters} iters  built in {build_s:.1f} s  "
          f"obs={C.OBS_DIM} act={C.ACT_DIM}  -> {run}")

    def on_log(msg):
        print(msg, flush=True)
        log.write(json.dumps({"event": "log", **ppo.last_log, **ppo.diagnostics(),
                              **{f"train_{k}": v for k, v in env.last_metrics.items()}}) + "\n")
        log.flush()

    def on_iter(it):
        ppo.meta["iter"] = it
        if args.sampler == "scaffold":
            env.set_iter(it)
        if it % args.save_every == 0 or it == args.iters:
            ppo.save(os.path.join(run, "ckpt", f"it{it:05d}.pt"))

    ppo.learn(args.iters, log_every=10, on_log=on_log, on_iter=on_iter)
    ppo.save(os.path.join(run, "final.pt"))
    log.write(json.dumps({"event": "end", "iters": args.iters, **ppo.last_log}) + "\n")
    log.close()
    print(f"[train] saved {os.path.join(run, 'final.pt')}")


if __name__ == "__main__":
    main()

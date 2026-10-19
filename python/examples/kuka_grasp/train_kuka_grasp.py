"""train_kuka_grasp.py — train the KUKA to reach, grasp by friction and lift a cube (threepp.rl PPO).

    python train_kuka_grasp.py --sampler sgs --envs 512 --iters 1500 --seed 0
    python train_kuka_grasp.py --sampler scaffold --envs 512 --iters 50     # the June env, short run

The sampler is what differs between arms:
    scaffold  the June env: shaped reward, reset-height curriculum advanced by set_iter(it)
    uniform   the reset pool (kuka_grasp_pool.npz), cells drawn uniformly, the shared reward
    sgs       the reset pool, cells drawn by threepp.rl.sgs.SuccessGuidedSampler (target 0.5,
              kappa 1, temperature 2: the paper's manipulation values and sgs.py's defaults)
PPO, the network and the evaluation are the same for all three. Every log line goes to stdout
and, as JSON, to <run>/train.jsonl; a policy snapshot goes to <run>/ckpt/ every --save-every
iterations for eval_kuka_grasp.py --watch, which runs as its own process (one PhysX foundation
per process).
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
from kuka_grasp_pool import FAMILIES, load_pool  # noqa: E402
from threepp.rl import PPO  # noqa: E402
from threepp.rl.sgs import SuccessGuidedSampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=("scaffold", "uniform", "sgs"), default="sgs")
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
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
    sampler = pool = None
    if args.sampler == "scaffold":
        env = KukaGraspEnv(num_envs=args.envs, seed=args.seed, mode="scaffold", graph=not args.no_graph)
    else:
        pool = load_pool()
        g = torch.Generator(device="cuda").manual_seed(args.seed + 1000)
        sampler = SuccessGuidedSampler(pool["n"], "cuda", mode=args.sampler, target=args.target,
                                       kappa=args.kappa, temperature=args.temperature, generator=g)
        env = KukaGraspEnv(num_envs=args.envs, seed=args.seed, mode="pool", pool=pool,
                           sampler=sampler, graph=not args.no_graph)
    build_s = time.perf_counter() - t_build
    ppo = PPO(env, C.ACT_DIM, hidden=(512, 256, 128), lr=args.lr, horizon=args.horizon,
              minibatches=args.minibatches, normalize_obs=True, normalize_returns=True,
              target_kl=0.02, log_std_init=args.log_std_init)
    ppo.meta.update({"sampler": args.sampler, "seed": args.seed, "envs": args.envs})
    log = open(os.path.join(run, "train.jsonl"), "a")
    log.write(json.dumps({"event": "start", "args": vars(args), "build_s": build_s}) + "\n")
    print(f"[train] {args.sampler}: {args.envs} envs x {args.iters} iters  built in {build_s:.1f} s  "
          f"obs={C.OBS_DIM} act={C.ACT_DIM}  -> {run}")

    def sampler_stats():
        if sampler is None:
            return {}
        p, seen = sampler.success_rate(), sampler.count > 0
        probs = sampler.probs()
        out = {"cells_seen": int(seen.sum()),
               "frontier": int(((p > 0.1) & (p < 0.9) & seen).sum())}
        for f, name in enumerate(FAMILIES):
            fam = pool["family"] == f
            m = fam & seen
            out[f"p_{name}"] = float(p[m].mean()) if bool(m.any()) else float("nan")
            out[f"draw_{name}"] = float(probs[fam].sum())
        return out

    def on_log(msg):
        st = sampler_stats()
        if st:
            msg += (f"\n        cells seen {st['cells_seen']}  frontier {st['frontier']}  p R/SG/NG "
                    f"{st['p_reaching']:.2f}/{st['p_stable_grasp']:.2f}/{st['p_near_goal']:.2f}  "
                    f"draw R/SG/NG {st['draw_reaching']:.2f}/{st['draw_stable_grasp']:.2f}/"
                    f"{st['draw_near_goal']:.2f}")
        print(msg, flush=True)
        log.write(json.dumps({"event": "log", **ppo.last_log, **ppo.diagnostics(),
                              **{f"train_{k}": v for k, v in env.last_metrics.items()}, **st}) + "\n")
        log.flush()

    def on_iter(it):
        ppo.meta["iter"] = it
        if args.sampler == "scaffold":
            env.set_iter(it)
        if it % args.save_every == 0 or it == args.iters:
            ppo.save(os.path.join(run, "ckpt", f"it{it:05d}.pt"))

    ppo.learn(args.iters, log_every=10, on_log=on_log, on_iter=on_iter)
    ppo.save(os.path.join(run, "final.pt"))
    if sampler is not None:
        torch.save(sampler.state(), os.path.join(run, "sampler.pt"))
    log.write(json.dumps({"event": "end", "iters": args.iters, **ppo.last_log}) + "\n")
    log.close()
    print(f"[train] saved {os.path.join(run, 'final.pt')}")


if __name__ == "__main__":
    main()

"""Train Spot's fall-recovery policy (spot_recovery_env.py), or evaluate one from fallen starts.

    python train_spot_recovery.py --iters 1000 --warmstart spot_steps.pt --out runs/recA/spot_recovery.pt     # arm A
    python train_spot_recovery.py --iters 1000 --warmstart "" --out runs/recB/spot_recovery.pt                # arm B, fresh
    python train_spot_recovery.py --eval runs/recA/spot_recovery.pt --eval-envs 2048 --eval-json eval.json

Arm A continues the shipped 96-d walking policy whole (actor, critic, log_std, obs norm) through the same continue path
as train_spot_steps.py; arm B starts from a fresh network. Both run the walking plant (stiff gains, real torque limits)
with self-collision, and write every choice into the checkpoint meta. --eval scores one episode per env from fallen
starts at the top level (40% back, 40% side, 20% any orientation), with no standing starts and the curriculum frozen.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from threepp.rl import PPO, load_policy
from spot_recovery_env import (SpotRecoveryEnv, CONFIG, OBS_DIM, ACT_DIM, N_LEVELS, KIND_NAMES, EPISODE_S, CONTROL_HZ)
from spot_terrain_env import HIDDEN
from spot_steps_symmetry import make_aux_loss


@torch.no_grad()
def evaluate(ckpt, k=2048, seed=0, self_collision=None, json_path=""):
    """One episode per env from fallen starts at level N_LEVELS-1, deterministic act_mean -> dict (and JSON)."""
    ac, norm, meta = load_policy(ckpt, device="cuda")
    sc = bool(meta.get("self_collision", True)) if self_collision is None else bool(self_collision)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    env = SpotRecoveryEnv(num_envs=k, seed=seed, self_collision=sc, stand_frac=0.0, init_level=N_LEVELS - 1,
                          freeze_level=True, count_episodes=1)
    build_s = time.perf_counter() - t0
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    obs = env.reset()
    env.recovery_stats(reset=True)
    steps = int(EPISODE_S * CONTROL_HZ) + 5
    t1 = time.perf_counter()
    for _ in range(steps):
        obs, _, _, _, _ = env.step(pol(obs))
    torch.cuda.synchronize()
    st = env.recovery_stats(reset=False)
    res = {"checkpoint": os.path.abspath(ckpt), "envs": k, "seed": seed, "self_collision": sc, "level": N_LEVELS - 1,
           "build_s": build_s, "steps": steps, "steps_s": time.perf_counter() - t1,
           "meta": {kk: meta.get(kk) for kk in ("warmstart", "arm", "iters", "seed", "self_collision", "lr")},
           **st}
    tot = st["total"]["fallen"]
    print(f"[eval] {os.path.basename(ckpt)}  K={k}  fallen episodes {tot['episodes']}: success by 3 s {tot['success_3s']:.3f}"
          f"  by 6 s {tot['success_6s']:.3f}  recovery {tot['recovery_s'] or float('nan'):.2f} s  blow-ups {tot['blowups']}"
          f"  self-contact ticks/ep {tot['contact_ticks_per_ep']:.1f}  p95 tau hx/hy/kn {st['tau_p95_hx_hy_kn']}")
    for name, r in st["by_level_kind"].items():
        print(f"        {name:10s} eps {r['episodes']:5d}  3 s {r['success_3s']:.3f}  6 s {r['success_6s']:.3f}  "
              f"recovery {r['recovery_s'] if r['recovery_s'] is None else round(r['recovery_s'], 2)} s")
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"[eval] -> {json_path}")
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 warm-started, 3e-4 fresh")
    ap.add_argument("--warmstart", default=os.path.join(_HERE, "spot_steps.pt"),
                    help="a 96-d checkpoint to continue whole (arm A: the shipped walking policy); '' = fresh network (arm B)")
    ap.add_argument("--log-std-init", dest="log_std_init", type=float, default=-1.0,
                    help="fresh network's initial action log-std")
    ap.add_argument("--log-std-floor", dest="log_std_floor", type=float, default=-1.0,
                    help="a warm start's log-std is raised to at least this (the walking policy explores at ~-1.5, too "
                         "narrow to find a roll); None-like values below -10 leave it alone")
    ap.add_argument("--self-collision", dest="self_collision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stand-frac", dest="stand_frac", type=float, default=CONFIG["stand_frac"])
    ap.add_argument("--init-level", dest="init_level", type=int, default=0)
    ap.add_argument("--sym-coef", dest="sym_coef", type=float, default=1.0)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--target-kl", dest="target_kl", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="", help="label recorded in meta (A / B)")
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_recovery.pt"))
    ap.add_argument("--metrics", default="")
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int, default=0)
    ap.add_argument("--eval", default="", help="evaluate this checkpoint from fallen starts instead of training")
    ap.add_argument("--eval-envs", dest="eval_envs", type=int, default=2048)
    ap.add_argument("--eval-json", dest="eval_json", default="")
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.eval:
        evaluate(args.eval, k=args.eval_envs, seed=args.seed, json_path=args.eval_json)
        return

    torch.manual_seed(args.seed)
    warm = bool(args.warmstart)
    lr = args.lr if args.lr is not None else (1e-4 if warm else 3e-4)
    t0 = time.perf_counter()
    env = SpotRecoveryEnv(num_envs=args.envs, seed=args.seed, self_collision=args.self_collision,
                          stand_frac=args.stand_frac, init_level=args.init_level)
    print(f"recovery env: K={args.envs}, self-collision {'on' if args.self_collision else 'off'}, built in "
          f"{time.perf_counter() - t0:.1f} s")
    aux = make_aux_loss(args.sym_coef) if args.sym_coef > 0 else None
    meta = {**CONFIG, "self_collision": args.self_collision, "warmstart": os.path.basename(args.warmstart) if warm else "",
            "warmstart_path": os.path.abspath(args.warmstart) if warm else "", "arm": args.arm, "iters": args.iters,
            "envs": args.envs, "seed": args.seed, "lr": lr, "horizon": args.horizon, "log_std_init": args.log_std_init,
            "log_std_floor": args.log_std_floor if warm else None, "stand_frac": args.stand_frac,
            "init_level": args.init_level, "sym_coef": args.sym_coef, "entropy": args.entropy, "target_kl": args.target_kl,
            "epochs": args.epochs, "minibatches": args.minibatches, "clip": args.clip, "select": "final",
            "drive_limits_are_forces": True, "height_source": "flat", "stand_mode": True}
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=lr, horizon=args.horizon, log_std_init=args.log_std_init,
              entropy=args.entropy, clip=args.clip, epochs=args.epochs, minibatches=args.minibatches,
              target_kl=args.target_kl if args.target_kl > 0 else None, normalize_obs=True, meta=meta, aux_loss=aux)
    if warm:
        src, src_norm, src_meta = load_policy(args.warmstart, device="cuda")
        if src_meta.get("obs_dim") != OBS_DIM:
            ap.error(f"--warmstart must be a {OBS_DIM}-d checkpoint, {args.warmstart} is {src_meta.get('obs_dim')}-d")
        ppo.ac.load_state_dict(src.state_dict())
        if ppo.norm is not None and src_norm is not None:
            ppo.norm.load(src_norm.state())
        before = float(ppo.ac.log_std.mean())
        if args.log_std_floor > -10:
            ppo.ac.log_std.data.clamp_(min=args.log_std_floor)
        ppo.meta["warmstart_log_std"] = [before, float(ppo.ac.log_std.mean())]
        print(f"continued from {os.path.basename(args.warmstart)} (96-d: actor+critic+log_std+norm), log_std mean "
              f"{before:.2f} -> {float(ppo.ac.log_std.mean()):.2f}")
    else:
        print(f"fresh network, log_std {args.log_std_init}")

    stem = os.path.splitext(args.out)[0]
    latest = stem + "_latest.pt"
    metrics_path = args.metrics or os.path.join(os.path.dirname(os.path.abspath(args.out)), "metrics.jsonl")
    for d in {os.path.dirname(os.path.abspath(args.out)), os.path.dirname(os.path.abspath(metrics_path))}:
        os.makedirs(d, exist_ok=True)

    def metric(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    metric({"type": "start", "out": os.path.abspath(args.out), "argv": sys.argv[1:], "meta": ppo.meta})
    env.recovery_stats(reset=True)

    def log(msg):
        ppo.save(latest)
        st = env.recovery_stats(reset=True)
        fa, sd = st["total"].get("fallen", {}), st["total"].get("stand", {})
        f3 = fa.get("success_3s"); f6 = fa.get("success_6s")
        print(f"{msg}\n        fallen eps {fa.get('episodes', 0)}: success 3 s {f3 if f3 is None else round(f3, 3)}  6 s "
              f"{f6 if f6 is None else round(f6, 3)}  recovery {fa.get('recovery_s') and round(fa['recovery_s'], 2)} s | stand "
              f"eps {sd.get('episodes', 0)} success {sd.get('success_6s') and round(sd['success_6s'], 3)} | levels "
              f"{st['levels']} | p95 tau {st['tau_p95_hx_hy_kn']} | contact ticks/ep {fa.get('contact_ticks_per_ep') and round(fa['contact_ticks_per_ep'], 1)}"
              f" | blow-ups {fa.get('blowups', 0) + sd.get('blowups', 0)} | spawn {st['spawn']}")
        metric({"type": "log", **ppo.last_log, **ppo.diagnostics(), **st})

    def on_iter(it):
        if args.snapshot_every > 0 and it % args.snapshot_every == 0:
            ppo.save(f"{stem}_it{it:05d}.pt")

    ppo.learn(args.iters, log_every=20, on_log=log, on_iter=on_iter)
    ppo.save(latest)
    ppo.save(args.out)
    print(f"saved -> {args.out} = {latest} (final weights) | metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

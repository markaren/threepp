"""spotv2 — STAIRS/terrain trainer: warm-start from scratch_flat_best.pt (50-d clock base gait,
normalize_obs=True, stiff gains) into the 96-d clock+terrain obs, then PPO-fine-tune.

    python train_spot_stairs.py --iters 1500
    python train_spot_stairs.py --eval spot_terrain.pt     # held-out flat-steering regression vs base gait

The shared helpers (warmstart_scratch_to_terrain, sanity_walk, stochastic_flat_baseline,
eval_flat_steering) live in _common.py, next to this file.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))   # for load_policy on scratch_flat_best.pt

import threepp as tp
from spot_terrain_env import ACT_DIM, CONFIG, HIDDEN, SpotTerrainEnv
from threepp.rl import PPO
from _common import (warmstart_scratch_to_terrain, sanity_walk, stochastic_flat_baseline,
                     eval_flat_steering)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)        # gentle: adapt the gait, don't forget it
    ap.add_argument("--rise_max", type=float, default=0.20)  # top step height in the graded lanes (curriculum lever)
    ap.add_argument("--gate", type=float, default=0.90)      # keep flat steering >= gate * the teacher's flat tracking
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_terrain.pt"))
    ap.add_argument("--eval", default="")                    # path to a .pt -> run the flat-steering regression and exit
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.eval:
        eval_flat_steering(SpotTerrainEnv, args.eval); return

    env = SpotTerrainEnv(num_envs=args.envs, device="cuda", rise_max=args.rise_max)
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=args.lr, horizon=args.horizon,
              log_std_init=-1.5, entropy=0.0, normalize_obs=True, meta=CONFIG)
    warmstart_scratch_to_terrain(ppo.ac, ppo.norm,
                                 os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt"),
                                 device="cuda")
    sanity_walk(env, ppo.ac, ppo.norm)                        # transplant quality (deterministic): tracks + climbs?
    flat0 = stochastic_flat_baseline(env, ppo.ac, ppo.norm)  # gate baseline, calibrated to the stochastic rollouts
    gate = args.gate * flat0
    print(f"flat-steering gate = {gate:.3f}  (= {args.gate:.2f} x warm-start STOCHASTIC flat tracking {flat0:.3f})")

    latest = os.path.splitext(args.out)[0] + "_latest.pt"     # always-current policy (resume / final)
    best = [-1e9]

    def log(msg):
        trk, ftrk, c = env.last_track, env.last_flat_track, env.last_climb
        ppo.save(latest)
        ok = ftrk >= gate                                     # steering still good?
        mark = ""
        if trk > best[0] and ok:                              # best overall tracking SUBJECT TO steering preserved
            best[0] = trk
            ppo.save(args.out)
            mark = "  <- saved best"
        print(f"{msg} | track {trk:.3f} | flat {ftrk:.3f}{'' if ok else ' LOW!'} | "
              f"climb {c:.2f} | fell {env.last_fell:.3f}{mark}")

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(latest)
    print(f"saved -> {args.out} (best track {best[0]:.3f}, steering gate {gate:.3f}) + {latest} (final)")
    print(f"next: python {os.path.basename(__file__)} --eval {args.out}   # confirm steering preserved")


if __name__ == "__main__":
    main()

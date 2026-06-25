"""spotv2 — train the velocity-tracking policy to climb DISCRETE STAIRS with an adaptive curriculum.

    python train_spot_steps.py --iters 1500                       # transfers spot_steps_1d.pt (2-D scan) by default
    python train_spot_steps.py --warmstart "" --iters 1500        # from the Isaac flat walker instead
    python train_spot_steps.py --warmstart spot_hf.pt --iters 1500  # (only if spot_hf.pt is a 2-D-scan ckpt)
    python train_spot_steps.py --score spot_steps.pt              # deterministic track/flat/fell + curriculum level
    python train_spot_steps.py --eval  spot_steps.pt              # flat-steering regression vs the teacher

The curriculum (per-env level, promote on clearing the tent / demote on a fall) lives in SpotStepsEnv;
the trainer just logs `level` (mean riser-band the stair envs are on) and `clear` (fraction clearing
their tent). The default warm-start TRANSFERS the preserved 1-D stair climber (spot_steps_1d.pt) into
the new 2-D scan obs via warmstart_expand_scan (centerline row = old scan, lateral cols zero-init), so
the policy begins exactly as the 1-D climber and learns to exploit the lateral scan from there.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))

import threepp as tp
from spot_deploy import fetch_assets
from spot_steps_env import ACT_DIM, CONFIG, HIDDEN, N_LEVELS, RISERS, SpotStepsEnv
from spot_symmetry import make_aux_loss
from train_spot_terrain import (warmstart_from_isaac, warmstart_expand_scan, sanity_walk,
                                stochastic_flat_baseline)
from threepp.rl import PPO, load_policy


@torch.no_grad()
def score_checkpoint(policy_path, k=512, device="cuda", steps=900, warm=200):
    """Deterministic track/flat/fell + the curriculum LEVEL it climbs to (how tall a riser it handles).
    The env starts every stair env at level 0 and promotes as the policy clears tents."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return None
    env = SpotStepsEnv(num_envs=k, device=device)
    ac, _, _ = load_policy(policy_path, device=device)
    obs = env.reset()
    trk, flt, fl = [], [], []
    for t in range(steps):
        obs, _, _, _, _ = env.step(ac.act_mean(obs))
        if t >= warm:
            trk.append(env.last_track); flt.append(env.last_flat_track); fl.append(env.last_fell)
    m = lambda a: sum(a) / max(1, len(a))
    lvl = env.last_level
    riser = RISERS[min(int(round(lvl)), N_LEVELS - 1)]
    print(f"[score] {os.path.basename(policy_path)}  (deterministic, K={k}, {steps - warm} steps)")
    print(f"        track {m(trk):.3f}/2.0   flat {m(flt):.3f}/2.0   fell/step {m(fl):.4f}   "
          f"curriculum level {lvl:.2f}/{N_LEVELS - 1}  (~{riser:.02f} m risers)")
    return m(trk), m(flt), m(fl), lvl


@torch.no_grad()
def eval_flat_steering(policy_path, k=512, device="cuda"):
    """Held-out steering regression on FLAT ground vs the Isaac teacher (the real steering test)."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    env = SpotStepsEnv(num_envs=k, device=device, flat_only=True)
    ac, _, _ = load_policy(policy_path, device=device)
    teacher = env.imit_policy
    pol = lambda o: ac.act_mean(o)
    tea = lambda o: teacher(o[:, :48])
    grid = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.5)]
    print(f"flat-steering regression ({os.path.basename(policy_path)} vs Isaac teacher, K={k}):")
    print("   cmd[vx,vy,wz]      policy_err   teacher_err   ratio")
    worst = 0.0
    for cmd in grid:
        ep = env.measure_tracking(pol, cmd); et = env.measure_tracking(tea, cmd)
        ratio = ep / max(et, 1e-6); worst = max(worst, ratio)
        flag = "" if ratio <= 1.10 else "  <- REGRESSED"
        print(f"   [{cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}]     {ep:8.3f}    {et:8.3f}    {ratio:5.2f}{flag}")
    print(f"worst ratio {worst:.2f}  ->  {'PASS (steering preserved)' if worst <= 1.10 else 'FAIL (steering degraded)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)        # ~52k stair boxes -> builds fine; 2x throughput vs 1024
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_steps.pt"))
    ap.add_argument("--warmstart", default=os.path.join(_HERE, "spot_steps_1d.pt"),
                    help="1-D-scan .pt to transfer into the 2-D scan via warmstart_expand_scan "
                         "(default spot_steps_1d.pt — the preserved 1-D stair climber); '' = from Isaac")
    ap.add_argument("--fell_max", type=float, default=0.006)
    ap.add_argument("--sym_coef", type=float, default=1.0)   # left-right symmetry augmentation (kills the veer)
    ap.add_argument("--eval", default="")
    ap.add_argument("--score", default="")
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.score:
        score_checkpoint(args.score); return
    if args.eval:
        eval_flat_steering(args.eval); return

    env = SpotStepsEnv(num_envs=args.envs, device="cuda")
    aux = make_aux_loss(args.sym_coef) if args.sym_coef > 0 else None
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=args.lr, horizon=args.horizon,
              log_std_init=-1.5, entropy=0.0, normalize_obs=False, meta=CONFIG, aux_loss=aux)
    if aux is not None:
        print(f"symmetry augmentation ON (coef {args.sym_coef})")
    if args.warmstart and os.path.exists(args.warmstart):
        # 1-D-scan checkpoint -> new 2-D-scan AC (input layer expanded; centerline = old scan, lateral = 0)
        warmstart_expand_scan(ppo.ac, args.warmstart, device="cuda")
    else:
        if args.warmstart:
            print(f"(warmstart {args.warmstart} not found — falling back to Isaac)")
        warmstart_from_isaac(ppo.ac, os.path.join(fetch_assets(), "spot_policy.pt"), device="cuda")
    sanity_walk(env, ppo.ac)
    flat0 = stochastic_flat_baseline(env, ppo.ac)
    gate = args.gate * flat0
    print(f"flat-steering gate = {gate:.3f}  (= {args.gate:.2f} x warm-start STOCHASTIC flat tracking {flat0:.3f})")

    latest = os.path.splitext(args.out)[0] + "_latest.pt"
    best = [-1e9]

    def log(msg):
        trk, ftrk, lvl = env.last_track, env.last_flat_track, env.last_level
        ppo.save(latest)
        ok = ftrk >= gate and env.last_fell <= args.fell_max
        # STAIRS: the objective is climb HEIGHT (curriculum level), not track — track DROPS with
        # difficulty, so best-by-track would pick the easy early checkpoint. Select by level (track ties).
        score = lvl + 0.01 * trk
        mark = ""
        if score > best[0] and ok:
            best[0] = score
            ppo.save(args.out)
            mark = "  <- saved best"
        print(f"{msg} | track {trk:.3f} | flat {ftrk:.3f}{'' if ok else ' LOW!'} | "
              f"level {lvl:.2f}/{N_LEVELS - 1} | clear {env.last_clear:.2f} | "
              f"fell {env.last_fell:.3f}{mark}")

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(latest)
    print(f"saved -> {args.out} (best level-score {best[0]:.3f}, steering gate {gate:.3f}) + {latest} (final)")
    print(f"next: python {os.path.basename(__file__)} --score {args.out}")


if __name__ == "__main__":
    main()

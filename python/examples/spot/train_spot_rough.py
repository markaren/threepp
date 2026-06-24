"""spotv2 — train the velocity-tracking policy on UNEVEN (rough) terrain (see spot_rough_env.py).

    python train_spot_rough.py --iters 1500
    python train_spot_rough.py --eval spot_rough.pt     # held-out flat-steering regression vs the teacher

Identical recipe to train_spot_terrain.py (warm-start the Isaac walker, gentle lr, raw obs, Pareto
steering gate) — only the env is swapped to gentle rolling bumps. The warm-start / baseline / sanity
helpers are reused directly from train_spot_terrain.
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
from spot_rough_env import ACT_DIM, AMP_MAX, CONFIG, HIDDEN, SpotRoughEnv
from train_spot_terrain import warmstart_from_isaac, sanity_walk, stochastic_flat_baseline
from threepp.rl import PPO, load_policy


@torch.no_grad()
def score_checkpoint(policy_path, k=512, amp_max=AMP_MAX, device="cuda", steps=500, warm=150):
    """DETERMINISTIC quality of a checkpoint on the rough env: mean command-tracking, flat-lane tracking,
    and fall rate over `steps-warm` ticks (act_mean, no exploration noise). The training-time 'best' is a
    noisy one-step stochastic snapshot and ignores falls — THIS is the signal to pick a deploy/ramp source.
    Compare e.g. spot_rough.pt vs spot_rough_latest.pt and warm-start the ramp from whichever scores best."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return None
    env = SpotRoughEnv(num_envs=k, device=device, amp_max=amp_max)
    ac, _, _ = load_policy(policy_path, device=device)
    obs = env.reset()
    trk, flt, fl = [], [], []
    for t in range(steps):
        obs, _, _, _, _ = env.step(ac.act_mean(obs))
        if t >= warm:
            trk.append(env.last_track); flt.append(env.last_flat_track); fl.append(env.last_fell)
    m = lambda a: sum(a) / max(1, len(a))
    track, flat, fell = m(trk), m(flt), m(fl)
    print(f"[score] {os.path.basename(policy_path)}  (deterministic, K={k}, amp_max={amp_max}, {steps - warm} steps)")
    print(f"        track {track:.3f}/2.0   flat {flat:.3f}/2.0   fell/step {fell:.4f}")
    return track, flat, fell


@torch.no_grad()
def eval_flat_steering(policy_path, k=512, device="cuda"):
    """Held-out STEERING REGRESSION on FLAT ground: policy tracking error vs the Isaac teacher across a
    command grid. PASS = within ~1.10x the teacher on every command (steering not degraded)."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    env = SpotRoughEnv(num_envs=k, device=device, flat_only=True)
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
        ep = env.measure_tracking(pol, cmd)
        et = env.measure_tracking(tea, cmd)
        ratio = ep / max(et, 1e-6)
        worst = max(worst, ratio)
        flag = "" if ratio <= 1.10 else "  <- REGRESSED"
        print(f"   [{cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}]     {ep:8.3f}    {et:8.3f}    {ratio:5.2f}{flag}")
    print(f"worst ratio {worst:.2f}  ->  {'PASS (steering preserved)' if worst <= 1.10 else 'FAIL (steering degraded)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)        # ~37 boxes/lane -> ~76k static boxes, builds in ~15s
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp_max", type=float, default=0.08)   # top bump amplitude across the graded lanes
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_rough.pt"))
    ap.add_argument("--warmstart", default="",               # continue from a rough .pt instead of from Isaac
                    help="continue from a previous rough .pt (full actor+critic+log_std) instead of warm-starting "
                         "from the Isaac walker — use with a higher --amp_max to ramp terrain difficulty in stages")
    ap.add_argument("--fell_max", type=float, default=0.005)  # auto-best must also keep falls below this
    ap.add_argument("--eval", default="")
    ap.add_argument("--score", default="")                   # deterministic quality of a .pt (pick a ramp source)
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.score:
        score_checkpoint(args.score, amp_max=args.amp_max); return
    if args.eval:
        eval_flat_steering(args.eval); return

    env = SpotRoughEnv(num_envs=args.envs, device="cuda", amp_max=args.amp_max)
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=args.lr, horizon=args.horizon,
              log_std_init=-1.5, entropy=0.0, normalize_obs=False, meta=CONFIG)
    if args.warmstart:                                        # ramp difficulty: keep the converged gait + log_std
        src, _, _ = load_policy(args.warmstart, device="cuda")
        ppo.ac.load_state_dict(src.state_dict())
        print(f"continued from checkpoint {os.path.basename(args.warmstart)} (full actor+critic+log_std)")
    else:
        warmstart_from_isaac(ppo.ac, os.path.join(fetch_assets(), "spot_policy.pt"), device="cuda")
    sanity_walk(env, ppo.ac)                                  # transplant quality (deterministic)
    flat0 = stochastic_flat_baseline(env, ppo.ac)
    gate = args.gate * flat0
    print(f"flat-steering gate = {gate:.3f}  (= {args.gate:.2f} x warm-start STOCHASTIC flat tracking {flat0:.3f})")

    latest = os.path.splitext(args.out)[0] + "_latest.pt"
    best = [-1e9]

    def log(msg):
        trk, ftrk, dist = env.last_track, env.last_flat_track, env.last_climb
        ppo.save(latest)
        ok = ftrk >= gate and env.last_fell <= args.fell_max     # steering preserved AND not falling
        mark = ""
        if trk > best[0] and ok:
            best[0] = trk
            ppo.save(args.out)
            mark = "  <- saved best"
        print(f"{msg} | track {trk:.3f} | flat {ftrk:.3f}{'' if ok else ' LOW!'} | "
              f"dist {dist:.2f} | fell {env.last_fell:.3f}{mark}")

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(latest)
    print(f"saved -> {args.out} (best track {best[0]:.3f}, steering gate {gate:.3f}) + {latest} (final)")
    print(f"next: python {os.path.basename(__file__)} --eval {args.out}")


if __name__ == "__main__":
    main()

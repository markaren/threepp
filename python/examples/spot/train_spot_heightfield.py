"""spotv2 — train the velocity-tracking policy on the SMOOTH heightfield terrain (spot_heightfield_env).

    python train_spot_heightfield.py --iters 1500
    python train_spot_heightfield.py --warmstart spot_terrain.pt --amp_max 0.18  # continue from the stairs/terrain policy
    python train_spot_heightfield.py --score spot_hf.pt                          # deterministic quality
    python train_spot_heightfield.py --eval  spot_hf.pt                          # flat-steering regression

Warm-starts from scratch_flat_best.pt (50-d clock base gait, normalize_obs=True, stiff gains=90)
via warmstart_scratch_to_terrain (in _common.py, next to this file). --sym_coef uses the 96-d mirror
from spot_steps_symmetry (same as train_spot_steps).
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
from spot_heightfield_env import ACT_DIM, CONFIG, HF_AMP_MAX, HIDDEN, SpotHeightfieldEnv
from spot_steps_symmetry import make_aux_loss
from _common import (warmstart_scratch_to_terrain, sanity_walk, stochastic_flat_baseline,
                     eval_flat_steering)
from threepp.rl import PPO, load_policy


@torch.no_grad()
def score_checkpoint(policy_path, k=512, amp_max=HF_AMP_MAX, device="cuda", steps=500, warm=150):
    """Deterministic track / flat / fell of a checkpoint on the heightfield (pick a deploy/ramp source)."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return None
    ac, norm, meta = load_policy(policy_path, device=device)
    src = meta.get("height_source", "analytic")
    print(f"scoring against the {src} height source")
    env = SpotHeightfieldEnv(num_envs=k, device=device, amp_max=amp_max, height_source=src)
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    obs = env.reset()
    trk, flt, fl = [], [], []
    for t in range(steps):
        obs, _, _, _, _ = env.step(pol(obs))
        if t >= warm:
            trk.append(env.last_track); flt.append(env.last_flat_track); fl.append(env.last_fell)
    m = lambda a: sum(a) / max(1, len(a))
    print(f"[score] {os.path.basename(policy_path)}  (deterministic, K={k}, amp_max={amp_max}, {steps - warm} steps)")
    print(f"        track {m(trk):.3f}/2.0   flat {m(flt):.3f}/2.0   fell/step {m(fl):.4f}")
    return m(trk), m(flt), m(fl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)        # ~1344-tri trimesh / lane; builds in ~18s
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp_max", type=float, default=HF_AMP_MAX)   # top amplitude (heightfield stays SMOOTH high)
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_hf.pt"))
    ap.add_argument("--warmstart", default="",
                    help="continue from a previous .pt (e.g. spot_terrain.pt or a prior spot_hf.pt) instead of scratch_flat")
    ap.add_argument("--fell_max", type=float, default=0.005)
    ap.add_argument("--sym_coef", type=float, default=1.0)   # left-right symmetry augmentation weight (0 = off)
    ap.add_argument("--height-source", dest="height_source", choices=("analytic", "raycast"),
                    default="analytic",
                    help="where the terrain height (h_here + the 45-cell scan) comes from. analytic = "
                         "amp * bilinear(grid); raycast = a ray into a Warp BVH over the triangle soup "
                         "PhysX actually cooked (threepp.rl.raycast) — the surface the feet stand on, "
                         "which the bilinear field approximates to about a centimetre. Recorded in the "
                         "checkpoint meta, so --score/--eval follow it.")
    ap.add_argument("--eval", default="")
    ap.add_argument("--score", default="")
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.score:
        score_checkpoint(args.score, amp_max=args.amp_max); return
    if args.eval:
        eval_flat_steering(SpotHeightfieldEnv, args.eval, height_source=True); return

    env = SpotHeightfieldEnv(num_envs=args.envs, device="cuda", amp_max=args.amp_max,
                             height_source=args.height_source)
    if args.height_source == "raycast":
        print(f"terrain height by raycast: {env.rays}")
    aux = make_aux_loss(args.sym_coef) if args.sym_coef > 0 else None
    # height_source rides in the meta so --score and --eval rebuild the env the way it was trained.
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=args.lr, horizon=args.horizon, log_std_init=-1.5,
              entropy=0.0, normalize_obs=True,
              meta={**CONFIG, "height_source": args.height_source}, aux_loss=aux)
    if aux is not None:
        print(f"symmetry augmentation ON (coef {args.sym_coef})")
    if args.warmstart:                                        # continue from a prior 96-d normalized .pt
        src, src_norm, _ = load_policy(args.warmstart, device="cuda")
        ppo.ac.load_state_dict(src.state_dict())
        if ppo.norm is not None and src_norm is not None:
            ppo.norm.load(src_norm.state())
        print(f"continued from checkpoint {os.path.basename(args.warmstart)} (full actor+critic+log_std+norm)")
    else:
        warmstart_scratch_to_terrain(ppo.ac, ppo.norm,
                                     os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt"),
                                     device="cuda")
    sanity_walk(env, ppo.ac, ppo.norm)
    flat0 = stochastic_flat_baseline(env, ppo.ac, ppo.norm)
    gate = args.gate * flat0
    print(f"flat-steering gate = {gate:.3f}  (= {args.gate:.2f} x warm-start STOCHASTIC flat tracking {flat0:.3f})")

    latest = os.path.splitext(args.out)[0] + "_latest.pt"
    best = [-1e9]

    def log(msg):
        trk, ftrk, dist = env.last_track, env.last_flat_track, env.last_climb
        ppo.save(latest)
        ok = ftrk >= gate and env.last_fell <= args.fell_max
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
    print(f"next: python {os.path.basename(__file__)} --score {args.out}")


if __name__ == "__main__":
    main()

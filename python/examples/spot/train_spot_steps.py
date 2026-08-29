"""spotv2 — train the velocity-tracking policy to climb DISCRETE STAIRS with an adaptive curriculum.

    python train_spot_steps.py --iters 1500                       # warm-starts from scratch_flat_best.pt (default)
    python train_spot_steps.py --score spot_steps.pt              # deterministic track/flat/fell + curriculum level
    python train_spot_steps.py --eval  spot_steps.pt              # flat-steering regression vs the base gait

The curriculum (per-env level, promote on clearing the tent / demote on a fall) lives in SpotStepsEnv.
The warm-start transfers the 50-d clock base gait (scratch_flat_best.pt, normalize_obs=True, stiff gains)
into the 96-d AC: input cols [0:50] copied (proprio+clock), terrain cols [50:96] zero-init; RunningNorm
expanded with matching stats (terrain dims left at fresh default mean=0/var=1 so they adapt freely).
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from spot_terrain_env import quat_rotate_inverse
from spot_steps_env import ACT_DIM, CONFIG, HIDDEN, N_LEVELS, RISERS, SpotStepsEnv
from spot_steps_symmetry import make_aux_loss
from threepp.rl import PPO, load_policy


def warmstart_scratch_to_terrain(ac, norm, scratch_path, n_keep=50, device="cuda"):
    """Transfer the 50-d clock base gait into the 96-d clock+terrain AC.

    Input-layer cols [0:n_keep] copied (proprio+clock), terrain cols zero-init; deeper layers +
    log_std copied verbatim. The obs RunningNorm is expanded: proprio+clock stats kept, terrain
    dims left at the fresh default (mean 0 / var 1), count reset so the norm adapts to the steps
    distribution during fine-tune (sanity_walk pre-train still sees the loaded proprio stats =
    bit-identical base gait)."""
    src_ac, src_norm, src_meta = load_policy(scratch_path, device=device)
    assert src_meta["obs_dim"] == n_keep, f"expected {n_keep}-d source, got {src_meta['obs_dim']}"
    assert norm is not None and src_norm is not None, "both sides need a RunningNorm"
    with torch.no_grad():
        for dst_net, src_net in ((ac.actor, src_ac.actor), (ac.critic, src_ac.critic)):
            for li in (0, 2, 4, 6):
                dst, src = dst_net[li], src_net[li]
                if li == 0:
                    dst.weight.zero_()
                    dst.weight[:, :n_keep].copy_(src.weight)   # proprio+clock cols; terrain stays 0
                    dst.bias.copy_(src.bias)
                else:
                    dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
        ac.log_std.copy_(src_ac.log_std)
        norm.mean[:n_keep].copy_(src_norm.mean)
        norm.var[:n_keep].copy_(src_norm.var)
        # leave norm.mean[n_keep:]=0, var[n_keep:]=1 (fresh defaults); keep norm.count fresh (eps)
        # so terrain dims (and proprio under the new distribution) adapt during fine-tune.
    print(f"warm-started clock base gait {os.path.basename(scratch_path)} -> 96-d: cols[:{n_keep}] "
          f"copied, {ac.actor[0].weight.shape[1]-n_keep} terrain cols zero-init; norm expanded")


@torch.no_grad()
def sanity_walk(env, ac, norm, steps=300):
    """Roll the warm-started DETERMINISTIC policy and report tracking. norm-aware: ac.act_mean(norm.norm(obs))."""
    obs = env.reset()
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ac.act_mean(norm.norm(obs)))
    print(f"warm-start sanity ({steps} steps): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"level={env.last_level:.2f}  fell/step={env.last_fell:.3f}")
    return env.last_flat_track


@torch.no_grad()
def stochastic_flat_baseline(env, ac, norm, steps=240, warm=80):
    """Mean flat-lane tracking under the STOCHASTIC policy (exploration noise).
    Gate baseline calibrated to this — not to the deterministic number — to avoid
    it reading LOW every iter from noise alone."""
    obs = env.reset()
    acc = []
    for t in range(steps):
        obs, _, _, _, _ = env.step(ac.act(norm.norm(obs))[0])
        if t >= warm:
            acc.append(env.last_flat_track)
    return sum(acc) / max(1, len(acc))


@torch.no_grad()
def score_checkpoint(policy_path, k=512, device="cuda", steps=900, warm=200, height_source=None,
                     perceive=None):
    """Deterministic track/flat/fell + the curriculum LEVEL it climbs to (how tall a riser it handles).
    The env starts every stair env at level 0 and promotes as the policy clears tents.

    The height source comes from the checkpoint's own meta, so a policy trained on the raycast scan
    is scored against the raycast scan — scoring it against the other one measures the mismatch."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return None
    ac, norm, meta = load_policy(policy_path, device=device)
    trained_on = meta.get("height_source", "analytic")
    src = height_source or trained_on
    saw = bool(meta.get("perceive", False))
    see = saw if perceive is None else bool(perceive)
    if see:
        src = "raycast"
    print(f"scoring against the {src} height source, scan {'PERCEIVED' if see else 'privileged'}"
          + ("" if (src == trained_on and see == saw) else
             f"  (MISMATCH: trained on {trained_on}, "
             f"{'perceived' if saw else 'privileged'} scan)"))
    env = SpotStepsEnv(num_envs=k, device=device, height_source=None if see else src,
                       perceive=see, perceive_noise=float(meta.get("perceive_noise", 0.0)))
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    obs = env.reset()
    trk, flt, fl = [], [], []
    for t in range(steps):
        obs, _, _, _, _ = env.step(pol(obs))
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
    """Held-out steering regression on FLAT ground vs the scratch base gait (the real steering test)."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    ac, norm, meta = load_policy(policy_path, device=device)
    env = SpotStepsEnv(num_envs=k, device=device, flat_only=True,
                       height_source=meta.get("height_source", "analytic"))
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    # Base gait teacher (50-d, norm-aware): compare against it so steering regression is defined
    # relative to the actual STARTING point of this fine-tune (not the old Isaac TorchScript).
    base_path = os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt")
    base_ac, base_norm, _ = load_policy(base_path, device=device)
    tea = lambda o: base_ac.act_mean(base_norm.norm(o[:, :50]))   # reads the 50-d slice
    grid = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.5)]
    print(f"flat-steering regression ({os.path.basename(policy_path)} vs base gait, K={k}):")
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
    ap.add_argument("--warmstart", default=os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt"),
                    help="50-d clock base gait (.pt) to expand into 96-d; default = scratch_flat_best.pt")
    ap.add_argument("--fell_max", type=float, default=0.006)
    ap.add_argument("--sym_coef", type=float, default=1.0)   # left-right symmetry augmentation (kills the veer)
    ap.add_argument("--graph", action="store_true",
                    help="replay the step's torch region from a CUDA graph (VecTask graph=True). "
                         "The region is launch-bound — a few hundred small kernels whose dispatch "
                         "costs far more than their execution — so this is ~1.7x on the step. "
                         "Verified against a fresh eager observation before training starts.")
    ap.add_argument("--perceive", action="store_true",
                    help="train on the scan a body-mounted depth camera could actually have seen "
                         "(threepp.rl.perception): occluded and never-observed cells read flat, and "
                         "the map remembers what the camera swept. Implies --height-source raycast. "
                         "Reward, termination and the spawn keep reading ground truth.")
    ap.add_argument("--perceive-noise", dest="perceive_noise", type=float, default=0.0,
                    help="elevation-map error (m) as a fixed per-cell bias, with --perceive")
    ap.add_argument("--height-source", dest="height_source", choices=("analytic", "raycast"),
                    default="analytic",
                    help="where the terrain height (h_here + the 45-cell scan) comes from. "
                         "analytic = the closed-form tent formula; raycast = a ray into a Warp BVH "
                         "over the boxes actually added to the world (threepp.rl.raycast). The two "
                         "agree exactly inside a lane; the raycast also sees the neighbour's tent. "
                         "It is recorded in the checkpoint meta, so --score/--eval follow it.")
    ap.add_argument("--eval", default="")
    ap.add_argument("--score", default="")
    ap.add_argument("--score-perceive", dest="score_perceive", choices=("on", "off"), default="",
                    help="score with the camera-limited scan on or off, whatever the checkpoint was "
                         "trained with — how much does the policy lose when it only sees what a "
                         "forward depth camera could have seen?")
    ap.add_argument("--score-source", dest="score_source", choices=("analytic", "raycast"), default="",
                    help="score against this height source instead of the one the checkpoint was "
                         "trained on — the deliberate mismatch experiment (how much does the "
                         "difference between the two oracles cost a trained policy?)")
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.score:
        score_checkpoint(args.score, height_source=args.score_source or None,
                         perceive=None if not args.score_perceive else args.score_perceive == "on")
        return
    if args.eval:
        eval_flat_steering(args.eval); return

    env = SpotStepsEnv(num_envs=args.envs, device="cuda", perceive=args.perceive,
                       perceive_noise=args.perceive_noise, graph=args.graph,
                       height_source=None if args.perceive else args.height_source)
    if env.rays is not None:
        print(f"terrain height by raycast: {env.rays}")
    if env.percept is not None:
        print(f"camera-limited scan: {env.percept}")
    aux = make_aux_loss(args.sym_coef) if args.sym_coef > 0 else None
    # height_source rides in the meta so --score and --eval rebuild the env the way it was trained.
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=args.lr, horizon=args.horizon, log_std_init=-1.5,
              entropy=0.0, normalize_obs=True,
              meta={**CONFIG, "height_source": env.height_source, "perceive": args.perceive,
                    "perceive_noise": args.perceive_noise}, aux_loss=aux)
    if aux is not None:
        print(f"symmetry augmentation ON (coef {args.sym_coef})")
    if args.warmstart and os.path.exists(args.warmstart):
        warmstart_scratch_to_terrain(ppo.ac, ppo.norm,
                                     args.warmstart, n_keep=50, device="cuda")
    else:
        print(f"(warmstart path not found: {args.warmstart} — starting from scratch)")
    if args.graph:
        env.reset()
        print(f"CUDA-graph step ON — replay verified to "
              f"{env.verify_graph(steps=32):.1e} against a recomputed observation")
    sanity_walk(env, ppo.ac, ppo.norm)
    flat0 = stochastic_flat_baseline(env, ppo.ac, ppo.norm)
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

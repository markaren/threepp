"""spotv2 — STAIRS/terrain trainer: warm-start from scratch_flat_best.pt (50-d clock base gait,
normalize_obs=True, stiff gains) into the 96-d clock+terrain obs, then PPO-fine-tune.

    python train_spot_stairs.py --iters 1500
    python train_spot_stairs.py --eval spot_terrain.pt     # held-out flat-steering regression vs base gait

Shared helpers (warmstart_scratch_to_terrain, sanity_walk, stochastic_flat_baseline, eval_flat_steering)
are defined here and imported by train_spot_heightfield.
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
from threepp.rl import PPO, load_policy


def warmstart_scratch_to_terrain(ac, norm, scratch_path, n_keep=50, device="cuda"):
    """Transfer the 50-d clock base gait into the 96-d clock+terrain AC.

    Input-layer cols [0:n_keep] copied (proprio+clock), terrain cols zero-init; deeper layers +
    log_std copied verbatim. The obs RunningNorm is expanded: proprio+clock stats kept, terrain
    dims left at the fresh default (mean 0 / var 1), count reset so the norm adapts to the terrain
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
          f"climb={env.ep_max_climb.mean().item() if hasattr(env, 'ep_max_climb') else 'n/a'}  fell/step={env.last_fell:.3f}")
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
def eval_flat_steering(policy_path, k=512, device="cuda"):
    """Held-out steering regression on FLAT ground vs the scratch base gait (the real steering test)."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    env = SpotTerrainEnv(num_envs=k, device=device, flat_only=True)
    ac, norm, _ = load_policy(policy_path, device=device)
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
        eval_flat_steering(args.eval); return

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

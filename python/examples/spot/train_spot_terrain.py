"""spotv2 — Phase 1 trainer: warm-start the Isaac Spot walker, PPO-fine-tune on tents with a
velocity-TRACKING objective + randomized commands (see spot_terrain_env.py for the design).

    python train_spot_terrain.py --iters 1500
    python train_spot_terrain.py --eval spot_terrain.pt     # held-out flat-steering regression vs the teacher

The actor is warm-started from the Isaac flat walker into a 58-d obs whose 10 terrain columns
zero-init, so the policy BEGINS bit-identical to the flat walker (steering exists from iteration 0)
and only adapts. Hypers mirror the working stairs fine-tune: gentle lr 1e-4, low log_std, raw obs
(normalize_obs=False is REQUIRED -- a RunningNorm would shift obs out from under the loaded Isaac
first-layer weights).

CHECKPOINTING preserves steering by construction: the BEST policy (-> --out, what deploy uses) is the
one with the highest overall velocity tracking SUBJECT TO flat-lane tracking staying >= 90% of the
warm-started teacher's (a Pareto gate -- a checkpoint that quietly traded steering for climbing is
rejected). The always-current policy is saved to <out>_latest.pt every log.
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
from spot_terrain_env import ACT_DIM, CONFIG, HIDDEN, N_DX, N_DY, N_SCAN, SCAN_CENTER, SpotTerrainEnv
from threepp.rl import PPO, load_policy


def warmstart_from_isaac(ac, policy_path, n_proprio=48, device="cuda"):
    """Load the Isaac TorchScript actor into ac.actor, expanding the first layer's input: the first
    n_proprio columns get the Isaac weights, the rest (the terrain scan) are zero -> the policy starts
    as the exact flat walker and learns to use the new inputs. Critic + log_std are left fresh."""
    m = torch.jit.load(policy_path, map_location=device).eval()
    p = dict(m.named_parameters())
    with torch.no_grad():
        for li in (0, 2, 4, 6):                       # the Linear layers (ELU between); same indexing both sides
            dst = ac.actor[li]
            w, b = p[f"actor.{li}.weight"], p[f"actor.{li}.bias"]
            if li == 0:
                dst.weight.zero_()
                dst.weight[:, :n_proprio].copy_(w)    # proprio columns from Isaac; terrain columns stay 0
            else:
                dst.weight.copy_(w)
            dst.bias.copy_(b)
    print(f"warm-started actor from {os.path.basename(policy_path)} (terrain input cols zero-init)")


def warmstart_expand_scan(ac, old_path, device="cuda"):
    """Transfer a trained 1-D-scan policy (old obs: 48 proprio + base_above + 9 forward-scan) into the new
    AC whose obs carries the 2-D scan GRID. Copy proprio + base_above verbatim, map each old forward-scan
    column onto the new grid's CENTERLINE row (dy=0), and zero-init the lateral columns -> the policy BEGINS
    bit-identical to the old walker (lateral cols contribute nothing) and only learns to exploit the lateral
    scan. Expands the input layer of BOTH actor and critic; deeper layers + log_std copy directly. Both
    sides use raw obs (normalize_obs=False), so no RunningNorm transfer."""
    old, _, _ = load_policy(old_path, device=device)
    HEAD = 48 + 1                                  # proprio (48) + base_above (1), copied as-is
    with torch.no_grad():
        for dst_net, src_net in ((ac.actor, old.actor), (ac.critic, old.critic)):
            for li in (0, 2, 4, 6):                # Linear layers (ELU between); index 0 is the input layer
                dst, src = dst_net[li], src_net[li]
                if li == 0:
                    dst.weight.zero_()
                    dst.weight[:, :HEAD].copy_(src.weight[:, :HEAD])                    # proprio + base_above
                    for fi in range(N_DX):                                              # old forward probe fi ...
                        dst.weight[:, HEAD + fi * N_DY + SCAN_CENTER].copy_(src.weight[:, HEAD + fi])  # ... -> grid centerline cell
                    dst.bias.copy_(src.bias)
                else:
                    dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
        ac.log_std.copy_(old.log_std)
    print(f"warm-started (1-D->2-D scan transfer) from {os.path.basename(old_path)}: centerline row copied, "
          f"{N_SCAN - N_DX} lateral scan cols zero-init (policy begins == the 1-D climber)")


@torch.no_grad()
def sanity_walk(env, ac, steps=300):
    """Roll the warm-started DETERMINISTIC policy on the tents and report tracking + peak climb — a
    correct warm-start should already track commands + climb the low lanes (~the Isaac walker), proving
    the gait survived the 48->58 transplant before we spend iterations on it."""
    obs = env.reset()
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ac.act_mean(obs))
    print(f"warm-start sanity ({steps} steps): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"climb={env.ep_max_climb.mean().item():.3f} m  fell/step={env.last_fell:.3f}")
    return env.last_flat_track


@torch.no_grad()
def stochastic_flat_baseline(env, ac, steps=240, warm=80):
    """Mean flat-lane tracking under the SAME STOCHASTIC policy the rollouts use (log_std exploration
    noise lowers tracking ~0.2 vs deterministic). The steering gate must be calibrated to THIS, not to
    the deterministic sanity number, or it reads 'LOW' from iteration 1 on exploration noise alone."""
    obs = env.reset()
    acc = []
    for t in range(steps):
        obs, _, _, _, _ = env.step(ac.act(obs)[0])
        if t >= warm:
            acc.append(env.last_flat_track)
    return sum(acc) / max(1, len(acc))


@torch.no_grad()
def eval_flat_steering(policy_path, k=512, device="cuda"):
    """Held-out STEERING REGRESSION: on a FLAT-only env, sweep a command grid and compare the trained
    policy's tracking error against the frozen Isaac teacher's. PASS = policy within ~1.10x the teacher
    on every command (so steering was NOT degraded by the terrain fine-tune)."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    env = SpotTerrainEnv(num_envs=k, device=device, flat_only=True)
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
              log_std_init=-1.5, entropy=0.0, normalize_obs=False, meta=CONFIG)
    warmstart_from_isaac(ppo.ac, os.path.join(fetch_assets(), "spot_policy.pt"), device="cuda")
    sanity_walk(env, ppo.ac)                                  # transplant quality (deterministic): tracks + climbs?
    flat0 = stochastic_flat_baseline(env, ppo.ac)            # gate baseline, calibrated to the stochastic rollouts
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

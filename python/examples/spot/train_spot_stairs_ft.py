"""Fine-tune the Isaac flat walker to CLIMB STAIRS (warm-start, not frozen residual).

    python train_spot_stairs_ft.py --iters 1500

The parked residual approach froze the flat Isaac walker and learned a small additive correction; that
self-fight capped climbing at ~0.17 m. This instead warm-starts the Isaac actor (48->512->256->128->12
ELU) into a wider actor whose observation also carries a 10-d terrain scan — the 10 new input columns
start at ZERO, so the policy BEGINS as the exact flat walker — then PPO fine-tunes the WHOLE gait on
graded stairs (raw obs, normalize_obs=False, to match the Isaac Identity normalizer). Goal: beat 0.17 m.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from spot_deploy import fetch_assets
from spot_stairs_env import ACT_DIM, FT_CONFIG, FT_HIDDEN, SpotStairFTEnv
from threepp.rl import PPO


def warmstart_from_isaac(ac, policy_path, n_proprio=48, device="cuda"):
    """Load the Isaac TorchScript actor into ac.actor, expanding the first layer's input: the first
    n_proprio columns get the Isaac weights, the rest (the terrain scan) are zero — so the policy starts
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


@torch.no_grad()
def sanity_walk(env, ac, steps=300):
    """Roll the warm-started DETERMINISTIC policy on the graded stairs and report mean peak climb — a
    correct warm-start should walk + climb the low lanes (~the blind Isaac walker), proving the gait
    survived the transplant before we spend iterations on it."""
    obs = env.reset()
    for _ in range(steps):
        obs, _, _, _, _ = env.step(ac.act_mean(obs))
    print(f"warm-start sanity ({steps} steps): mean peak climb={env.ep_max_climb.mean().item():.3f} m  "
          f"fell/step={env.last_fell:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)        # gentle: adapt the gait, don't forget it
    ap.add_argument("--rise_max", type=float, default=0.13)  # top step height in the graded lanes (curriculum lever)
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_stairs_ft.pt"))
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)

    env = SpotStairFTEnv(num_envs=args.envs, device="cuda", rise_max=args.rise_max)
    ppo = PPO(env, ACT_DIM, hidden=FT_HIDDEN, lr=args.lr, horizon=args.horizon,
              log_std_init=-1.5, entropy=0.0, normalize_obs=False, meta=FT_CONFIG)
    warmstart_from_isaac(ppo.ac, os.path.join(fetch_assets(), "spot_policy.pt"), device="cuda")
    sanity_walk(env, ppo.ac)

    latest = os.path.splitext(args.out)[0] + "_latest.pt"   # always-current policy (resume / final converged)
    best = [0.0]

    def log(msg):
        c = env.last_climb
        ppo.save(latest)                      # LATEST every log (so the final/converged policy is never lost)
        mark = ""
        if c > best[0]:                       # BEST-by-climb -> args.out, what the viewer hot-reloads & deploy uses
            best[0] = c
            ppo.save(args.out)                #   (PPO oscillates; latest alone can be a post-peak/veered checkpoint)
            mark = "  <- saved best"
        print(f"{msg} | climb {c:.3f} m | fell {env.last_fell:.3f}{mark}")

    ppo.learn(args.iters, log_every=20, on_log=log)
    ppo.save(latest)
    print(f"saved -> {args.out} (best climb {best[0]:.3f} m) + {latest} (final)")


if __name__ == "__main__":
    main()

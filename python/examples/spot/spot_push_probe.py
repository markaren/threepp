"""How hard a shove does the policy survive?

Stability is not a feeling, it is a survival curve. Shove the robot at random moments with a
known impulse and count what fraction of pushes it walks away from, sweeping the magnitude until
it stops walking away from them.

The shove is `threepp.rl.GpuSim.apply_link_force` — a one-substep world-frame impulse on the base,
sized so it delivers a given velocity change (`push_vel`, m/s). That path only exists under
direct-GPU since PhysX rejects `ArticulationLink.add_force` there, so before this the whole
question could not be asked at training scale.

    python spot_push_probe.py                                   # sweep the shipped checkpoint
    python spot_push_probe.py --model a.pt --model b.pt         # compare two

Reported per magnitude: falls per 1000 pushes, and the fraction of envs still upright at the end.
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from threepp.rl import load_policy
from spot_steps_env import SpotStepsEnv, PUSH_MASS

MAGS = [float(v) for v in os.environ.get(
    "PUSH_MAGS", "0.0,0.6,1.2,2.0,3.0,4.0,5.0").split(",")]


@torch.no_grad()
def sweep(model, envs, steps, prob, seed, flat_only):
    torch.manual_seed(seed)
    env = SpotStepsEnv(num_envs=envs, seed=seed, flat_only=flat_only, graph=True,
                       push_vel=max(MAGS), push_prob=prob)
    ac, norm, meta = load_policy(model, device=env.device)
    ac.eval()
    act = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    print(f"\n=== {os.path.basename(model)} "
          f"(trained with push_vel={meta.get('push_vel', 'n/a')}) ===")
    print(f"{'push':>7} {'impulse':>9} {'falls/1k pushes':>17} {'upright at end':>15} {'track':>7}")
    for mag in MAGS:
        env.push_vel = mag
        obs = env.reset()
        falls = pushes = 0
        trk = []
        for _ in range(steps):
            obs, _, done, _, timeout = env.step(act(obs))
            falls += int((done & ~timeout).sum())      # a true terminal is a fall, not a time limit
            pushes += env.last_push * env.K
            trk.append(env.last_track)
        up = (env.up > 0.35).float().mean().item()
        # With no pushes there is no falls-PER-PUSH to report; dividing by a floor of 1 turns the
        # handful of natural falls into a headline "1000 per 1000", which is nonsense.
        rate = (f"{falls / pushes * 1000.0:16.1f}" if pushes >= 10
                else f"{'(' + str(falls) + ' falls, no pushes)':>16s}")
        print(f"  {mag:5.2f} {mag * PUSH_MASS:7.1f}Ns {rate} {up * 100:14.1f}% "
              f"{np.mean(trk[len(trk) // 4:]):7.3f}")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--prob", type=float, default=0.02, help="per-env per-step push probability")
    ap.add_argument("--flat", action="store_true", help="flat ground only (isolate the shove)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return 0
    for m in (args.model or ["spot_steps.pt"]):
        path = m if os.path.isabs(m) else os.path.join(_HERE, m)
        if not os.path.exists(path):
            print(f"\n{m}: not found"); continue
        sweep(path, args.envs, args.steps, args.prob, args.seed, args.flat)
    return 0


if __name__ == "__main__":
    sys.exit(main())

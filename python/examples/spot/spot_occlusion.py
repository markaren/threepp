"""How much of the 45-cell height scan could Spot's own camera actually have seen?

The policy is trained on a privileged scan: the exact terrain height at every grid cell. At deploy
those 45 numbers come from a body-mounted depth camera fused into an elevation map, and they are
not the same numbers — a cell behind a riser is occluded, a cell outside the frustum was never
observed, and a cell under the robot is known only from memory. `perceive=True` puts that camera
into the training loop (threepp.rl.perception); this script measures what it costs.

    python spot_occlusion.py                     # census + obs error, 512 envs
    python spot_occlusion.py --envs 2048 --steps 300
    python spot_occlusion.py --noise 0.03        # add 3 cm of elevation-map error

Reported per cell of the 9-forward x 5-lateral grid: how often the camera sees it directly this
tick, how often the answer comes out of memory instead, and how often there is no answer at all
and the cell reads flat. Then the same comparison as a height error against the privileged scan
the policy was trained on.
"""
import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from threepp.rl import load_policy
from spot_steps_env import SpotStepsEnv, N_LEVELS
from spot_terrain_env import scan_xy, heading_cossin, PROBE_DX, PROBE_DY, N_SCAN


def _grid(vals, fmt="{:5.0f}"):
    """45 numbers -> the 9-forward x 5-lateral grid they live on (forward-major, +y = LEFT)."""
    n_dy = len(PROBE_DY)
    head = "   dx\\dy " + " ".join(f"{dy:>+5.2f}" for dy in PROBE_DY)
    rows = [head]
    for i, dx in enumerate(PROBE_DX):
        cells = " ".join(fmt.format(vals[i * n_dy + j]) for j in range(n_dy))
        rows.append(f"  {dx:+5.2f}  {cells}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.0, help="elevation-map error (m, per-cell bias)")
    ap.add_argument("--policy", default="spot_steps.pt")
    ap.add_argument("--no-policy", action="store_true", help="stand still instead of driving")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds in the policy A/B")
    ap.add_argument("--roll", type=int, default=400, help="steps per rollout in the policy A/B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return 0

    env = SpotStepsEnv(num_envs=args.envs, seed=args.seed, perceive=True, perceive_noise=args.noise)
    dev = env.device
    print(f"\n{env.rays}\n{env.percept}")

    act = None
    if not args.no_policy:
        ckpt = os.path.join(_HERE, args.policy)
        if os.path.exists(ckpt):
            ac, norm, _ = load_policy(ckpt, device=dev)
            ac.eval()
            act = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
            print(f"driving {os.path.basename(ckpt)}")
    if act is None:
        act = lambda o: torch.zeros(args.envs, env.act_dim, device=dev)
        print("standing (zero actions)")

    g = torch.Generator(device=dev).manual_seed(args.seed)
    env.level = torch.randint(0, N_LEVELS, (args.envs,), device=dev, generator=g)
    obs = env.reset()

    seen_n = torch.zeros(N_SCAN, device=dev)          # directly visible this tick
    known_n = torch.zeros(N_SCAN, device=dev)         # answerable (this tick or memory)
    mem_n = torch.zeros(N_SCAN, device=dev)           # answerable but NOT visible right now
    here_n = 0.0                                      # ticks the map could answer h_here
    err_sum = torch.zeros(N_SCAN, device=dev)
    err_max = torch.zeros(N_SCAN, device=dev)
    blind_err = torch.zeros(N_SCAN, device=dev)       # error contributed by cells with NO answer
    ticks = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.steps):
            # obs was produced by observe() on the CURRENT state, so the census matches this obs
            s = env.state
            px, py = scan_xy(s.root_pos[:, 0], s.root_pos[:, 1], *heading_cossin(s.root_quat),
                             env.gx, env.gy)
            truth_here = env._terrain_h(s.root_pos[:, 0], s.root_pos[:, 1])
            truth = (env._terrain_h(px, py) - truth_here[:, None]).clamp(-1.0, 1.0)
            got = obs[:, -N_SCAN:]
            d = (got - truth).abs()
            vis, kn = env.percept.last_visible.float(), env.percept.last_known.float()
            seen_n += vis.mean(0)
            known_n += kn.mean(0)
            mem_n += (kn * (1.0 - vis)).mean(0)        # elementwise: never negative
            here_n += env.percept.last_here_known.float().mean().item()
            err_sum += d.mean(0)
            err_max = torch.maximum(err_max, d.max(0).values)
            blind_err += (d * (1.0 - kn)).mean(0)
            ticks += 1
            obs, _, _, _, _ = env.step(act(obs))
    dt = (time.perf_counter() - t0) / ticks * 1e3

    vis_pct = (seen_n / ticks * 100).tolist()
    mem_pct = (mem_n / ticks * 100).tolist()
    unk_pct = (100 - known_n / ticks * 100).tolist()

    print(f"\n[1] directly visible this tick (% of ticks the camera has line of sight), "
          f"{ticks} steps x {args.envs} envs")
    print(_grid(vis_pct))
    print(f"\n[2] answered from MEMORY instead (seen earlier, occluded or out of frame now)")
    print(_grid(mem_pct))
    print(f"\n[3] NO answer — the cell reads flat, as it does at deploy")
    print(_grid(unk_pct))
    print(f"\n  overall: {sum(vis_pct)/N_SCAN:5.1f}% visible  {sum(mem_pct)/N_SCAN:5.1f}% remembered  "
          f"{sum(unk_pct)/N_SCAN:5.1f}% blind")

    print(f"\n[4] height error vs the privileged scan (m), mean over env x tick")
    print(_grid((err_sum / ticks).tolist(), fmt="{:5.3f}"))
    e = err_sum / ticks
    print(f"\n  mean {e.mean().item():.4f} m   worst cell {e.max().item():.4f} m   "
          f"worst single reading {err_max.max().item():.4f} m")
    print(f"  of the mean error, {blind_err.sum().item() / max(err_sum.sum().item(), 1e-9) * 100:.1f}% "
          f"comes from cells with no answer at all")
    print(f"  the map could answer h_here on {here_n / ticks * 100:.1f}% of ticks; the rest reuse "
          f"the last height it could, as the deploy scanner does")
    print(f"\n  env.step with perception: {dt:.2f} ms at K={args.envs}")

    # ---- does it change what the policy does? -------------------------------------------
    # Same env, same spawns, the observation switched between the camera-limited scan and the
    # privileged one. Interleaved, and spread over every riser band rather than the low levels
    # the curriculum would leave a fresh policy on.
    print(f"\n[5] the policy under each scan — {args.rounds} interleaved rounds of {args.roll} steps, "
          f"levels spread over all {N_LEVELS} bands")
    saved = env.percept
    score = {"perceived": [], "privileged": []}
    with torch.no_grad():
        for r in range(args.rounds):
            for arm in ("perceived", "privileged"):
                env.percept = saved if arm == "perceived" else None
                if env.percept is not None:
                    env.percept.forget()
                g2 = torch.Generator(device=dev).manual_seed(1000 + r)
                env.level = torch.randint(0, N_LEVELS, (args.envs,), device=dev, generator=g2)
                o = env.reset()
                trk, fell = 0.0, 0.0
                for t in range(args.roll):
                    o, _, _, _, _ = env.step(act(o))
                    if t >= args.roll // 4:                      # skip the settle transient
                        trk += env.last_track
                        fell += env.last_fell
                n = args.roll - args.roll // 4
                score[arm].append((trk / n, fell / n))
    env.percept = saved
    for arm in ("privileged", "perceived"):
        t = [v[0] for v in score[arm]]
        f = [v[1] for v in score[arm]]
        print(f"  {arm:<11s} track {sum(t)/len(t):.4f}  fell/step {sum(f)/len(f):.5f}   "
              f"(rounds {['%.3f' % v for v in t]})")
    tp_, tv = ([v[0] for v in score['perceived']], [v[0] for v in score['privileged']])
    print(f"  perceived / privileged tracking: "
          f"{sum(tp_)/len(tp_) / max(sum(tv)/len(tv), 1e-9) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

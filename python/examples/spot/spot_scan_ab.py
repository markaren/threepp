"""A/B the Warp raycast height scan against the analytic oracle it replaces.

Every terrain env has always answered "how high is the ground here?" with a closed-form formula
it evaluates in torch. `height_source="raycast"` answers it instead by casting a ray down into a
Warp BVH over the geometry the world was actually built from (threepp.rl.raycast). This script
measures whether that changes the answer, and what it costs.

Both arms run the env's own `_terrain_h`, switched by clearing `env.rays`, so this compares the
production code paths rather than a reimplementation of them.

    python spot_scan_ab.py                        # stairs, 512 envs
    python spot_scan_ab.py --env heightfield      # the continuous trimesh terrain
    python spot_scan_ab.py --envs 2048            # the training batch size
    python spot_scan_ab.py --no-policy            # stand still instead of driving a checkpoint

What to expect, and why the two envs differ:

  stairs       the tent formula is EXACT for the boxes inside a lane, so agreement there should be
               bit-for-bit. It is wrong past |dy| = HALF_W_STEPS: the tent boxes are a full lane
               wide and tile with no gap, so the neighbour's tread is really there and the formula
               reports flat ground. Disagreements should be entirely on that side of the line, plus
               a handful of one-riser ties within ~1 um of a tread seam, where the ray takes the
               upper box and floor() takes the lower.

  heightfield  the formula is amp * BILINEAR(grid) but the collider is the TRIANGULATION of that
               same grid -- two different surfaces. The gap is the bilinear-vs-triangle difference,
               and the raycast is the arm that matches what the feet stand on.

Timing is interleaved (ray, analytic, ray, analytic, ...) so a drifting clock or a busy GPU hits
both arms equally.
"""
import argparse
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from threepp.rl import load_policy
from spot_terrain_env import scan_xy, heading_cossin, N_SCAN


def _steps_env():
    from spot_steps_env import SpotStepsEnv, N_LEVELS, STRIP_LEN, HALF_W_STEPS

    def spread(env, g):                       # spawn across the difficulty bands, not all at level 0
        env.level = torch.randint(0, N_LEVELS, (env.K,), device=env.device, generator=g)
    return dict(make=SpotStepsEnv, x_lo=-1.0, x_hi=STRIP_LEN + 1.0, half_w=HALF_W_STEPS,
                spread=spread,
                edge_note="the neighbour lane's tent, which the formula does not model")


def _hf_env():
    from spot_heightfield_env import SpotHeightfieldEnv, HF_X0, HF_X1
    from spot_terrain_env import HALF_W
    # Past |dy| = HALF_W (1.38 m) the tile ends and the ground is flat, until the NEIGHBOUR's tile
    # begins at SPACING - HALF_W = 1.62 m — which the 1.2x-wide sweep just reaches, so this row
    # mixes true flat ground with a slice of the next lane's tapered rim.
    return dict(make=SpotHeightfieldEnv, x_lo=HF_X0 - 1.0, x_hi=HF_X1 + 1.0, half_w=HALF_W,
                spread=lambda env, g: None,
                edge_note="flat ground, plus the neighbour tile's rim past SPACING - HALF_W")


ENVS = {"steps": _steps_env, "heightfield": _hf_env}


def _both(env, x, y):
    """(raycast, analytic) heights at the same points, from the env's own _terrain_h."""
    h_ray = env._terrain_h(x, y)
    saved, env.rays = env.rays, None
    try:
        h_ana = env._terrain_h(x, y)
    finally:
        env.rays = saved
    return h_ray, h_ana


def _row(name, d, tol=1e-3, note=""):
    over = d > tol
    print(f"  {name:<26s} n={d.numel():<9d} max={d.max().item() if d.numel() else 0.0:.6f} m  "
          f"mean={d.mean().item() if d.numel() else 0.0:.2e} m  "
          f"over 1 mm: {int(over.sum())} ({over.float().mean().item()*100 if d.numel() else 0:.4f}%)"
          + (f"   <- {note}" if note else ""))


def _time(fn, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(ENVS), default="steps")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=150, help="control steps of in-the-loop comparison")
    ap.add_argument("--probes", type=int, default=256, help="sweep points per lane")
    ap.add_argument("--iters", type=int, default=100, help="timed calls per interleaved round")
    ap.add_argument("--rounds", type=int, default=5, help="interleaved A/B rounds")
    ap.add_argument("--policy", default="spot_steps.pt",
                    help="checkpoint to drive (the 96-d contract is shared across terrains)")
    ap.add_argument("--no-policy", action="store_true", help="stand still instead of driving a policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return 0

    spec = ENVS[args.env]()
    t0 = time.perf_counter()
    env = spec["make"](num_envs=args.envs, seed=args.seed, height_source="raycast")
    print(f"\n{args.env}: {env.rays}   built in {time.perf_counter() - t0:.2f} s (world + BVH, once)")
    dev = env.device
    half_w = spec["half_w"]
    g = torch.Generator(device=dev).manual_seed(args.seed)

    # ---- 1. geometry sweep: does the BVH describe the same surface as the formula? --------
    print(f"\n[1] geometry sweep — random points over every lane, x in "
          f"[{spec['x_lo']:.1f}, {spec['x_hi']:.1f}], 20% wider than the lane")
    P = args.probes
    x = torch.rand(args.envs, P, device=dev, generator=g) * (spec["x_hi"] - spec["x_lo"]) + spec["x_lo"]
    y = (env.lane_y[:, None]
         + (torch.rand(args.envs, P, device=dev, generator=g) - 0.5) * 2.4 * half_w)
    h_ray, h_ana = _both(env, x, y)
    d = (h_ray - h_ana).abs()
    own = (y - env.lane_y[:, None]).abs() < half_w
    _row("all points", d)
    _row("inside own lane", d[own])
    _row("past the lane edge", d[~own], note=spec["edge_note"])

    # ---- 2. in the loop: the scan the policy actually sees, step after step ---------------
    print(f"\n[2] in the loop — {args.steps} control steps, the 45-cell scan as observed")
    act = None
    if not args.no_policy:
        ckpt = args.policy if os.path.isabs(args.policy) else os.path.join(_HERE, args.policy)
        if os.path.exists(ckpt):
            ac, norm, _ = load_policy(ckpt, device=dev)
            ac.eval()
            act = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
            print(f"  driving {os.path.basename(ckpt)}")
    if act is None:
        act = lambda o: torch.zeros(args.envs, env.act_dim, device=dev)
        print("  standing (zero actions)")
    spec["spread"](env, g)
    obs = env.reset()
    worst = worst_in = 0.0
    n_over = n_in = n_cells = 0
    with torch.no_grad():
        for _ in range(args.steps):
            s = env.state
            px, py = scan_xy(s.root_pos[:, 0], s.root_pos[:, 1], *heading_cossin(s.root_quat),
                             env.gx, env.gy)
            h_ray, h_ana = _both(env, px, py)
            d = (h_ray - h_ana).abs()
            own = (py - env.lane_y[:, None]).abs() < half_w
            worst = max(worst, d.max().item())
            if own.any():
                worst_in = max(worst_in, d[own].max().item())
                n_in += int(((d > 1e-3) & own).sum())
            n_over += int((d > 1e-3).sum())
            n_cells += d.numel()
            obs, _, _, _, _ = env.step(act(obs))
    print(f"  {'observed scan cells':<26s} n={n_cells:<9d} max={worst:.6f} m  "
          f"over 1 mm: {n_over} ({n_over / n_cells * 100:.4f}%)")
    print(f"  {'...of those, own lane':<26s} {'':<9s} max={worst_in:.6f} m  over 1 mm: {n_in}")

    # ---- 3. cost, interleaved ------------------------------------------------------------
    print(f"\n[3] cost — {args.rounds} interleaved rounds of {args.iters} calls, K={args.envs}")
    s = env.state
    px, py = scan_xy(s.root_pos[:, 0], s.root_pos[:, 1], *heading_cossin(s.root_quat),
                     env.gx, env.gy)
    saved = env.rays
    ray_ms, ana_ms = [], []
    for _ in range(2):                                   # warm both paths (kernel load, allocator)
        env.rays = saved; _time(lambda: env._terrain_h(px, py), 10)
        env.rays = None;  _time(lambda: env._terrain_h(px, py), 10)
    for _ in range(args.rounds):
        env.rays = saved
        ray_ms.append(_time(lambda: env._terrain_h(px, py), args.iters))
        env.rays = None
        ana_ms.append(_time(lambda: env._terrain_h(px, py), args.iters))
    env.rays = saved
    r, a = statistics.median(ray_ms), statistics.median(ana_ms)
    print(f"  raycast  {r:7.3f} ms   ({args.envs * N_SCAN / r / 1e3:.0f} M rays/s over "
          f"{env.rays.n_tris} tris)")
    print(f"  analytic {a:7.3f} ms")
    print(f"  ratio    {r / a:7.2f}x   (rounds: ray {['%.3f' % v for v in ray_ms]}, "
          f"ana {['%.3f' % v for v in ana_ms]})")

    zero = torch.zeros(args.envs, env.act_dim, device=dev)
    step_ms = {}
    for _ in range(args.rounds):
        for src in ("raycast", "analytic"):
            env.rays = saved if src == "raycast" else None
            step_ms.setdefault(src, []).append(_time(lambda: env.step(zero), 20))
    env.rays = saved
    sr, sa = statistics.median(step_ms["raycast"]), statistics.median(step_ms["analytic"])
    print(f"  full env.step: raycast {sr:.2f} ms   analytic {sa:.2f} ms   ({(sr - sa) / sa * 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

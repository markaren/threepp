"""Where a Spot training step spends its time, phase by phase, at one env count.

    python spot_step_bench.py --envs 2048 --graph
    python spot_step_bench.py --envs 16384 --graph

Written for the first IDUN H100 runs (2026-09-08): 64.6k steps/s at 2048 envs
and 90k at 16384, against 52.6k on an RTX 4070 at 2048 -- eight times the
environments bought 1.4 times the throughput, so something per step is not
scaling with the GPU. This splits the step so the flat part can be named.

Phases, each the median of --repeats runs between device syncs:
  physics   SUBSTEPS batch.step() advances with no state read
  read      sim.read() alone: one CPU<->GPU sync per state buffer
  env.step  the full control step (physics + read + the torch obs/reward region,
            replayed from a CUDA graph when --graph)
  ppo iter  one PPO iteration = horizon env.steps + the update; the update is
            reported as the remainder
Zero actions keep the robots standing, so no resets perturb the timing; a
training step also pays for resets and the policy forward, so expect the
trainer's steps/s a little under the env.step figure here. Saves nothing and
loads no checkpoint.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

from spot_steps_env import ACT_DIM, HIDDEN, SpotStepsEnv   # noqa: E402
from threepp.rl import PPO                                  # noqa: E402


def timed(fn, repeats):
    torch.cuda.synchronize()
    fn()                                   # warm: allocations, first-call paths
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def gpu_mem_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:                      # noqa: BLE001 - informational only
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    K, T = args.envs, args.horizon

    t0 = time.perf_counter()
    env = SpotStepsEnv(num_envs=K, device="cuda", graph=args.graph)
    env.reset()                            # the graph capture happens here
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    dt_sub = env.dt / env.substeps
    a = torch.zeros(K, ACT_DIM, device="cuda")

    physics = timed(lambda: [env.sim.batch.step(dt_sub) for _ in range(env.substeps)], args.repeats)
    read = timed(env.sim.read, args.repeats)
    step = timed(lambda: env.step(a), args.repeats)

    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, horizon=T, normalize_obs=True, log_std_init=-1.5)
    ppo.learn(1, log_every=10 ** 9)        # warm-up iteration (prints its own line)
    torch.cuda.synchronize()
    t = time.perf_counter()
    ppo.learn(1, log_every=10 ** 9)
    torch.cuda.synchronize()
    it_ms = (time.perf_counter() - t) * 1e3
    update = it_ms - T * step

    torch_region = step - physics - read
    print(f"\n{K:,} envs, substeps {env.substeps}, graph {'on' if args.graph else 'off'}, "
          f"build {build_s:.1f}s, GPU memory used {gpu_mem_mb()} MB, "
          f"CPU cores {os.cpu_count()} (sched: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'})")
    print(f"  {'phase':<12}{'ms':>9}{'share':>8}")
    for name, ms in (("physics", physics), ("read", read), ("torch region", torch_region)):
        print(f"  {name:<12}{ms:9.2f}{100 * ms / step:7.0f}%")
    print(f"  {'env.step':<12}{step:9.2f}   = {K / step * 1e3 / 1e3:7.1f}k env-steps/s")
    print(f"  {'ppo update':<12}{update:9.2f}   per iteration of {T} steps "
          f"({100 * update / it_ms:.0f}% of the iteration)")
    print(f"  {'ppo iter':<12}{it_ms:9.2f}   = {K * T / it_ms * 1e3 / 1e3:7.1f}k env-steps/s")


if __name__ == "__main__":
    main()

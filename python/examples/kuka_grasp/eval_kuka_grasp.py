"""eval_kuka_grasp.py — the fixed evaluation: held-out Reaching starts, deterministic policy.

Every env starts in its own cell of the pool's eval split (256 Reaching configurations, half
with the arm near DEFAULT_Q, half IK-aligned above the cube; a different seed from the
training pool), runs one episode under the policy's mean action, and scores the shared
success definition (kuka_grasp_env.py). The same env and cells serve every arm, the scaffold
included, so the numbers compare.

The env takes each policy's action semantics from its checkpoint (delta for the pool arms,
the June absolute mapping for the scaffold) and that arm's terminal rules: a velocity over
twice the limit ends a pool-arm episode but not a scaffold one, as in their training; every
policy's violation rate is reported. Observation, cells and success are shared.

PhysX allows one foundation per process, so a training process cannot also hold this env;
the evaluation runs as its own process over the snapshots the trainer writes:

    python eval_kuka_grasp.py --policy kuka_grasp.pt                  # one policy
    python eval_kuka_grasp.py --oracle                                # scripted IK grasp, the reference
    python eval_kuka_grasp.py --watch runs/sgs_s0 runs/uniform_s0     # follow training runs

--watch evaluates each run's ckpt/itNNNNN.pt in order as they appear and appends one JSON
line per snapshot to <run>/eval.jsonl, until every run has written final.pt and every
snapshot is evaluated.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_env import KukaGraspEnv  # noqa: E402
from kuka_grasp_kin import down_frame  # noqa: E402
from kuka_grasp_pool import load_pool  # noqa: E402
from threepp.rl import load_policy  # noqa: E402


def make_env(device="cuda", graph=True):
    pool = load_pool(device=device, split="eval")
    return KukaGraspEnv(pool["n"], device=device, mode="pool", pool=pool,
                        cells=torch.arange(pool["n"], device=device), graph=graph)


@torch.no_grad()
def evaluate(env, act_fn, action="delta"):
    """One episode per env under the action semantics the policy was trained with (its
    checkpoint's meta["action"]: "delta" for the pool arms, "absolute" for the scaffold);
    returns the per-cell outcome and a summary dict."""
    env.delta = action == "delta"
    env.vel_terminal.fill_(action == "delta")
    obs = env.reset()
    fail_cube = torch.zeros(env.K, dtype=torch.bool, device=env.device)
    fail_vel = torch.zeros_like(fail_cube)
    seen = torch.zeros_like(fail_cube)
    violated = torch.zeros_like(fail_cube)     # any step over twice a velocity limit, first episode
    t_success = torch.full((env.K,), float("nan"), device=env.device)
    for t in range(env.max_steps):
        obs, _, done, _, _ = env.step(act_fn(obs, t))
        first = done & ~seen
        violated |= ~seen & env.fail_vel
        fail_cube |= first & env.fail_cube & ~env.success
        fail_vel |= first & env.fail_vel & ~env.success & env.vel_terminal
        t_success = torch.where(first & env.success, torch.full_like(t_success, (t + 1) * env.dt), t_success)
        seen |= done
    succ = (env.first_outcome == 1)
    sub = env.pool["sub"]
    out = {"success": succ.float().mean().item(),
           "success_home": succ[sub == 0].float().mean().item(),
           "success_aligned": succ[sub == 1].float().mean().item(),
           "fail_cube": fail_cube.float().mean().item(),
           "fail_vel": fail_vel.float().mean().item(),
           "timeout": (~succ & ~fail_cube & ~fail_vel).float().mean().item(),
           "vel_violation": violated.float().mean().item(), "action": action,
           "t_success_median": float(np.nanmedian(t_success.cpu().numpy())) if succ.any() else float("nan"),
           "cells": env.K}
    return succ, out


def policy_fn(path, device):
    ac, norm, meta = load_policy(path, device)
    ac.eval()

    def act(obs, t):
        return ac.act_mean(norm.norm(obs) if norm is not None else obs)
    return act, meta


def oracle_fn(env):
    """A scripted top-down grasp with privileged cube poses, through the same action interface:
    above the cube (8 cm), down to the grasp height, close, lift 0.16 m, hold."""
    dev, K = env.device, env.K
    cube = env.pool["cube"]
    x, y = cube[:, 4], cube[:, 5]
    yaw = 2.0 * torch.atan2(cube[:, 2], cube[:, 3])
    q0 = torch.tensor(C.DEFAULT_Q, device=dev).expand(K, 7).clone()
    q0[:, 0] = torch.atan2(y, x)
    _, R0 = env.kin.grasp_frame(q0)
    ref = torch.atan2(R0[:, 1, 0], R0[:, 0, 0])
    jaw = yaw + torch.round((ref - yaw) / (np.pi / 2)) * (np.pi / 2)
    Rt = down_frame(jaw)

    def solve(z):
        q, pe, re = env.kin.ik(torch.stack([x, y, torch.full_like(x, z)], 1), Rt, q0)
        return q
    q_above, q_grasp, q_lift = solve(C.GRASP_TABLE_Z + 0.08), solve(C.GRASP_TABLE_Z), solve(C.GRASP_TABLE_Z + 0.16)
    phases = [(40, q_above, -1.0), (60, q_grasp, -1.0), (75, q_grasp, 1.0), (10 ** 9, q_lift, 1.0)]

    def act(obs, t):
        for end, q, g in phases:
            if t < end:
                break
        a = torch.empty(K, C.ACT_DIM, device=dev)
        a[:, 0:7] = ((q - env.target[:, 0:7]) / env.arm_step).clamp(-1.0, 1.0)   # delta actions
        a[:, 7] = g
        return a
    return act


def _snapshots(run):
    out = []
    for p in glob.glob(os.path.join(run, "ckpt", "it*.pt")):
        m = re.search(r"it(\d+)\.pt$", p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def watch(env, runs, poll=20.0):
    done = {r: set() for r in runs}
    for r in runs:
        path = os.path.join(r, "eval.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    done[r].add(json.loads(line)["it"])
    while True:
        did = False
        for r in runs:
            for it, p in _snapshots(r):
                if it in done[r]:
                    continue
                try:
                    act, meta = policy_fn(p, env.device)
                except Exception as e:          # a snapshot being written right now
                    print(f"[eval] {p}: {e}; retrying")
                    continue
                t0 = time.perf_counter()
                _, out = evaluate(env, act, meta.get("action", "absolute"))
                rec = {"it": it, "run": os.path.basename(os.path.normpath(r)),
                       "sampler": meta.get("sampler"), "seed": meta.get("seed"), **out,
                       "eval_s": time.perf_counter() - t0, "t": time.time()}
                with open(os.path.join(r, "eval.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[r].add(it)
                did = True
                print(f"[eval] {rec['run']:14s} it {it:5d}  success {out['success']:.3f} "
                      f"(home {out['success_home']:.3f}, aligned {out['success_aligned']:.3f})  "
                      f"fail_cube {out['fail_cube']:.3f}  fail_vel {out['fail_vel']:.3f}", flush=True)
        finished = all(os.path.exists(os.path.join(r, "final.pt"))
                       and {it for it, _ in _snapshots(r)} <= done[r] for r in runs)
        if finished:
            return
        if not did:
            time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy")
    g.add_argument("--oracle", action="store_true")
    g.add_argument("--watch", nargs="+")
    args = ap.parse_args()
    env = make_env()
    if args.watch:
        watch(env, args.watch)
        return
    if args.oracle:
        act, action = oracle_fn(env), "delta"
    else:
        act, meta = policy_fn(args.policy, env.device)
        action = meta.get("action", "absolute")
    t0 = time.perf_counter()
    _, out = evaluate(env, act, action)
    print(json.dumps({**out, "eval_s": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()

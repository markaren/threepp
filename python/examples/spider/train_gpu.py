"""Train the hexapod to walk on the GPU — the whole point of the direct-GPU path.

    python train_gpu.py --envs 4096 --iters 800

K hexapods step in one PhysX direct-GPU scene; obs/reward/reset and the PPO update all
run in torch on the GPU, so an iteration never touches the CPU. A walking gait trains in
minutes (vs ~an hour on the CPU vec env). Saves hexapod_gpu.pt (model + obs-normalizer
stats + metadata), loadable by play_gpu.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from gpu_env import ACT_DIM, JOINT_SCALE, OBS_DIM, HexapodGpuVecEnv
from gpu_ppo import ActorCritic, RunningNorm, compute_gae, save_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--horizon", type=int, default=24, help="rollout steps per iteration")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--log_std_init", type=float, default=-0.5,
                    help="initial action log-std; higher = more exploration (turning is hard to discover)")
    ap.add_argument("--vfcoef", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(_HERE, "hexapod_gpu.pt"))
    ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)

    dev = torch.device("cuda")
    K, T = args.envs, args.horizon
    env = HexapodGpuVecEnv(num_envs=K, device="cuda")
    ac = ActorCritic(OBS_DIM, ACT_DIM, log_std_init=args.log_std_init).to(dev)
    norm = RunningNorm(OBS_DIM, dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    meta = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": (256, 256),
            "joint_scale": JOINT_SCALE.tolist(), "clock_hz": 1.6, "max_yaw": 1.2}

    # Rollout buffers [T, K]
    b_obs = torch.zeros(T, K, OBS_DIM, device=dev)
    b_act = torch.zeros(T, K, ACT_DIM, device=dev)
    b_logp = torch.zeros(T, K, device=dev)
    b_val = torch.zeros(T, K, device=dev)
    b_rew = torch.zeros(T, K, device=dev)
    b_done = torch.zeros(T, K, device=dev)

    # Episode-return tracking (on GPU): accumulate, snapshot on done.
    ep_ret = torch.zeros(K, device=dev)
    ep_len = torch.zeros(K, device=dev)
    recent_ret, recent_len = [], []

    obs = env.reset()
    mb = K * T // args.minibatches
    t_start = time.perf_counter()
    total_steps = 0

    for it in range(1, args.iters + 1):
        # --- rollout ---
        for t in range(T):
            norm.update(obs)
            nobs = norm.norm(obs)
            a, logp, val = ac.act(nobs)
            b_obs[t] = nobs; b_act[t] = a; b_logp[t] = logp; b_val[t] = val
            obs, rew, done = env.step(a)
            b_rew[t] = rew; b_done[t] = done.float()
            ep_ret += rew; ep_len += 1
            d = done.nonzero(as_tuple=False).squeeze(-1)
            if d.numel() > 0:
                recent_ret.extend(ep_ret[d].tolist())
                recent_len.extend(ep_len[d].tolist())
                ep_ret[d] = 0.0; ep_len[d] = 0.0
        total_steps += T * K

        with torch.no_grad():
            last_val = ac.critic(norm.norm(obs)).squeeze(-1)
        adv, ret = compute_gae(b_rew, b_val, b_done, last_val, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten and PPO-update
        f_obs = b_obs.reshape(-1, OBS_DIM)
        f_act = b_act.reshape(-1, ACT_DIM)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = ret.reshape(-1)
        n = f_obs.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                j = idx[s:s + mb]
                newlogp, ent, val = ac.evaluate(f_obs[j], f_act[j])
                # Clamp the log-ratio before exp() so a large early policy shift can't
                # overflow the surrogate to inf (which would NaN the gradients).
                ratio = (newlogp - f_logp[j]).clamp(-20.0, 20.0).exp()
                a1 = ratio * f_adv[j]
                a2 = ratio.clamp(1 - args.clip, 1 + args.clip) * f_adv[j]
                pg = -torch.min(a1, a2).mean()
                vf = (val - f_ret[j]).pow(2).mean()
                loss = pg + args.vfcoef * vf - args.entropy * ent.mean()
                if not torch.isfinite(loss):
                    continue  # skip a bad minibatch rather than poison the weights
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()

        if it <= 3:
            pf = all(torch.isfinite(p).all() for p in ac.parameters())
            print(f"  [diag it{it}] params_finite={pf}  log_std={ac.log_std.mean().item():.2f}  "
                  f"ret[{f_ret.min():.1f},{f_ret.max():.1f}]  adv_std={adv.std():.2f}")

        if it % 10 == 0 or it == 1:
            elapsed = time.perf_counter() - t_start
            sps = total_steps / elapsed
            rr = np.mean(recent_ret[-200:]) if recent_ret else float("nan")
            rl = np.mean(recent_len[-200:]) if recent_len else float("nan")
            print(f"it {it:4d} | ep_ret {rr:7.2f} | ep_len {rl:6.1f} | "
                  f"{sps/1e3:6.1f}k steps/s | {elapsed:6.1f}s")
            recent_ret = recent_ret[-200:]; recent_len = recent_len[-200:]

        if it % args.save_every == 0 or it == args.iters:
            save_policy(args.out, ac, norm, meta)

    save_policy(args.out, ac, norm, meta)
    print("saved policy ->", args.out)


if __name__ == "__main__":
    main()

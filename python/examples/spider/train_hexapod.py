"""Train the hexapod's residual-on-CPG policy on the GPU (owned stack, no stable_baselines3).

    python train_hexapod.py --iters 400

K hexapods walk in one PhysX direct-GPU scene (threepp.rl.GpuSim); the CPG gait + the policy's
residual corrections, the obs/reward/reset, and the PPO update all run in torch on the GPU. The
policy is rewarded for tracking a commanded (forward, turn) velocity while staying upright, so it
sharpens the open-loop gait (faster, straighter, push-resistant). Saves hexapod_policy.pt.
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
from hexapod_gpu_env import ACT_DIM, CONFIG, OBS_DIM, HexapodGpuEnv
from threepp.rl import ActorCritic, RunningNorm, compute_gae, save_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--vfcoef", type=float, default=1.0)
    ap.add_argument("--log_std_init", type=float, default=-1.5)   # small initial residuals
    ap.add_argument("--out", default=os.path.join(_HERE, "hexapod_policy.pt"))
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)

    dev = torch.device("cuda")
    K, T = args.envs, args.horizon
    env = HexapodGpuEnv(num_envs=K, device="cuda")
    ac = ActorCritic(OBS_DIM, ACT_DIM, hidden=(256, 256), log_std_init=args.log_std_init).to(dev)
    norm = RunningNorm(OBS_DIM, dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    meta = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": [256, 256], **CONFIG}

    b_obs = torch.zeros(T, K, OBS_DIM, device=dev)
    b_term = torch.zeros(T, K, OBS_DIM, device=dev)
    b_act = torch.zeros(T, K, ACT_DIM, device=dev)
    b_logp = torch.zeros(T, K, device=dev)
    b_val = torch.zeros(T, K, device=dev)
    b_rew = torch.zeros(T, K, device=dev)
    b_done = torch.zeros(T, K, device=dev)
    b_to = torch.zeros(T, K, device=dev)     # 1.0 where a done was a timeout (truncation), else 0

    ep_ret = torch.zeros(K, device=dev)
    ep_len = torch.zeros(K, device=dev)
    recent_ret, recent_len = [], []

    obs = env.reset()
    mb = K * T // args.minibatches
    t0 = time.perf_counter()
    total = 0
    for it in range(1, args.iters + 1):
        for t in range(T):
            norm.update(obs)
            nobs = norm.norm(obs)
            a, logp, val = ac.act(nobs)
            b_obs[t] = nobs; b_act[t] = a; b_logp[t] = logp; b_val[t] = val
            obs, rew, done, term_obs, timeout = env.step(a)
            b_rew[t] = rew; b_done[t] = done.float(); b_term[t] = term_obs; b_to[t] = timeout.float()
            ep_ret += rew; ep_len += 1
            d = done.nonzero(as_tuple=False).squeeze(-1)
            if d.numel() > 0:
                recent_ret.extend(ep_ret[d].tolist()); recent_len.extend(ep_len[d].tolist())
                ep_ret[d] = 0.0; ep_len[d] = 0.0
        total += T * K

        with torch.no_grad():
            last_val = ac.critic(norm.norm(obs)).squeeze(-1)
            # bootstrap V(terminal) on a TIMEOUT (truncation) but zero it on a real fall (b_to gates it)
            term_val = ac.critic(norm.norm(b_term.reshape(-1, OBS_DIM))).squeeze(-1).reshape(T, K) * b_to
        adv, ret = compute_gae(b_rew, b_val, b_done, last_val, term_val, args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        f_obs = b_obs.reshape(-1, OBS_DIM); f_act = b_act.reshape(-1, ACT_DIM)
        f_logp = b_logp.reshape(-1); f_adv = adv.reshape(-1); f_ret = ret.reshape(-1)
        n = f_obs.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                j = idx[s:s + mb]
                newlogp, ent, val = ac.evaluate(f_obs[j], f_act[j])
                ratio = (newlogp - f_logp[j]).clamp(-20, 20).exp()
                pg = -torch.min(ratio * f_adv[j], ratio.clamp(1 - args.clip, 1 + args.clip) * f_adv[j]).mean()
                vf = (val - f_ret[j]).pow(2).mean()
                loss = pg + args.vfcoef * vf - args.entropy * ent.mean()
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()

        if it % 10 == 0 or it == 1:
            el = time.perf_counter() - t0
            rr = np.mean(recent_ret[-400:]) if recent_ret else float("nan")
            rl = np.mean(recent_len[-400:]) if recent_len else float("nan")
            print(f"it {it:4d} | ep_ret {rr:7.1f} | ep_len {rl:5.0f} | {total/el/1e3:6.1f}k steps/s | {el:5.1f}s")
            recent_ret = recent_ret[-400:]; recent_len = recent_len[-400:]
        if it % 50 == 0 or it == args.iters:
            save_policy(args.out, ac, norm, meta)

    save_policy(args.out, ac, norm, meta)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()

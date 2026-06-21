"""Train the residual policy for the hexapod (Stage 2) with PPO.

    python train.py --steps 1500000 --envs 8        # full run (minutes on a good CPU/GPU)
    python train.py --steps 80000   --envs 1        # quick smoke run

Single env uses DummyVecEnv; >1 uses SubprocVecEnv (each subprocess gets its own
PhysX world — required, since one process allows only one PhysX foundation).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)


def make_env():
    from hexapod_env import HexapodEnv
    return HexapodEnv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "hexapod_policy"))
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    if args.envs <= 1:
        venv = DummyVecEnv([make_env])
    else:
        venv = SubprocVecEnv([make_env for _ in range(args.envs)])
    venv = VecMonitor(venv)  # records episode returns -> rollout/ep_rew_mean

    model = PPO(
        "MlpPolicy", venv, verbose=1, device="auto",
        n_steps=1024, batch_size=256, n_epochs=8,
        gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
        ent_coef=0.0,
        # Start with a small action std so initial residuals barely perturb the
        # already-good gait — the policy improves from ~baseline instead of digging
        # out of a hole.
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.5),
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out)
    venv.close()
    print("saved policy ->", args.out + ".zip")


if __name__ == "__main__":
    main()

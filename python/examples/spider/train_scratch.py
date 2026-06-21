"""Train from-scratch hexapod locomotion (no CPG) with PPO — the slow, finicky
path that residual RL (train_vec.py) was built to avoid. Expect tens of minutes
to hours, and possibly a few reward-tuning runs.

    python train_scratch.py --envs 64 --steps 20000000

Checkpoints land next to --out every few M steps, so you can watch intermediate
policies (play_scratch.py) and the run survives interruption.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "hexapod_scratch"))
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from scratch_env import HexapodScratchVecEnv

    venv = HexapodScratchVecEnv(num_envs=args.envs, num_threads=args.threads)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv, verbose=1, device="auto",
        n_steps=512, batch_size=4096, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.004,
        policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0),
    )
    ckpt = CheckpointCallback(
        save_freq=max(1, 4_000_000 // args.envs),  # ~ every 4M env steps
        save_path=_HERE, name_prefix="hexapod_scratch_ck",
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=args.steps, callback=ckpt, progress_bar=False)
    model.save(args.out)
    venv.save(args.out + "_vecnorm.pkl")
    print("saved policy ->", args.out + ".zip")


if __name__ == "__main__":
    main()

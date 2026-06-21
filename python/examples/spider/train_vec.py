"""Train the residual policy on the vectorized (single-scene) hexapod env (Stage 3).

    python train_vec.py --envs 64 --steps 4000000

All K robots live in one PhysX scene and step together, so this is ~7x the
throughput of train.py's SubprocVecEnv — long runs that took ~an hour now take
minutes, which is also enough training for the policy to start beating the gait.
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
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "hexapod_policy"))
    args = ap.parse_args()

    from stable_baselines3 import PPO

    from hexapod_vec_env import HexapodVecEnv

    venv = HexapodVecEnv(num_envs=args.envs, num_threads=args.threads)
    model = PPO(
        "MlpPolicy", venv, verbose=1, device="auto",
        n_steps=512, batch_size=2048, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.5),
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out)
    print("saved policy ->", args.out + ".zip")


if __name__ == "__main__":
    main()

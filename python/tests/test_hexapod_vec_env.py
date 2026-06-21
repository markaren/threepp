"""Smoke test for the vectorized (single-scene) hexapod env (Stage 3).

Verifies K robots step together in one world and return batched obs/rew/done +
episode infos. Skips without stable-baselines3 / gymnasium / the PhysX backend.
"""
import os
import sys

import numpy as np
import pytest

import threepp as tp

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples", "spider"))

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")


def test_vec_env_batched_step():
    from hexapod_vec_env import HexapodVecEnv
    K = 4
    env = HexapodVecEnv(num_envs=K, num_threads=2, episode_s=2.0, perturb=False)
    try:
        obs = env.reset()
        assert obs.shape == (K, 34) and obs.dtype == np.float32 and np.isfinite(obs).all()

        finished = 0
        for _ in range(130):  # > 2 episodes of 60 steps
            env.step_async(np.zeros((K, 12), np.float32))  # gait baseline
            obs, rews, dones, infos = env.step_wait()
            assert obs.shape == (K, 34) and np.isfinite(obs).all()
            assert rews.shape == (K,) and dones.shape == (K,) and np.isfinite(rews).all()
            for d, inf in zip(dones, infos):
                if d:
                    assert "episode" in inf and "terminal_observation" in inf
                    finished += 1
        assert finished >= K, "every env should finish at least one short episode"
    finally:
        env.close()

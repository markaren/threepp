"""Smoke test for the residual-RL environment (Stage 2). No training — just that
the gym env steps, shapes/values are sane, and episodes reset.

Skips without gymnasium or the PhysX backend.
"""
import os
import sys

import numpy as np
import pytest

import threepp as tp

pytest.importorskip("gymnasium")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples", "spider"))

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")


def test_env_steps_and_resets():
    from hexapod_env import HexapodEnv
    env = HexapodEnv(episode_s=2.0)
    assert env.observation_space.shape == (34,)
    assert env.action_space.shape == (12,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (34,) and obs.dtype == np.float32 and np.isfinite(obs).all()

    resets = 0
    for _ in range(150):
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == (34,) and np.isfinite(obs).all()
        assert np.isfinite(r)
        if terminated or truncated:
            obs, _ = env.reset()
            resets += 1
    assert resets >= 1, "episodes should terminate/truncate and reset within 150 steps"


def test_zero_action_is_the_gait():
    # With zero residuals the env is exactly the open-loop gait, and on flat ground
    # (no pushes) it should not fall over a short rollout.
    from hexapod_env import HexapodEnv
    env = HexapodEnv(episode_s=4.0, perturb=False)
    env.reset(seed=1)
    fell = False
    for _ in range(120):
        _, _, terminated, truncated, _ = env.step(np.zeros(12, np.float32))
        fell = fell or terminated
        if terminated or truncated:
            break
    assert not fell, "open-loop gait should stay upright on flat ground"

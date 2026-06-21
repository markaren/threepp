"""Vectorized hexapod env: K robots in ONE PhysX scene, stepped together (Stage 3).

Following the pyrl-branch pattern — one PxScene, many instances, a single
simulate()/fetchResults() per control step — all K hexapods advance in one
world.step(). Measured ~35x the throughput of running K separate processes
(SubprocVecEnv). Implements SB3's VecEnv directly, so PPO trains on it with no
multiprocessing.

  python train_vec.py --envs 64 --steps 4000000     # minutes, not hours
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from hexapod import Hexapod
from hexapod_env import RESIDUAL_SCALE, START, compute_reward, make_observation

SPACING = 2.5  # metres between adjacent robots — far enough to never collide


class HexapodVecEnv(VecEnv):
    def __init__(self, num_envs=64, control_hz=30, episode_s=8.0, command_hold_s=2.5,
                 perturb=True, push_prob=0.015, push_mag=(3.0, 11.0), num_threads=8, seed=0):
        self.control_dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.command_hold = int(command_hold_s * control_hz)
        self.perturb = perturb
        self.push_prob = push_prob
        self.push_mag = push_mag
        self.rng = np.random.default_rng(seed)

        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), num_threads=num_threads)
        ground = tp.Mesh(tp.BoxGeometry(SPACING * num_envs + 40, 1, 60), tp.MeshStandardMaterial())
        ground.position.set(SPACING * num_envs * 0.5, -0.5, 0.0)
        self.world.add_static(ground)
        self.bases = [(i * SPACING, START[1], 0.0) for i in range(num_envs)]
        self.spiders = [Hexapod(self.world, position=b) for b in self.bases]

        super().__init__(num_envs,
                         spaces.Box(-np.inf, np.inf, (34,), np.float32),
                         spaces.Box(-1.0, 1.0, (12,), np.float32))

        K = num_envs
        self.command = np.zeros((K, 2), np.float32)
        self.steps = np.zeros(K, np.int32)
        self.prev = np.zeros((K, 2), np.float32)
        self.prev_yaw = np.zeros(K, np.float32)
        self.vel = np.zeros((K, 2), np.float32)
        self.yawrate = np.zeros(K, np.float32)
        self.ep_ret = np.zeros(K, np.float64)
        self.ep_len = np.zeros(K, np.int32)
        self._actions = np.zeros((K, 12), np.float32)

    # --- helpers ---------------------------------------------------------
    def _sample_command(self):
        return np.array([self.rng.uniform(-0.4, 1.0), self.rng.uniform(-1.0, 1.0)], np.float32)

    def _teleport_reset(self, i):
        self.spiders[i].reset(self.bases[i])
        self.command[i] = self._sample_command()
        self.steps[i] = 0
        self.ep_ret[i] = 0.0
        self.ep_len[i] = 0

    def _post(self, i):
        p = self.spiders[i].position
        self.prev[i] = (p.x, p.z)
        self.prev_yaw[i] = self.spiders[i].yaw
        self.vel[i] = 0.0
        self.yawrate[i] = 0.0

    def _measure(self, i):
        s = self.spiders[i]
        p = s.position
        v = np.array([(p.x - self.prev[i][0]) / self.control_dt,
                      (p.z - self.prev[i][1]) / self.control_dt], np.float32)
        self.vel[i] = 0.8 * self.vel[i] + 0.2 * v
        dy = s.yaw - self.prev_yaw[i]
        dy = (dy + math.pi) % (2 * math.pi) - math.pi
        self.yawrate[i] = 0.8 * self.yawrate[i] + 0.2 * (dy / self.control_dt)
        self.prev[i] = (p.x, p.z)
        self.prev_yaw[i] = s.yaw

    def _obs(self, i):
        return make_observation(self.spiders[i], self.vel[i], self.yawrate[i], self.command[i])

    # --- VecEnv API ------------------------------------------------------
    def reset(self):
        for i in range(self.num_envs):
            self._teleport_reset(i)
        for _ in range(6):  # settle the whole batch into a stand
            for i in range(self.num_envs):
                self.spiders[i].set_command(0.0, 0.0)
                self.spiders[i].update(self.control_dt, None)
            self.world.step(self.control_dt)
        for i in range(self.num_envs):
            self._post(i)
        return np.stack([self._obs(i) for i in range(self.num_envs)]).astype(np.float32)

    def step_async(self, actions):
        self._actions = np.clip(np.asarray(actions, np.float32), -1.0, 1.0)

    def step_wait(self):
        K = self.num_envs
        for i in range(K):
            s = self.spiders[i]
            s.set_command(float(self.command[i][0]), float(self.command[i][1]))
            s.update(self.control_dt, (self._actions[i] * RESIDUAL_SCALE).tolist())
            if self.perturb and self.rng.random() < self.push_prob:
                mag = float(self.rng.uniform(*self.push_mag))
                ang = float(self.rng.uniform(0.0, 2 * math.pi))
                s.chassis.add_impulse(tp.Vector3(mag * math.cos(ang), 0.0, mag * math.sin(ang)))
        self.world.step(self.control_dt)   # one batched step for ALL K robots

        self.steps += 1
        rews = np.zeros(K, np.float32)
        dones = np.zeros(K, bool)
        infos = [{} for _ in range(K)]
        for i in range(K):
            self._measure(i)
            if self.steps[i] % self.command_hold == 0:
                self.command[i] = self._sample_command()
            r, fell = compute_reward(self.spiders[i], self.vel[i], self.yawrate[i],
                                     self.command[i], self._actions[i])
            if fell:
                r -= 5.0
            done = bool(fell or self.steps[i] >= self.max_steps)
            rews[i] = r
            dones[i] = done
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            if done:
                infos[i]["episode"] = {"r": float(self.ep_ret[i]), "l": int(self.ep_len[i])}
                infos[i]["terminal_observation"] = self._obs(i)
                self._teleport_reset(i)  # chassis pose is set directly -> obs is valid pre-step
                self._post(i)
        obs = np.stack([self._obs(i) for i in range(K)]).astype(np.float32)
        return obs, rews, dones, infos

    def close(self):
        pass

    # Minimal stubs for the rest of the VecEnv interface (PPO doesn't need them).
    def get_attr(self, attr_name, indices=None):
        val = getattr(self, attr_name, None)
        return [val] * self.num_envs if indices is None else [val for _ in self._idx(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._idx(indices)]

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
